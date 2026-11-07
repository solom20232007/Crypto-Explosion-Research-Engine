"""Do lagged futures-context features (OI change, funding, long/short + taker ratios) add ranking power?

Decision at hour t uses futures data of the PREVIOUS complete UTC day only (what an archive-based live feed could deliver).
Compares base vs base+futures on the identical row subset, with a paired cluster bootstrap (symbol-week) on the AUC difference.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, requests
sys.path.insert(0, str(Path(__file__).parent))
import train_baseline as tb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

FUT = ["oi_chg_1d", "oi_chg_3d", "funding_mean", "funding_sum3", "ls_ratio_mean", "top_ls_ratio_mean", "taker_ratio_mean"]

def futures():
    rows, off = [], 0
    while True:
        r = requests.get(f"{tb.URL}/rest/v1/futures_daily_v1", headers={**{"apikey": tb.KEY, "Authorization": f"Bearer {tb.KEY}"}, "Range": f"{off}-{off+999}"},
                         params={"select": "symbol,day,oi_value_end,oi_chg_1d,ls_ratio_mean,top_ls_ratio_mean,taker_ratio_mean,funding_mean,funding_sum", "order": "symbol,day"}, timeout=120)
        r.raise_for_status(); b = r.json(); rows += b
        if len(b) < 1000: break
        off += 1000
    f = pd.DataFrame(rows); f["day"] = pd.to_datetime(f.day, utc=True)
    for c in f.columns[2:]: f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.sort_values(["symbol", "day"])
    f["oi_chg_3d"] = f.groupby("symbol").oi_value_end.pct_change(3)
    f["funding_sum3"] = f.groupby("symbol").funding_sum.transform(lambda s: s.rolling(3, min_periods=2).sum())
    f["day"] = f["day"] + pd.Timedelta(days=1)  # available from the next UTC day -> join on the decision day
    f[FUT] = f[FUT].replace([np.inf, -np.inf], np.nan).clip(-1e3, 1e3)
    return f[["symbol", "day"] + FUT]

def model(kind):
    if kind == "logistic": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=7)

def fit(m, X, y, w):
    if hasattr(m, "steps"): m.fit(X, y, logisticregression__sample_weight=w)
    else: m.fit(X, y, sample_weight=w)
    return m

def main():
    df = tb.fetch(); df["open_time"] = pd.to_datetime(df.open_time, utc=True)
    for c in tb.BASE + tb.REL + ["sample_weight"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["y_10_24", "y_25_48", "y_40_72", "y_any", "market_wide", "in_cooldown"]: df[c] = df[c].astype(int)
    df["day"] = df.open_time.dt.floor("D")
    df = df.merge(futures(), on=["symbol", "day"], how="left")
    df[tb.BASE] = df[tb.BASE].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["oi_chg_1d", "rel_rv24"]).reset_index(drop=True)
    print("rows with futures context", len(df), "symbols", df.symbol.nunique())
    split = df.open_time.quantile(0.7); tr = df[df.open_time < split - pd.Timedelta(hours=72)]; te = df[df.open_time >= split].copy()
    te["cluster"] = te.symbol + te.open_time.dt.strftime("%G-%V")
    rng = np.random.default_rng(7); out = {"rows": len(df), "train": len(tr), "test": len(te), "results": {}}
    lines = ["| target | model | segment | AUC base | AUC +futures | diff (95% CI) | top5% lift base | top5% lift +fut |", "|---|---|---|---|---|---|---|---|"]
    for target in ["y_10_24", "y_25_48"]:
        for kind in ["logistic", "gbm"]:
            s = {}
            for name, feats in {"base": tb.BASE, "fut": tb.BASE + FUT}.items():
                s[name] = fit(model(kind), tr[feats], tr[target], tr.sample_weight).predict_proba(te[feats])[:, 1]
            for seg, mask in {"all": np.ones(len(te), bool), "market_wide_false": (te.market_wide == 0).values}.items():
                sub = te[mask].reset_index(drop=True); y = sub[target].values; w = sub.sample_weight.values
                if y.sum() < 50: continue
                a0 = roc_auc_score(y, s["base"][mask], sample_weight=w); a1 = roc_auc_score(y, s["fut"][mask], sample_weight=w)
                groups = sub.groupby("cluster").indices; keys = list(groups); diffs = []
                for _ in range(150):
                    idx = np.concatenate([groups[keys[i]] for i in rng.choice(len(keys), len(keys))])
                    yy = y[idx]
                    if yy.min() == yy.max(): continue
                    diffs.append(roc_auc_score(yy, s["fut"][mask][idx], sample_weight=w[idx]) - roc_auc_score(yy, s["base"][mask][idx], sample_weight=w[idx]))
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                l0 = tb.topfrac(sub, s["base"][mask], 0.05, target)[0]["lift"]; l1 = tb.topfrac(sub, s["fut"][mask], 0.05, target)[0]["lift"]
                out["results"][f"{target}|{kind}|{seg}"] = {"auc_base": a0, "auc_fut": a1, "diff": a1 - a0, "ci": [lo, hi], "lift_base": l0, "lift_fut": l1}
                lines.append(f"| {target} | {kind} | {seg} | {a0:.3f} | {a1:.3f} | {a1-a0:+.3f} ({lo:+.3f}, {hi:+.3f}) | {l0:.2f} | {l1:.2f} |")
    Path("reports").mkdir(exist_ok=True); Path("reports/futures_eval.json").write_text(json.dumps(out, indent=2)); print("\n".join(lines))
main()
