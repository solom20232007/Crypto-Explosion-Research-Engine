"""Baseline models on prebreakout_samples_v1 (research only: indicators, not trade signals).

Reads the sample table from Supabase, trains logistic regression and gradient boosting on a
temporal split (72h embargo), and reports ranking quality overall, on market_wide=false,
and at episode level (episodes caught, lead time, false-alarm share). Writes reports/baseline_models.{json,md}.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
BASE = ["ret_1h","ret_3h","ret_6h","ret_12h","ret_24h","ret_48h","ret_72h","ret_168h","vol_ratio_24","vol_ratio_168",
        "range_24h","range_168h","rv24","rv168","vol_compression","close_vs_sma20","close_vs_sma50","dist_from_high_168",
        "dist_from_low_168","btc_ret_24h","btc_ret_72h","breadth_24h","pump_count_24h","hour_utc","dow"]
REL = ["base_rv_30d","rel_rv24","rel_range24","rel_vol24","rel_ret24","rel_ret72"]
TARGET = os.environ.get("TARGET", "y_10_24")

def fetch() -> pd.DataFrame:
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
    cols = "symbol,open_time,sample_weight,market_wide,in_cooldown,next_onset_time,hours_to_onset,y_10_24,y_25_48,y_40_72,y_any," + ",".join(BASE + REL)
    rows, off, page = [], 0, 1000
    while True:
        r = requests.get(f"{URL}/rest/v1/prebreakout_samples_v1", headers={**h, "Range": f"{off}-{off+page-1}"},
                         params={"select": cols, "order": "symbol,open_time"}, timeout=120)
        r.raise_for_status(); b = r.json()
        rows += b
        if len(b) < page: break
        off += page
    return pd.DataFrame(rows)

def wmetrics(y, s, w):
    return {"auc": float(roc_auc_score(y, s, sample_weight=w)), "pr_auc": float(average_precision_score(y, s, sample_weight=w)),
            "base_rate": float(np.average(y, weights=w))}

def topfrac(df, score, frac, target):
    d = df.assign(s=score).sort_values("s", ascending=False)
    cw = d.sample_weight.cumsum() / d.sample_weight.sum()
    top = d[cw <= frac]
    prec = np.average(top[target], weights=top.sample_weight)
    base = np.average(df[target], weights=df.sample_weight)
    rec = (top[target] * top.sample_weight).sum() / (df[target] * df.sample_weight).sum()
    return {"top_frac": frac, "precision": float(prec), "lift": float(prec / base), "recall_rows": float(rec), "false_alarm_share": float(1 - prec)}, top

def episodes(top, df, target):
    """Label-episodes: maximal runs of positive rows per symbol (gap <= 72h merged).
    Caught = at least one flagged row inside the run. delay = hours from run start to first flag
    (0 = flagged at the earliest moment the move was within the label horizon)."""
    pos = df[df[target] == 1][["symbol", "open_time"]].sort_values(["symbol", "open_time"]).copy()
    if pos.empty: return {}
    gap = pos.groupby("symbol").open_time.diff() > pd.Timedelta(hours=72)
    pos["run"] = (gap | pos.symbol.ne(pos.symbol.shift())).cumsum()
    start = pos.groupby("run").open_time.min()
    fl = top[top[target] == 1][["symbol", "open_time"]].merge(pos, on=["symbol", "open_time"])
    first = fl.groupby("run").open_time.min()
    delay = ((first - start.loc[first.index]).dt.total_seconds() / 3600)
    return {"episodes_in_test": int(len(start)), "episodes_caught": int(len(first)),
            "catch_rate": float(len(first) / len(start)),
            "median_delay_from_run_start_h": float(delay.median()) if len(delay) else None}

def daily_topk(sub, score, target, k=5, universe=117):
    """One decision per symbol per day (00:00 UTC row). Listing the top-k of `universe` symbols per day is
    approximated by the top k/universe weighted share of the pooled 00:00 rows (per-day truncation on
    weight-12 negatives would bias precision upward)."""
    m = (sub.hour_utc == 0).values
    d = sub[m]; ss = score[m]
    if d[target].sum() < 20: return {}
    t, top = topfrac(d, ss, k / universe, target)
    return {"k_per_day": k, "decision_rows_weighted": float(d.sample_weight.sum()), "precision": t["precision"], "lift": t["lift"], **episodes(top, d, target)}, top

def run_target(df, target):
    split = df.open_time.quantile(0.7)
    tr = df[df.open_time < split - pd.Timedelta(hours=72)]; te = df[df.open_time >= split].copy()
    te["day"] = te.open_time.dt.floor("D")
    out = {"target": target, "split_time": str(split), "train_rows": len(tr), "test_rows": len(te), "models": {}}
    keep = {}
    for fs_name, feats in {"base": BASE, "base+relative": BASE + REL}.items():
        models = {"logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
                  "gbm": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=7)}
        for mn, m in models.items():
            m.fit(tr[feats], tr[target], **({"logisticregression__sample_weight": tr.sample_weight} if mn == "logistic" else {"sample_weight": tr.sample_weight}))
            s = m.predict_proba(te[feats])[:, 1]; res = {}
            for seg, mask in {"all": np.ones(len(te), bool), "market_wide_false": (te.market_wide == 0).values,
                              "market_wide_true": (te.market_wide == 1).values}.items():
                sub = te[mask]; ss = s[mask]
                if sub[target].sum() < 20: continue
                r = wmetrics(sub[target], ss, sub.sample_weight)
                for f in (0.01, 0.05):
                    t, top = topfrac(sub, ss, f, target); r[f"top{int(f*100)}pct"] = {**t, **episodes(top, sub, target)}
                dt = daily_topk(sub, ss, target, 5)
                r["daily_top5"] = dt[0] if dt else {}
                res[seg] = r
            out["models"][f"{mn}|{fs_name}"] = res
            keep[f"{mn}|{fs_name}"] = s
    return out, te, keep

def main():
    t0 = time.time(); df = fetch(); print("rows", len(df), "fetch s", int(time.time() - t0))
    df["open_time"] = pd.to_datetime(df.open_time, utc=True)
    for c in BASE + REL + ["sample_weight"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["y_10_24", "y_25_48", "y_40_72", "y_any", "market_wide", "in_cooldown"]: df[c] = df[c].astype(int)
    df = df.dropna(subset=["rel_rv24"]).reset_index(drop=True)
    Path("reports").mkdir(exist_ok=True)
    results, lines = {}, ["# Baseline models (research indicators, not trade signals)", ""]
    lines.append("| target | model|features | segment | AUC | base | top5% lift | top5% precision | episodes caught | median delay h | daily top-5 precision (00:00 rows) | daily lift |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for target in ["y_10_24", "y_25_48", "y_40_72"]:
        out, te, keep = run_target(df, target); results[target] = out
        for k, v in out["models"].items():
            for seg, r in v.items():
                t = r["top5pct"]; d = r["daily_top5"]
                lines.append(f"| {target} | {k} | {seg} | {r['auc']:.3f} | {r['base_rate']:.3f} | {t['lift']:.2f} | {t['precision']:.3f} | {t['episodes_caught']}/{t['episodes_in_test']} | {t['median_delay_from_run_start_h']} | {d.get('precision',0):.3f} | {d.get('lift',0):.2f} |")
    Path("reports/baseline_models.json").write_text(json.dumps(results, indent=2))
    Path("reports/baseline_models.md").write_text("\n".join(lines))
    print("\n".join(lines))
main()
