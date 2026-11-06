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

def topfrac(df, score, frac):
    d = df.assign(s=score).sort_values("s", ascending=False)
    cw = d.sample_weight.cumsum() / d.sample_weight.sum()
    top = d[cw <= frac]
    prec = np.average(top[TARGET], weights=top.sample_weight)
    base = np.average(df[TARGET], weights=df.sample_weight)
    rec = (top[TARGET] * top.sample_weight).sum() / (df[TARGET] * df.sample_weight).sum()
    return {"top_frac": frac, "precision": float(prec), "lift": float(prec / base), "recall_rows": float(rec), "false_alarm_share": float(1 - prec)}, top

def episodes(top, df):
    pos = df[df[TARGET] == 1]
    keys = pos[["symbol", "next_onset_time"]].drop_duplicates()
    hit = top[top[TARGET] == 1].groupby(["symbol", "next_onset_time"]).hours_to_onset.max().reset_index()
    return {"episodes_in_test": int(len(keys)), "episodes_caught": int(len(hit)),
            "catch_rate": float(len(hit) / max(len(keys), 1)),
            "median_lead_hours": float(hit.hours_to_onset.median()) if len(hit) else None}

def main():
    t0 = time.time(); df = fetch(); print("rows", len(df), "fetch s", int(time.time() - t0))
    df["open_time"] = pd.to_datetime(df.open_time, utc=True)
    for c in BASE + REL + ["sample_weight"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["y_10_24", "y_25_48", "y_40_72", "y_any", "market_wide", "in_cooldown"]: df[c] = df[c].astype(int)
    df = df.dropna(subset=["rel_rv24"]).reset_index(drop=True)
    split = df.open_time.quantile(0.7)
    tr = df[df.open_time < split - pd.Timedelta(hours=72)]; te = df[df.open_time >= split]
    out = {"target": TARGET, "split_time": str(split), "train_rows": len(tr), "test_rows": len(te), "models": {}}
    sets = {"base": BASE, "base+relative": BASE + REL}
    for fs_name, feats in sets.items():
        models = {"logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
                  "gbm": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=7)}
        for mn, m in models.items():
            m.fit(tr[feats], tr[TARGET], **({"logisticregression__sample_weight": tr.sample_weight} if mn == "logistic" else {"sample_weight": tr.sample_weight}))
            s = m.predict_proba(te[feats])[:, 1]
            res = {}
            for seg, mask in {"all": np.ones(len(te), bool), "market_wide_false": (te.market_wide == 0).values,
                              "market_wide_true": (te.market_wide == 1).values, "not_in_cooldown": (te.in_cooldown == 0).values}.items():
                sub = te[mask]; ss = s[mask]
                if sub[TARGET].sum() < 20: continue
                r = wmetrics(sub[TARGET], ss, sub.sample_weight)
                for f in (0.01, 0.05, 0.10):
                    t, top = topfrac(sub, ss, f); r[f"top{int(f*100)}pct"] = {**t, **episodes(top, sub)}
                res[seg] = r
            out["models"][f"{mn}|{fs_name}"] = res
            print(mn, fs_name, json.dumps(res.get("all", {}))[:300])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_models.json").write_text(json.dumps(out, indent=2))
    lines = ["# Baseline models (indicators only, not trade signals)", f"target={TARGET}, split={split}", ""]
    lines.append("| model|features | segment | AUC | PR-AUC | base | top5% lift | top5% precision | episodes caught | median lead h |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in out["models"].items():
        for seg, r in v.items():
            t = r["top5pct"]; lines.append(f"| {k} | {seg} | {r['auc']:.3f} | {r['pr_auc']:.3f} | {r['base_rate']:.3f} | {t['lift']:.2f} | {t['precision']:.3f} | {t['episodes_caught']}/{t['episodes_in_test']} | {t['median_lead_hours']} |")
    Path("reports/baseline_models.md").write_text("\n".join(lines))
main()
