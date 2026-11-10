"""Does Ethereum on-chain activity add to the price-based model? (sampled days, tokens with a high-confidence Ethereum contract). Research only."""
import os
import numpy as np, pandas as pd, requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]; H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
BASE = ["ret_1h","ret_3h","ret_6h","ret_12h","ret_24h","ret_48h","ret_72h","ret_168h","vol_ratio_24","vol_ratio_168","range_24h","range_168h","rv24","rv168",
        "vol_compression","close_vs_sma20","close_vs_sma50","dist_from_high_168","dist_from_low_168","btc_ret_24h","btc_ret_72h","breadth_24h","pump_count_24h","hour_utc","dow"]

def fetch(table, select, order, extra=None):
    rows, off = [], 0
    while True:
        r = requests.get(f"{URL}/rest/v1/{table}", headers={**H, "Range": f"{off}-{off+999}"}, params={"select": select, "order": order, **(extra or {})}, timeout=120); r.raise_for_status()
        b = r.json(); rows += b
        if len(b) < 1000: return pd.DataFrame(rows)
        off += 1000

def fit(tr, feats, t):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)); m.fit(tr[feats], tr[t].astype(int)); return m

def main():
    oc = fetch("onchain_hist_sample_v1", "*", "symbol,bar_time"); oc["open_time"] = pd.to_datetime(oc.bar_time, utc=True)
    hd = fetch("history_daily_v1", "symbol,open_time,already_moved,y_10_24,y_25_48,y_40_72," + ",".join(BASE), "symbol,open_time")
    hd["open_time"] = pd.to_datetime(hd.open_time, utc=True)
    for c in BASE: hd[c] = pd.to_numeric(hd[c], errors="coerce")
    df = hd.merge(oc, on=["symbol", "open_time"], how="inner"); df = df[~df.already_moved.astype(bool)].copy()
    df = df[df.complete.fillna(False)]
    print("on-chain rows", len(oc), "joined (not-already-moved, complete)", len(df), "symbols", df.symbol.nunique(), df.open_time.min(), df.open_time.max())
    df["ratio"] = ((df.tc_recent6 / 6) / ((df.tc_prior24 / 24).clip(lower=0.25))).clip(upper=100)
    df["lratio"] = np.log1p(df.ratio); df["whale_flag"] = (df.whale_n_6 > 0).astype(int); df["lwhale"] = np.log1p(df.whale_usd_6)
    df["vratio"] = np.log1p(((df.vol_recent6_usd / 6) / ((df.vol_prior24_usd / 24).clip(lower=50))).clip(upper=1000))
    df["lrcv"] = np.log1p(df.rcv_recent6)
    NEW = ["lratio", "whale_flag", "lwhale", "vratio", "lrcv"]
    df["blk"] = (df.open_time >= "2024-11-01").astype(int) + (df.open_time >= "2025-11-01").astype(int)
    print(df.groupby("blk").agg(n=("symbol", "size"), b10=("y_10_24", "mean"), b25=("y_25_48", "mean"), b40=("y_40_72", "mean")).round(4).to_string())
    print("\\nمعدل الانفجار حسب نشاط التحويلات (قبل الشمعة):")
    for name, mask in (("نشاط عادي (<1.5×)", df.ratio < 1.5), ("1.5–3×", (df.ratio >= 1.5) & (df.ratio < 3)), ("3–10×", (df.ratio >= 3) & (df.ratio < 10)), ("≥10×", df.ratio >= 10), ("لا تحويلات حيتان", df.whale_flag == 0), ("تحويل حوت ≥1", df.whale_flag == 1)):
        g = df[mask]
        if len(g) > 20: print(f"{name:22s} n={len(g):6d}  y10={g.y_10_24.mean():.3f}  y25={g.y_25_48.mean():.3f}  y40={g.y_40_72.mean():.3f}")
    for t in ("y_10_24", "y_25_48", "y_40_72"):
        P = {k: pd.Series(index=df.index, dtype=float) for k in ("base", "plus", "onchain_only")}
        for b in (0, 1, 2):
            te = df[df.blk == b]
            if len(te) < 200: continue
            lo, hi = te.open_time.min(), te.open_time.max(); emb = pd.Timedelta(days=3)
            tr = df[(df.blk != b) & ~((df.open_time > lo - emb) & (df.open_time < hi + emb))]
            P["base"].loc[te.index] = fit(tr, BASE, t).predict_proba(te[BASE])[:, 1]; P["plus"].loc[te.index] = fit(tr, BASE + NEW, t).predict_proba(te[BASE + NEW])[:, 1]
            P["onchain_only"].loc[te.index] = fit(tr, NEW, t).predict_proba(te[NEW])[:, 1]
        print(f"\\n=== {t}: leave-one-year-out على عملات إيثيريوم فقط ===")
        for k, lab in (("base", "السعر فقط (الحالي)"), ("plus", "السعر + التحويلات"), ("onchain_only", "التحويلات فقط")):
            ok = P[k].notna(); y = df.loc[ok, t].astype(int); p = P[k][ok]
            auc = roc_auc_score(y, p); d = df.loc[ok].assign(p=p, y=y); top = d.sort_values("p", ascending=False).groupby("open_time").head(3)
            print(f"{lab:22s} AUC={auc:.3f}  top3 يوميًا: دقة={top.y.mean():.3f} (المعدل العام {y.mean():.3f}, ضعف {top.y.mean()/y.mean():.2f})")
        if t == "y_10_24":
            for b in (0, 1, 2):
                ok = (df.blk == b) & P["base"].notna()
                if ok.sum() > 200: print(f"  سنة {b+1}: AUC سعر={roc_auc_score(df.loc[ok,t].astype(int), P['base'][ok]):.3f}  سعر+تحويلات={roc_auc_score(df.loc[ok,t].astype(int), P['plus'][ok]):.3f}  تحويلات فقط={roc_auc_score(df.loc[ok,t].astype(int), P['onchain_only'][ok]):.3f}")
main()
