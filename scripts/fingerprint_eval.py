"""Fingerprint test (research only): do pre-explosion "fingerprints" from analyst reports add ranking power, and what do they look like against a CONTROL group?\n\nFingerprints (all point-in-time at the 00:00 UTC decision bar, from hourly klines): volume burst while price is flat, net taker-buy share (spot CVD proxy),\nand small-liquidity. Compared with the production BASE model using leave-one-year-out + week-block bootstrap.\n\n(old docstring follows)\nRealistic test

The training label uses the max HIGH over the next 24-72h, which nobody can capture. Here every pick is traded like a spot buyer would:
  * entry at the CLOSE of the decision bar (ideal, k=0) or k hours later (k=8: recommendation reaches the user ~09:00 UTC; k=12 later still);
  * exit at the CLOSE H hours after entry (H=24/48), or earlier at a -8% stop (filled at the stop price = optimistic);
  * costs per side: 0.10% fee + slippage by prior-24h liquidity (<200k: 0.5%, 200k-1M: 0.3%, >=1M: 0.15% USDT).
Scores come from leave-one-year-out logistic models trained on history_daily_v1 (72h embargo), exactly like event_level_eval.py.
Edge = top-K net return minus the average net return of ALL eligible coins that day (what a blind buyer gets). Week-block bootstrap CIs.
"""
import io, os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]; H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
BASE = ["ret_1h","ret_3h","ret_6h","ret_12h","ret_24h","ret_48h","ret_72h","ret_168h","vol_ratio_24","vol_ratio_168","range_24h","range_168h","rv24","rv168",
        "vol_compression","close_vs_sma20","close_vs_sma50","dist_from_high_168","dist_from_low_168","btc_ret_24h","btc_ret_72h","breadth_24h","pump_count_24h","hour_utc","dow"]
ARCH = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{m}.zip"
RNG = np.random.default_rng(11)
FEE, STOP = 0.001, -0.08
DELAYS, HOLDS = (0, 8, 12), (24, 48)

def fetch():
    rows, off = [], 0
    sel = "symbol,open_time,already_moved,market_wide,y_10_24,y_40_72," + ",".join(BASE)
    while True:
        r = requests.get(f"{URL}/rest/v1/history_daily_v1", headers={**H, "Range": f"{off}-{off+999}"}, params={"select": sel, "order": "symbol,open_time"}, timeout=120); r.raise_for_status()
        b = r.json(); rows += b
        if len(b) < 1000: return pd.DataFrame(rows)
        off += 1000

def month_file(a):
    s, m = a
    for _ in range(3):
        try:
            r = requests.get(ARCH.format(s=s, m=m), timeout=60)
            if r.status_code == 404: return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content)); raw = z.read(z.namelist()[0]).decode()
            df = pd.read_csv(io.StringIO(raw), header=None, usecols=[0, 4, 7, 10], names=["t", "close", "qv", "tbq"])
            df = df[pd.to_numeric(df.t, errors="coerce").notna()].astype({"t": "int64"})
            df["t"] = np.where(df.t > 10**14, df.t // 1000, df.t); return df
        except Exception as e:
            err = e
    print("fail", s, m, err, file=sys.stderr); return None

NEW = ["burst_flat_24", "n_burst_flat_24", "tb_ratio_24", "tb_delta", "cvd_norm_72", "log_qv24", "log_qv7d"]

def kfeat(a):
    s, months, times = a
    parts = [p for p in (month_file((s, m)) for m in months) if p is not None]
    if not parts: return None
    d = pd.concat(parts).drop_duplicates("t").sort_values("t")
    d.index = pd.to_datetime(d.t, unit="ms", utc=True); d = d.drop(columns="t")
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h", tz="UTC"))
    c, qv, tb = d.close, d.qv.fillna(0), d.tbq.fillna(0)
    r1 = (c / c.shift(1) - 1).abs(); vr = qv / qv.shift(1).rolling(24).mean().replace(0, np.nan)
    flat = r1 < 0.015
    f = pd.DataFrame(index=d.index)
    f["burst_flat_24"] = vr.where(flat, 0).rolling(24).max()
    f["n_burst_flat_24"] = (flat & (vr >= 3)).astype(float).rolling(24).sum()
    q24, q168, t24, t168 = qv.rolling(24).sum(), qv.rolling(168).sum(), tb.rolling(24).sum(), tb.rolling(168).sum()
    f["tb_ratio_24"] = t24 / q24.replace(0, np.nan); f["tb_delta"] = f.tb_ratio_24 - t168 / q168.replace(0, np.nan)
    f["cvd_norm_72"] = (2 * tb - qv).rolling(72).sum() / qv.rolling(72).sum().replace(0, np.nan)
    f["log_qv24"] = np.log1p(q24); f["log_qv7d"] = np.log1p(q168 / 7)
    f["qv24_raw"] = q24
    f = f.loc[f.index.isin(times)].copy(); f["symbol"] = s; return f.reset_index(names="open_time")

def loyo(d, cols, tgt):
    p = pd.Series(np.nan, index=d.index)
    for b in (0, 1, 2):
        te = d[d.blk == b]; lo, hi = te.open_time.min(), te.open_time.max(); emb = pd.Timedelta(days=3)
        tr = d[(d.blk != b) & ~((d.open_time > lo - emb) & (d.open_time < hi + emb))]
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)); m.fit(tr[cols], tr[tgt])
        p.loc[te.index] = m.predict_proba(te[cols])[:, 1]
    return p

def main():
    df = fetch()
    for c in BASE: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df.open_time, utc=True); df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["blk"] = (df.open_time >= "2024-11-01").astype(int) + (df.open_time >= "2025-11-01").astype(int)
    d = df[~df.already_moved.astype(bool)].copy(); d["y_10_24"] = d.y_10_24.astype(int); d["y_40_72"] = d.y_40_72.astype(int)
    months = [str(p) for p in pd.period_range("2023-09", "2026-10", freq="M")]
    times = pd.DatetimeIndex(sorted(d.open_time.unique())); syms = sorted(d.symbol.unique())
    with ThreadPoolExecutor(8) as ex: res = [r for r in ex.map(kfeat, [(s, months, times) for s in syms]) if r is not None]
    d = d.merge(pd.concat(res), on=["symbol", "open_time"], how="inner"); d = d[d.qv24_raw >= 5e4].reset_index(drop=True)
    d["day"] = d.open_time.dt.normalize(); d["week"] = d.day.dt.strftime("%G-%V")
    print("rows", len(d), "days", d.day.nunique(), "coins", d.symbol.nunique(), flush=True)
    weeks = d.week.unique()

    # ---- 1. fingerprint flags vs CONTROL (everything else) ----
    flats = d.ret_24h.abs() < 0.03
    flags = {"A: حجم مرتفع (>=3x) بسعر ثابت خلال 24س": (d.n_burst_flat_24 >= 1) & flats, "B: ضغط شراء صافٍ (cvd72>0.05) وسعر ثابت 72س": (d.cvd_norm_72 > 0.05) & (d.ret_72h.abs() < 0.05),
             "C: سيولة صغيرة (<1M دولار/24س)": d.qv24_raw < 1e6, "A+B": None, "A+B+C (البصمة الكاملة)": None}
    flags["A+B"] = flags["A: حجم مرتفع (>=3x) بسعر ثابت خلال 24س"] & flags["B: ضغط شراء صافٍ (cvd72>0.05) وسعر ثابت 72س"]
    flags["A+B+C (البصمة الكاملة)"] = flags["A+B"] & flags["C: سيولة صغيرة (<1M دولار/24س)"]
    base10, base40 = d.y_10_24.mean(), d.y_40_72.mean(); print(f"\nالمعدل العام (بلا مجموعات): +10%/24س = {base10:.3f} | +40%/72س = {base40:.4f}")
    rows = []
    for name, m in flags.items():
        sub = d[m.fillna(False)]
        if len(sub) < 50: rows.append({"بصمة": name, "n": len(sub)}); continue
        def rate(x, y): return x[y].mean()
        bs10 = [rate(d[m.fillna(False) & d.week.isin(RNG.choice(weeks, len(weeks)))], "y_10_24") for _ in range(200)]
        rows.append({"بصمة": name, "n": len(sub), "% من الصفوف": round(100 * len(sub) / len(d), 1), "نسبة +10%/24س": round(sub.y_10_24.mean(), 3), "ضعف": round(sub.y_10_24.mean() / base10, 2),
                     "CI95 للنسبة": f"[{np.percentile(bs10, 2.5):.3f},{np.percentile(bs10, 97.5):.3f}]", "نسبة +40%/72س": round(sub.y_40_72.mean(), 4), "ضعف 40": round(sub.y_40_72.mean() / base40, 2)})
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n== البصمة مقابل المجموعة الضابطة (كل الصفوف الأخرى) =="); print(pd.DataFrame(rows).to_string(index=False))

    # ---- 2. do they add ranking power over BASE? ----
    for tgt in ("y_10_24", "y_40_72"):
        d["p0"] = loyo(d, BASE, tgt); d["p1"] = loyo(d, BASE + NEW, tgt)
        auc0, auc1 = roc_auc_score(d[tgt], d.p0), roc_auc_score(d[tgt], d.p1)
        per = {}
        for nm in ("p0", "p1"):
            rk = d.groupby("day")[nm].rank(ascending=False, method="first"); t = d[rk <= 5]; per[nm] = t.groupby("day")[tgt].agg(["sum", "count"])
        j = per["p0"].join(per["p1"], lsuffix="0", rsuffix="1").join(d.groupby("day").week.first()); wk = j.week.unique(); idx = {w: np.where(j.week.values == w)[0] for w in wk}
        def st(jj): return jj.sum0.sum() / jj.count0.sum(), jj.sum1.sum() / jj.count1.sum()
        pt = st(j); bs = np.array([ (lambda a: a[1] - a[0])(st(j.iloc[np.concatenate([idx[w] for w in RNG.choice(wk, len(wk))])])) for _ in range(500)])
        print(f"\n== {tgt}: AUC أساسي {auc0:.4f} -> مع البصمات {auc1:.4f} | دقة أفضل 5: {pt[0]:.3f} -> {pt[1]:.3f} (فرق {pt[1]-pt[0]:+.3f}, CI95 [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]) | معدل عام {d[tgt].mean():.3f}")
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)); m.fit(d[BASE + NEW], d.y_10_24)
    print("\nمعاملات البصمات الجديدة (قياسية، y_10_24):"); print(pd.Series(m[-1].coef_[0][len(BASE):], index=NEW).round(3).to_string())
main()
