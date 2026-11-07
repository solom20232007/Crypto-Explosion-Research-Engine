"""3-year historical study from the public Binance archive (data.binance.vision). Research only.

Builds the same 00:00-UTC daily feature rows + labels as the production model (hourly klines -> features), then tests whether the
signal holds across different market years (train on earlier year(s), test on a later one). No raw candles are stored.
"""
import io, os, sys, zipfile, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
START = os.environ.get("START_MONTH", "2023-10"); END = os.environ.get("END_MONTH", "2026-10")
BASE = ["ret_1h","ret_3h","ret_6h","ret_12h","ret_24h","ret_48h","ret_72h","ret_168h","vol_ratio_24","vol_ratio_168",
        "range_24h","range_168h","rv24","rv168","vol_compression","close_vs_sma20","close_vs_sma50","dist_from_high_168",
        "dist_from_low_168","btc_ret_24h","btc_ret_72h","breadth_24h","pump_count_24h","hour_utc","dow"]
SKIP = {"BTCUSDT","ETHUSDT","SOLUSDT","EURIUSDT","WBETHUSDT","WBTCUSDT"}
ARCH = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{m}.zip"

def universe():
    r = requests.get(f"{URL}/rest/v1/candles", headers=H, params={"select": "symbol", "timeframe": "eq.1h", "open_time": "eq.2026-10-15T00:00:00+00:00", "limit": "1000"}, timeout=60)
    r.raise_for_status(); return sorted({x["symbol"] for x in r.json()} | {"BTCUSDT"})

def month_file(a):
    s, m = a
    for _ in range(3):
        try:
            r = requests.get(ARCH.format(s=s, m=m), timeout=60)
            if r.status_code == 404: return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content)); raw = z.read(z.namelist()[0]).decode()
            df = pd.read_csv(io.StringIO(raw), header=None, usecols=[0, 2, 3, 4, 7], names=["t", "high", "low", "close", "qv"])
            df = df[pd.to_numeric(df.t, errors="coerce").notna()].astype({"t": "int64"})
            df["t"] = np.where(df.t > 10**14, df.t // 1000, df.t); return df
        except Exception as e:
            err = e
    print("fail", s, m, err, file=sys.stderr); return None

def load(sym, months):
    parts = [p for p in (month_file((sym, m)) for m in months) if p is not None]
    if not parts: return None
    d = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    d["open_time"] = pd.to_datetime(d.t, unit="ms", utc=True)
    return d.drop(columns="t")

def feats(d):
    c = d.close; r1 = c / c.shift(1) - 1; f = pd.DataFrame({"open_time": d.open_time})
    for k in (1, 3, 6, 12, 24, 48, 72, 168): f[f"ret_{k}h"] = c / c.shift(k) - 1
    f["vol_ratio_24"] = d.qv / d.qv.shift(1).rolling(24).mean(); f["vol_ratio_168"] = d.qv / d.qv.shift(1).rolling(168).mean()
    f["range_24h"] = (d.high.rolling(24).max() - d.low.rolling(24).min()) / c; f["range_168h"] = (d.high.rolling(168).max() - d.low.rolling(168).min()) / c
    f["rv24"] = r1.rolling(24).std(); f["rv168"] = r1.rolling(168).std(); f["vol_compression"] = f.rv24 / f.rv168
    f["close_vs_sma20"] = c / c.rolling(20).mean() - 1; f["close_vs_sma50"] = c / c.rolling(50).mean() - 1
    f["dist_from_high_168"] = c / d.high.rolling(168).max() - 1; f["dist_from_low_168"] = c / d.low.rolling(168).min() - 1
    f["hour_utc"] = d.open_time.dt.hour; f["dow"] = d.open_time.dt.isocalendar().day.astype(int)
    hi = d.high[::-1]
    for n, name in ((24, "y_10_24"), (48, "y_25_48"), (72, "y_40_72")):
        fm = hi.shift(1).rolling(n, min_periods=n).max()[::-1]; f["fwd_" + name] = fm / c - 1  # NaN when fewer than n future bars
    f["qv24"] = d.qv.rolling(24).sum()
    return f

def one(a):
    s, months = a; d = load(s, months)
    if d is None or len(d) < 400: return s, None
    f = feats(d); f = f[f.open_time.dt.hour == 0].copy(); f["symbol"] = s; return s, f

def main():
    months = [str(p) for p in pd.period_range(START, END, freq="M")]
    syms = universe(); print("symbols", len(syms), "months", len(months), flush=True)
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, [(s, months) for s in syms]))
    allf = pd.concat([f for s, f in res if f is not None]); btc = allf[allf.symbol == "BTCUSDT"].set_index("open_time")
    allf = allf[~allf.symbol.isin(SKIP)].copy()
    allf["btc_ret_24h"] = allf.open_time.map(btc.ret_24h); allf["btc_ret_72h"] = allf.open_time.map(btc.ret_72h)
    g = allf.groupby("open_time")
    allf["breadth_24h"] = g.ret_24h.transform(lambda x: (x > 0).mean()); allf["pump_count_24h"] = g.ret_24h.transform(lambda x: (x >= 0.10).sum())
    allf["y_10_24"] = allf.fwd_y_10_24 >= 0.10; allf["y_25_48"] = allf.fwd_y_25_48 >= 0.25; allf["y_40_72"] = allf.fwd_y_40_72 >= 0.40
    allf["already_moved"] = (allf.ret_24h >= 0.10) | (allf.ret_48h >= 0.25) | (allf.ret_72h >= 0.40)
    allf["market_wide"] = (allf.btc_ret_24h >= 0.03) | (allf.pump_count_24h >= 12)
    d = allf.dropna(subset=BASE + ["fwd_y_40_72"]).copy()
    d = d[d.qv24 > 0]; d["ym"] = d.open_time.dt.to_period("M").astype(str)
    d.to_csv("history_daily.csv.gz", index=False)
    print("rows", len(d), "symbols", d.symbol.nunique(), d.open_time.min(), d.open_time.max())
    d = d[~d.already_moved]
    print("\nBase rates by year-block (not already moved):")
    d["blk"] = ((d.open_time >= "2024-11-01").astype(int) + (d.open_time >= "2025-11-01").astype(int)).map({0: "Y1 2023-11..2024-10", 1: "Y2 2024-11..2025-10", 2: "Y3 2025-11..2026-10"})
    print(d.groupby("blk").agg(n=("symbol", "size"), b10=("y_10_24", "mean"), b25=("y_25_48", "mean"), b40=("y_40_72", "mean"), mkt_wide=("market_wide", "mean")).round(4).to_string())
    def fit(t): m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)); m.fit(t[BASE], t.tgt.astype(int)); return m
    def ev(te, p, k=5):
        te = te.assign(p=p); out = {"n": len(te), "base": te.tgt.mean()}
        try: out["auc"] = roc_auc_score(te.tgt, p)
        except Exception: out["auc"] = np.nan
        top = te.sort_values("p", ascending=False).groupby("open_time").head(k); out["top5_prec"] = top.tgt.mean(); out["lift5"] = out["top5_prec"] / out["base"] if out["base"] else np.nan
        nm = te[~te.market_wide]; 
        if len(nm) > 50:
            t2 = nm.sort_values("p", ascending=False).groupby("open_time").head(k); out["top5_prec_nonmkt"] = t2.tgt.mean(); out["base_nonmkt"] = nm.tgt.mean()
        return out
    blks = ["Y1 2023-11..2024-10", "Y2 2024-11..2025-10", "Y3 2025-11..2026-10"]
    rows = []
    for tg in ("y_10_24", "y_25_48", "y_40_72"):
        d["tgt"] = d[tg]
        for tr_b, te_b in ((blks[:1], blks[1]), (blks[:1], blks[2]), (blks[:2], blks[2]), (blks[1:2], blks[2]), (blks[1:2], blks[0]), (blks[2:], blks[0]), (blks[2:], blks[1])):
            tr = d[d.blk.isin(tr_b)]; te = d[d.blk == te_b]
            if tg != "y_10_24": pass
            gap = pd.Timedelta(hours=72)
            if tr.open_time.max() + gap > te.open_time.min() and tr.open_time.min() < te.open_time.min(): tr = tr[tr.open_time < te.open_time.min() - gap]
            m = fit(tr); r = ev(te, m.predict_proba(te[BASE])[:, 1]); r.update(target=tg, train="+".join(x[:2] for x in tr_b), test=te_b[:2]); rows.append(r)
    res = pd.DataFrame(rows)[["target", "train", "test", "n", "base", "auc", "top5_prec", "lift5", "base_nonmkt", "top5_prec_nonmkt"]].round(4)
    print("\nCross-year generalization (train -> test; top5 = best 5 coins per day):"); print(res.to_string(index=False))
    # which features carry over years: standardized logistic coefs per block, target y_10_24
    d["tgt"] = d.y_10_24; co = {}
    for b in blks:
        m = fit(d[d.blk == b]); co[b[:2]] = pd.Series(m[-1].coef_[0], index=BASE)
    print("\nStandardized logistic coefficients by year (sign/size stability):"); print(pd.DataFrame(co).round(3).assign(abs_mean=lambda x: x.abs().mean(axis=1)).sort_values("abs_mean", ascending=False).head(12).to_string())
    json.dump({"rows": len(d)}, open("history_summary.json", "w"))
main()
