"""Realistic tradeability test (research only): does the model's top-K produce NET profit, not just a high intraday peak?

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
    sel = "symbol,open_time,already_moved,market_wide,y_10_24," + ",".join(BASE)
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
            df = pd.read_csv(io.StringIO(raw), header=None, usecols=[0, 3, 4, 7], names=["t", "low", "close", "qv"])
            df = df[pd.to_numeric(df.t, errors="coerce").notna()].astype({"t": "int64"})
            df["t"] = np.where(df.t > 10**14, df.t // 1000, df.t); return df
        except Exception as e:
            err = e
    print("fail", s, m, err, file=sys.stderr); return None

def fwd(a):
    s, months, times = a
    parts = [p for p in (month_file((s, m)) for m in months) if p is not None]
    if not parts: return None
    d = pd.concat(parts).drop_duplicates("t").sort_values("t")
    d.index = pd.to_datetime(d.t, unit="ms", utc=True); d = d.drop(columns="t")
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h", tz="UTC"))
    c, lo, qv = d.close, d.low, d.qv.fillna(0)
    out = pd.DataFrame(index=d.index); out["qv24"] = qv.rolling(24).sum()
    for k in DELAYS:
        entry = c.shift(-k)
        for h in HOLDS:
            out[f"g_{k}_{h}"] = c.shift(-(k + h)) / entry - 1
            fm = lo[::-1].rolling(h, min_periods=h).min()[::-1].shift(-(k + 1)) / entry - 1
            out[f"s_{k}_{h}"] = np.where(fm <= STOP, STOP, out[f"g_{k}_{h}"]); out.loc[out[f"g_{k}_{h}"].isna(), f"s_{k}_{h}"] = np.nan
    out = out.loc[out.index.isin(times)].copy(); out["symbol"] = s; return out.reset_index(names="open_time")

def slip(q): return np.where(q < 2e5, 0.005, np.where(q < 1e6, 0.003, 0.0015))

def main():
    df = fetch()
    for c in BASE: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df.open_time, utc=True); df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["y"] = df.y_10_24.astype(int); df["blk"] = (df.open_time >= "2024-11-01").astype(int) + (df.open_time >= "2025-11-01").astype(int)
    ok = ~df.already_moved.astype(bool); df["p"] = np.nan
    for b in (0, 1, 2):
        te = df[(df.blk == b) & ok]; lo, hi = te.open_time.min(), te.open_time.max(); emb = pd.Timedelta(days=3)
        tr = df[ok & (df.blk != b) & ~((df.open_time > lo - emb) & (df.open_time < hi + emb))]
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)); m.fit(tr[BASE], tr.y)
        df.loc[te.index, "p"] = m.predict_proba(te[BASE])[:, 1]
    d = df[ok].copy()
    months = [str(p) for p in pd.period_range("2023-09", "2026-10", freq="M")]
    times = pd.DatetimeIndex(sorted(d.open_time.unique()))
    syms = sorted(d.symbol.unique()); print("symbols", len(syms), "rows", len(d), flush=True)
    with ThreadPoolExecutor(8) as ex: res = [r for r in ex.map(fwd, [(s, months, times) for s in syms]) if r is not None]
    f = pd.concat(res); d = d.merge(f, on=["symbol", "open_time"], how="inner")
    d = d[d.qv24 >= 5e4].copy()  # at least 50k USDT traded in the prior 24h
    cost = 2 * (FEE + slip(d.qv24.values)); d["cost"] = cost
    d["day"] = d.open_time.dt.normalize(); d["week"] = d.open_time.dt.strftime("%G-%V"); d["year"] = d.blk
    d["rk"] = d.groupby("day").p.rank(ascending=False, method="first")
    print("tradable rows", len(d), "days", d.day.nunique(), "coins", d.symbol.nunique(), flush=True)

    def block(col, K, sub):
        x = sub.dropna(subset=[col]).copy(); x["net"] = x[col] - x.cost
        g = x.groupby("day"); allm = g.net.mean()
        top = x[x.rk <= K].groupby("day").net.mean(); j = pd.concat([top.rename("top"), allm.rename("all")], axis=1).dropna()
        j["wk"] = j.index.strftime("%G-%V"); return j, x
    def summarize(col, K, sub, label):
        j, x = block(col, K, sub)
        if len(j) < 30: return None
        weeks = j.wk.unique(); idx = {w: np.where(j.wk.values == w)[0] for w in weeks}
        def st(jj): return jj.top.mean(), jj["all"].mean(), (jj.top - jj["all"]).mean()
        pt = st(j); bs = np.array([st(j.iloc[np.concatenate([idx[w] for w in RNG.choice(weeks, len(weeks))])]) for _ in range(600)])
        lo, hi = np.percentile(bs[:, 2], [2.5, 97.5]); tr = x[x.rk <= K].net
        return {"حالة": label, "أيام": len(j), "صافي أفضل K %/يوم": round(pt[0] * 100, 2), "كل العملات %/يوم": round(pt[1] * 100, 2),
                "الفرق %": round(pt[2] * 100, 2), "CI95 للفرق": f"[{lo*100:.2f}, {hi*100:.2f}]", "وسيط الصفقة %": round(tr.median() * 100, 2), "نسبة الرابحة": round((tr > 0).mean(), 3)}
    rows = []
    for k in DELAYS:
        for h in HOLDS:
            for kind, nm in (("g", "بلا وقف"), ("s", "وقف -8%")):
                for K in (5,):
                    r = summarize(f"{kind}_{k}_{h}", K, d, f"دخول+{k}س، إمساك {h}س، {nm}، أفضل {K}")
                    if r: rows.append(r)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print("\n== النتيجة الرئيسية (صافي بعد الرسوم والانزلاق) =="); print(pd.DataFrame(rows).to_string(index=False))
    print("\n== حساسية: الحالة الواقعية (دخول+8س، إمساك 24س، بلا وقف) =="); rows = []
    for K in (1, 3, 5, 10, 30):
        r = summarize("g_8_24", K, d, f"أفضل {K}"); rows.append(r) if r else None
    for y, nm in ((0, "سنة1"), (1, "سنة2"), (2, "سنة3")):
        r = summarize("g_8_24", 5, d[d.year == y], f"أفضل 5 - {nm}"); rows.append(r) if r else None
    r = summarize("g_8_24", 5, d[~d.market_wide.astype(bool)], "أفضل 5 - بلا أيام market_wide"); rows.append(r) if r else None
    r = summarize("g_8_24", 5, d[d.qv24 >= 1e6], "أفضل 5 - سيولة >=1M فقط"); rows.append(r) if r else None
    print(pd.DataFrame(rows).to_string(index=False))
    x = d[d.rk <= 5].dropna(subset=["g_8_24"]); print("\nإجمالي قبل التكاليف (أفضل 5، +8س/24س): متوسط %.2f%% | بعد التكاليف: %.2f%% | متوسط التكلفة %.2f%%" % (x.g_8_24.mean() * 100, (x.g_8_24 - x.cost).mean() * 100, x.cost.mean() * 100))
    print("أسوأ 1%% من الصفقات (أفضل 5، صافي): %.1f%% | أفضل 1%%: %.1f%%" % (np.percentile(x.g_8_24 - x.cost, 1) * 100, np.percentile(x.g_8_24 - x.cost, 99) * 100))
main()
