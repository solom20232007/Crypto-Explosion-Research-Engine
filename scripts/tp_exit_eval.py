"""Take-profit / stop exit test (research only). Same scoring, costs and entry timing as realistic_pnl_eval.py, but exits at a target.
TP fills at the target price (optimistic); if stop and target are both touched in one hourly bar the STOP is assumed first (conservative).

Realistic tradeability test (research only): does the model's top-K produce NET profit, not just a high intraday peak?

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
DELAYS, HOLDS = (0, 8), (24, 48)
TPS = (0.03, 0.05, 0.08, 0.10, 0.15); STOPS = (None, -0.05, -0.08)

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
            df = pd.read_csv(io.StringIO(raw), header=None, usecols=[0, 2, 3, 4, 7], names=["t", "high", "low", "close", "qv"])
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
    c, hi, lo, qv = d.close, d.high, d.low, d.qv.fillna(0)
    out = pd.DataFrame(index=d.index); out["qv24"] = qv.rolling(24).sum()
    for k in DELAYS:
        entry = c.shift(-k); H = max(HOLDS)
        HI = np.column_stack([(hi.shift(-(k + j)) / entry - 1).values for j in range(1, H + 1)])   # j-th bar after entry
        LO = np.column_stack([(lo.shift(-(k + j)) / entry - 1).values for j in range(1, H + 1)])
        CL = np.column_stack([(c.shift(-(k + j)) / entry - 1).values for j in range(1, H + 1)])
        for h in HOLDS:
            hh, ll, cc = HI[:, :h], LO[:, :h], CL[:, :h]; valid = ~np.isnan(cc[:, -1])
            for tp in TPS:
                for st in STOPS:
                    th = hh >= tp; sh = (ll <= st) if st is not None else np.zeros_like(th)
                    ft = np.where(th.any(1), th.argmax(1), 10**6); fs = np.where(sh.any(1), sh.argmax(1), 10**6)
                    r = np.where(ft < fs, tp, np.where(fs <= ft, np.where(fs < 10**6, st if st is not None else 0, cc[:, -1]), cc[:, -1]))
                    r = np.where((ft < 10**6) & (ft < fs), tp, np.where(fs < 10**6, st if st is not None else 0, cc[:, -1]))
                    out[f"r_{k}_{h}_{int(tp*100)}_{'n' if st is None else int(-st*100)}"] = np.where(valid, r, np.nan)
            out[f"hit10_{k}_{h}"] = np.where(valid, (hh >= 0.10).any(1), np.nan)
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
    times = pd.DatetimeIndex(sorted(d.open_time.unique())); syms = sorted(d.symbol.unique()); print("symbols", len(syms), flush=True)
    with ThreadPoolExecutor(8) as ex: res = [r for r in ex.map(fwd, [(s, months, times) for s in syms]) if r is not None]
    d = d.merge(pd.concat(res), on=["symbol", "open_time"], how="inner"); d = d[d.qv24 >= 5e4].copy()
    d["cost"] = 2 * (FEE + slip(d.qv24.values)); d["day"] = d.open_time.dt.normalize(); d["rk"] = d.groupby("day").p.rank(ascending=False, method="first")
    d["wk"] = d.day.dt.strftime("%G-%V"); print("rows", len(d), "days", d.day.nunique(), flush=True)
    for k in DELAYS:
        for h in HOLDS:
            c = f"hit10_{k}_{h}"; x = d.dropna(subset=[c])
            print(f"نسبة الوصول فعليًا لـ+10% بعد الدخول (+{k}س، {h}س): أفضل 5 = {x[x.rk<=5][c].mean():.3f} | كل العملات = {x[c].mean():.3f}")
    rows = []
    for col in [c for c in d.columns if c.startswith("r_")]:
        x = d.dropna(subset=[col]).copy(); x["net"] = x[col] - x.cost
        top = x[x.rk <= 5].groupby("day").net.mean(); allm = x.groupby("day").net.mean(); j = pd.concat([top.rename("t"), allm.rename("a")], axis=1).dropna()
        j["wk"] = j.index.strftime("%G-%V"); weeks = j.wk.unique(); idx = {w: np.where(j.wk.values == w)[0] for w in weeks}
        bs = [(lambda jj: (jj.t - jj.a).mean())(j.iloc[np.concatenate([idx[w] for w in RNG.choice(weeks, len(weeks))])]) for _ in range(300)]
        lo, hi = np.percentile(bs, [2.5, 97.5]); _, k, h, tp, st = col.split("_")
        rows.append({"دخول+س": k, "إمساك": h, "هدف%": tp, "وقف%": st, "صافي أفضل5 %": round(j.t.mean() * 100, 2), "كل العملات %": round(j.a.mean() * 100, 2),
                     "الفرق %": round((j.t - j.a).mean() * 100, 2), "CI95": f"[{lo*100:.2f},{hi*100:.2f}]"})
    r = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print("\n== كل التركيبات (صافي بعد التكاليف) =="); print(r.to_string(index=False))
    print("\n== أفضل 8 حسب صافي أفضل5 =="); print(r.sort_values("صافي أفضل5 %", ascending=False).head(8).to_string(index=False))
main()
