"""Honest uncertainty for the headline numbers: day-clustered bootstrap CIs + event-level (episode) catch rates. Research only.

Rows are not independent (a market-wide day pumps many coins at once; a coin can stay 'positive' for consecutive days), so:
  * row-level top-5 precision / lift get 95% CIs from resampling WEEKS (blocks) with replacement;
  * an EVENT = a maximal run of consecutive days with y_10_24 = 1 for one coin, counted once; 'caught' = coin ranked in the day's top-K
    on the first day of the run (strict) or any day of the run (lenient), compared with picking K coins at random.
"""
import os
import numpy as np, pandas as pd, requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]; H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
BASE = ["ret_1h","ret_3h","ret_6h","ret_12h","ret_24h","ret_48h","ret_72h","ret_168h","vol_ratio_24","vol_ratio_168","range_24h","range_168h","rv24","rv168",
        "vol_compression","close_vs_sma20","close_vs_sma50","dist_from_high_168","dist_from_low_168","btc_ret_24h","btc_ret_72h","breadth_24h","pump_count_24h","hour_utc","dow"]
RNG = np.random.default_rng(7)

def fetch():
    rows, off = [], 0
    sel = "symbol,open_time,already_moved,market_wide,y_10_24," + ",".join(BASE)
    while True:
        r = requests.get(f"{URL}/rest/v1/history_daily_v1", headers={**H, "Range": f"{off}-{off+999}"}, params={"select": sel, "order": "symbol,open_time"}, timeout=120); r.raise_for_status()
        b = r.json(); rows += b
        if len(b) < 1000: return pd.DataFrame(rows)
        off += 1000

def ci(x): return np.percentile(x, [2.5, 97.5])

def main():
    df = fetch()
    for c in BASE: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df.open_time, utc=True); df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["y"] = df.y_10_24.astype(int); df["blk"] = (df.open_time >= "2024-11-01").astype(int) + (df.open_time >= "2025-11-01").astype(int)
    # leave-one-year-out scores on not-already-moved rows (training excludes already-moved rows, as in production)
    ok = ~df.already_moved.astype(bool); df["p"] = np.nan
    for b in (0, 1, 2):
        te = df[(df.blk == b) & ok]; lo, hi = te.open_time.min(), te.open_time.max(); emb = pd.Timedelta(days=3)
        tr = df[ok & (df.blk != b) & ~((df.open_time > lo - emb) & (df.open_time < hi + emb))]
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)); m.fit(tr[BASE], tr.y)
        df.loc[te.index, "p"] = m.predict_proba(te[BASE])[:, 1]
    d = df[ok].copy(); d["day"] = d.open_time.dt.normalize(); d["week"] = d.open_time.dt.strftime("%G-%V")
    d["rk"] = d.groupby("day").p.rank(ascending=False, method="first")
    print("rows", len(d), "days", d.day.nunique(), "coin-days positives", int(d.y.sum()), "base", round(d.y.mean(), 4))

    # ---- row-level with week-block bootstrap ----
    per = d.groupby("day").apply(lambda g: pd.Series({"week": g.week.iloc[0], "n": len(g), "h": g.y.sum(), "n5": (g.rk <= 5).sum(), "h5": g.y[g.rk <= 5].sum(),
                                                     "n30": (g.rk <= 30).sum(), "h30": g.y[g.rk <= 30].sum(), "mw": g.market_wide.astype(bool).iloc[0]}))
    weeks = per.week.unique(); idx = {w: per.index[per.week == w] for w in weeks}
    def stats(p):
        base = p.h.sum() / p.n.sum(); a5 = p.h5.sum() / p.n5.sum(); a30 = p.h30.sum() / p.n30.sum(); return base, a5, a5 / base, a30, a30 / base
    pt = stats(per); bs = np.array([stats(per.loc[np.concatenate([idx[w] for w in RNG.choice(weeks, len(weeks))])]) for _ in range(1000)])
    print("\n== مستوى الصفوف (CI 95% بإعادة عينات الأسابيع) ==")
    for i, name in enumerate(["المعدل العام", "دقة أفضل 5", "ضعف أفضل 5", "دقة أفضل 30", "ضعف أفضل 30"]):
        lo, hi = ci(bs[:, i]); print(f"{name:14s} {pt[i]:.3f}  [{lo:.3f} – {hi:.3f}]")
    print("أسابيع:", len(weeks), "| أيام السوق الصاعد عامًا:", int(per.mw.sum()))
    print("ضعف أفضل 5 بدون أيام market_wide:", round(stats(per[~per.mw])[2], 2))

    # ---- day-level: at least one hit in top 5 ----
    from math import comb
    per["any5"] = 0
    anyhit = d[d.rk <= 5].groupby("day").y.max(); per["any5"] = anyhit.reindex(per.index).fillna(0)
    per["rand_any5"] = [1 - comb(int(n - h), 5) / comb(int(n), 5) if n >= 5 and n - h >= 5 else 1.0 for n, h in zip(per.n, per.h)]
    print("\n== مستوى اليوم: هل أصابت واحدة على الأقل من أفضل 5؟ ==")
    print(f"فعليًا {per.any5.mean():.3f} مقابل اختيار عشوائي {per.rand_any5.mean():.3f}")

    # ---- event-level ----
    ev = []
    for s, g in d.groupby("symbol"):
        g = g.sort_values("day"); run = None
        for _, r in g.iterrows():
            if r.y == 1:
                if run is None or (r.day - run["last"]).days != 1: 
                    if run: ev.append(run)
                    run = {"symbol": s, "first": r.day, "last": r.day, "rk_first": r.rk, "rk_min": r.rk, "week": r.week, "n_day": per.n.get(r.day, np.nan)}
                else: run["last"] = r.day; run["rk_min"] = min(run["rk_min"], r.rk)
            else:
                if run: ev.append(run); run = None
        if run: ev.append(run)
    E = pd.DataFrame(ev); print("\n== مستوى الحدث ==\nأحداث (تتابعات أيام إيجابية متصلة لعملة):", len(E), "مقابل", int(d.y.sum()), "صف إيجابي (", round(d.y.sum() / len(E), 2), "صف لكل حدث )")
    for K in (5, 30):
        for kind, col in (("صارم: اليوم الأول للحدث", "rk_first"), ("متساهل: أي يوم في الحدث", "rk_min")):
            hit = (E[col] <= K).astype(float); rnd = (K / E.n_day).clip(upper=1)
            wk = E.week.unique(); ix = {w: E.index[E.week == w] for w in wk}
            bsl = [ (hit.loc[i].mean() / rnd.loc[i].mean()) for i in (np.concatenate([ix[w] for w in RNG.choice(wk, len(wk))]) for _ in range(1000)) ]
            lo, hi = ci(bsl); print(f"أفضل {K:2d} | {kind:26s} نسبة الالتقاط {hit.mean():.3f}  عشوائيًا {rnd.mean():.3f}  ضعف {hit.mean()/rnd.mean():.2f}  [{lo:.2f} – {hi:.2f}]")
    print("\nحسب السنة (ضعف التقاط الحدث الصارم، أفضل 5):")
    E["blk"] = (E["first"] >= "2024-11-01").astype(int) + (E["first"] >= "2025-11-01").astype(int)
    for b, g in E.groupby("blk"): print(f"  سنة {b+1}: أحداث={len(g)}  التقاط={ (g.rk_first<=5).mean():.3f}  عشوائيًا={ (5/g.n_day).clip(upper=1).mean():.3f}")
main()
