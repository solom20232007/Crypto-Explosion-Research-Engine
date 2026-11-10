"""Volume-surge study (research only): when an hour's quote volume is >= K x its trailing-24h mean, what happens AFTER that hour closes?

Detection is concurrent (the surge hour itself already moved), so this measures whether detection leaves anything to capture, not prediction.
"""
import io, os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]; H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
SKIP = {"BTCUSDT","ETHUSDT","SOLUSDT","EURIUSDT","WBETHUSDT","WBTCUSDT"}
ARCH = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{m}.zip"

def universe():
    r = requests.get(f"{URL}/rest/v1/candles", headers=H, params={"select": "symbol", "timeframe": "eq.1h", "open_time": "eq.2026-10-15T00:00:00+00:00", "limit": "1000"}, timeout=60); r.raise_for_status()
    return sorted({x["symbol"] for x in r.json()} - SKIP)

def mf(a):
    s, m = a
    try:
        r = requests.get(ARCH.format(s=s, m=m), timeout=60)
        if r.status_code == 404: return None
        z = zipfile.ZipFile(io.BytesIO(r.content)); df = pd.read_csv(io.StringIO(z.read(z.namelist()[0]).decode()), header=None, usecols=[0, 1, 2, 3, 4, 7], names=["t", "open", "high", "low", "close", "qv"])
        df = df[pd.to_numeric(df.t, errors="coerce").notna()].astype({"t": "int64"}); df["t"] = np.where(df.t > 10**14, df.t // 1000, df.t); return df
    except Exception: return None

def one(s):
    months = [str(p) for p in pd.period_range("2023-10", "2026-10", freq="M")]
    parts = [p for p in (mf((s, m)) for m in months) if p is not None]
    if not parts: return None
    d = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    d["open_time"] = pd.to_datetime(d.t, unit="ms", utc=True)
    c, hi, lo = d.close, d.high, d.low
    d["vr"] = d.qv / d.qv.shift(1).rolling(24, min_periods=24).mean()
    d["hr_ret"] = c / d.open - 1
    d["quiet"] = (c.shift(1) / c.shift(25) - 1).abs() < 0.03   # previous 24h within +-3%
    for k in (1, 6, 24): d[f"f{k}"] = c.shift(-k) / c - 1
    d["fmax24"] = hi[::-1].shift(1).rolling(24, min_periods=24).max()[::-1] / c - 1
    d["fmin24"] = lo[::-1].shift(1).rolling(24, min_periods=24).min()[::-1] / c - 1
    d["symbol"] = s; d["qv24"] = d.qv.shift(1).rolling(24).sum()
    return d[(d.vr >= 5)].dropna(subset=["f24", "fmax24"]) , d.shape[0], int(d.fmax24.notna().sum())

def main():
    syms = universe(); print("symbols", len(syms), flush=True)
    with ThreadPoolExecutor(8) as ex: res = [r for r in ex.map(one, syms) if r is not None]
    ev = pd.concat([r[0] for r in res]); total_h = sum(r[1] for r in res)
    ev = ev[(ev.qv24 > 20000)]  # ignore dead coins (<20k USDT traded in prior 24h)
    print("hours total", total_h, "surge events (vr>=5, liquid)", len(ev), "years", ev.open_time.min(), ev.open_time.max(), flush=True)
    def summ(g, name):
        return {"group": name, "n": len(g), "med_f1": g.f1.median(), "med_f6": g.f6.median(), "med_f24": g.f24.median(), "mean_f24": g.f24.mean(),
                "p_more10_24": (g.fmax24 >= 0.10).mean(), "p_drop10_24": (g.fmin24 <= -0.10).mean(), "med_maxfwd": g.fmax24.median(), "med_minfwd": g.fmin24.median()}
    rows = []
    for k in (5, 10, 20, 50):
        g = ev[ev.vr >= k]
        rows.append(summ(g, f"surge>={k}x (all)"))
        rows.append(summ(g[g.hr_ret >= 0.10], f"surge>={k}x, hour ret>=+10%"))
        rows.append(summ(g[(g.hr_ret >= 0.03) & (g.hr_ret < 0.10)], f"surge>={k}x, hour ret +3..10%"))
        rows.append(summ(g[(g.hr_ret.abs() < 0.03)], f"surge>={k}x, hour ret within +-3%"))
        rows.append(summ(g[g.hr_ret <= -0.03], f"surge>={k}x, hour ret<=-3%"))
        rows.append(summ(g[g.quiet & (g.hr_ret >= 0.10)], f"surge>={k}x, was quiet 24h then +10% hour"))
    allh = pd.concat([r[0] for r in res])
    print("\nبعد إغلاق ساعة القفزة (دخول عند إغلاقها)؛ النسب كسور:"); print(pd.DataFrame(rows).round(3).to_string(index=False))
    # base: random liquid hours
    print("\nالمرجع: أي ساعة سيولة (احتمال ارتفاع +10% خلال 24س لاحقة / هبوط -10%) ≈ حسب الصفوف أعلاه مقارنة بـ ~", "10% / ~10%")
    ev["yr"] = ev.open_time.dt.year
    g = ev[(ev.vr >= 10) & (ev.hr_ret >= 0.10)]
    print("\nثبات عبر السنوات (قفزة>=10x وساعة >=+10%):"); print(g.groupby("yr").apply(lambda x: pd.Series({"n": len(x), "med_f24": x.f24.median(), "p_more10": (x.fmax24 >= 0.10).mean(), "p_drop10": (x.fmin24 <= -0.10).mean()})).round(3).to_string())
main()
