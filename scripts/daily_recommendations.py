"""Daily research indicators (NOT buy/sell signals).

Trains logistic models on prebreakout_samples_v1 (all rows, base features), builds the same features for the
latest 00:00 UTC bar of every research symbol from the candles table, and upserts ranked results to
daily_recommendations_v1. Probabilities are population-calibrated (sample weights restore prevalence).
"""
import json, os, sys
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = os.environ["SUPABASE_URL"].rstrip("/"); KEY = os.environ["SUPABASE_SECRET_KEY"]
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
BASE = ["ret_1h","ret_3h","ret_6h","ret_12h","ret_24h","ret_48h","ret_72h","ret_168h","vol_ratio_24","vol_ratio_168",
        "range_24h","range_168h","rv24","rv168","vol_compression","close_vs_sma20","close_vs_sma50","dist_from_high_168",
        "dist_from_low_168","btc_ret_24h","btc_ret_72h","breadth_24h","pump_count_24h","hour_utc","dow"]
TARGETS = ["y_10_24", "y_25_48", "y_40_72"]
MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
NOT_CANDIDATES = MAJORS | {"EURIUSDT", "WBETHUSDT", "WBTCUSDT"}  # stable / wrapped majors: context only
TOP_N = int(os.environ.get("TOP_N", "30"))

def rest(table, params, rng=None):
    h = dict(H); 
    if rng: h["Range"] = rng
    r = requests.get(f"{URL}/rest/v1/{table}", headers=h, params=params, timeout=120); r.raise_for_status(); return r.json()

def fetch_all(table, select, order):
    rows, off = [], 0
    while True:
        b = rest(table, {"select": select, "order": order}, f"{off}-{off+999}"); rows += b
        if len(b) < 1000: return pd.DataFrame(rows)
        off += 1000

def _fit(d, t):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    m.fit(d[BASE], d[t].astype(int), logisticregression__sample_weight=d.sample_weight); return m

def train():
    df = fetch_all("prebreakout_samples_v1", "symbol,open_time,sample_weight," + ",".join(TARGETS + BASE), "symbol,open_time")
    for c in BASE + ["sample_weight"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df.open_time, utc=True)
    split = df.open_time.quantile(0.7)
    tr = df[df.open_time < split - pd.Timedelta(hours=72)]; ho = df[df.open_time >= split].sort_values("open_time")
    half = ho.open_time.quantile(0.5); ho_a, ho_b = ho[ho.open_time < half], ho[ho.open_time >= half]
    models, cals = {}, {}
    for t in TARGETS:
        early = _fit(tr, t)
        # isotonic map raw score -> observed (weighted) rate, fitted on the first half of the holdout
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(early.predict_proba(ho_a[BASE])[:, 1], ho_a[t].astype(int), sample_weight=ho_a.sample_weight)
        # honest check on the second half of the holdout
        p = early.predict_proba(ho_b[BASE])[:, 1]; q = iso.predict(p); w = ho_b.sample_weight.values; y = ho_b[t].astype(int).values
        bins = pd.qcut(pd.Series(p), 5, labels=False, duplicates="drop")
        rep = [(int(k), round(float(np.average(q[bins == k], weights=w[bins == k])), 3), round(float(np.average(y[bins == k], weights=w[bins == k])), 3)) for k in sorted(bins.unique())]
        print(t, "calibration check (quintile, predicted, observed):", rep)
        cals[t] = iso; models[t] = _fit(df, t)
    return models, cals, sorted(df.symbol.unique()), str(df.open_time.max())

def candles(symbol, n=260):
    b = rest("candles", {"select": "open_time,high,low,close,quote_volume", "symbol": f"eq.{symbol}", "timeframe": "eq.1h",
                         "order": "open_time.desc", "limit": str(n)})
    d = pd.DataFrame(b)
    for c in ["high", "low", "close", "quote_volume"]: d[c] = pd.to_numeric(d[c])
    d["open_time"] = pd.to_datetime(d.open_time, utc=True)
    return d.sort_values("open_time").reset_index(drop=True)

def features(d):
    c = d.close; r1 = c / c.shift(1) - 1; f = pd.DataFrame({"open_time": d.open_time})
    for k in (1, 3, 6, 12, 24, 48, 72, 168): f[f"ret_{k}h"] = c / c.shift(k) - 1
    f["vol_ratio_24"] = d.quote_volume / d.quote_volume.shift(1).rolling(24).mean()
    f["vol_ratio_168"] = d.quote_volume / d.quote_volume.shift(1).rolling(168).mean()
    f["range_24h"] = (d.high.rolling(24).max() - d.low.rolling(24).min()) / c
    f["range_168h"] = (d.high.rolling(168).max() - d.low.rolling(168).min()) / c
    f["rv24"] = r1.rolling(24).std(); f["rv168"] = r1.rolling(168).std()
    f["vol_compression"] = f.rv24 / f.rv168
    f["close_vs_sma20"] = c / c.rolling(20).mean() - 1; f["close_vs_sma50"] = c / c.rolling(50).mean() - 1
    f["dist_from_high_168"] = c / d.high.rolling(168).max() - 1; f["dist_from_low_168"] = c / d.low.rolling(168).min() - 1
    f["hour_utc"] = d.open_time.dt.hour; f["dow"] = d.open_time.dt.isocalendar().day.astype(int)
    return f

def main():
    models, cals, symbols, last_train = train()
    symbols = [s for s in symbols if s not in NOT_CANDIDATES]
    print("trained; universe", len(symbols), "samples through", last_train)
    now = datetime.now(timezone.utc); bar = pd.Timestamp(now.date(), tz="UTC")  # today's 00:00 bar (closed at 01:00)
    if now.hour < 1: bar -= pd.Timedelta(days=1)
    frames = {s: features(candles(s)) for s in symbols}
    btc = candles("BTCUSDT"); btc["btc_ret_24h"] = btc.close / btc.close.shift(24) - 1; btc["btc_ret_72h"] = btc.close / btc.close.shift(72) - 1
    bt = btc[btc.open_time == bar]
    if bt.empty: sys.exit(f"no BTC bar for {bar}")
    btc24, btc72 = float(bt.btc_ret_24h.iloc[0]), float(bt.btc_ret_72h.iloc[0])
    rows = []
    for s, f in frames.items():
        x = f[f.open_time == bar]
        if x.empty or x.drop(columns="open_time").isna().any(axis=1).iloc[0]: continue
        rows.append({"symbol": s, **x.iloc[0].to_dict()})
    d = pd.DataFrame(rows)
    if len(d) < 60: sys.exit(f"only {len(d)} symbols have a complete bar for {bar}; refusing to publish")
    d["btc_ret_24h"], d["btc_ret_72h"] = btc24, btc72
    d["breadth_24h"] = float((d.ret_24h > 0).mean()); d["pump_count_24h"] = int((d.ret_24h >= 0.10).sum())
    market_wide = bool(btc24 >= 0.03 or d.pump_count_24h.iloc[0] >= 12)
    for t, m in models.items():
        raw = m.predict_proba(d[BASE])[:, 1]; d["score_" + t[2:]] = cals[t].predict(raw)
        if t == "y_10_24": d["raw_10_24"] = raw
    d["already_moved"] = (d.ret_24h >= 0.10) | (d.ret_48h >= 0.25) | (d.ret_72h >= 0.40)
    d = d.sort_values("raw_10_24", ascending=False).reset_index(drop=True); d["rank"] = d.index + 1
    top = d.head(TOP_N)
    cols = ["symbol","rank","raw_10_24","score_10_24","score_25_48","score_40_72","already_moved","btc_ret_24h","breadth_24h","pump_count_24h",
            "ret_24h","ret_72h","rv24","range_24h","vol_ratio_24","vol_compression","close_vs_sma20","dist_from_high_168","dist_from_low_168"]
    out = top[cols].copy(); out["bar_time"] = bar.isoformat(); out["market_wide"] = market_wide
    out["model"] = f"logistic_base_v2_isotonic; samples_through={last_train[:10]}"
    payload = json.loads(out.to_json(orient="records"))
    r = requests.post(f"{URL}/rest/v1/daily_recommendations_v1?on_conflict=bar_time,symbol",
                      headers={**H, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    print(f"bar={bar} scored={len(d)} market_wide={market_wide} published={len(payload)}")
    print(out[["rank","symbol","score_10_24","score_25_48","score_40_72","already_moved","ret_24h","rv24","vol_ratio_24"]].head(15).to_string(index=False))
main()
