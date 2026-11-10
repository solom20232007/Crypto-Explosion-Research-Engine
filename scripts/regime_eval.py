"""Regime tests on history_daily_v1: (1) does the signal work in BTC up vs down regimes? (2) do longer BTC trend features help? Research only."""
import io, os, zipfile
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
NEW = ["btc_ret_30d", "btc_vs_sma200"]

def fetch():
    rows, off = [], 0
    sel = "symbol,open_time,already_moved,market_wide,y_10_24,y_25_48,y_40_72," + ",".join(BASE)
    while True:
        r = requests.get(f"{URL}/rest/v1/history_daily_v1", headers={**H, "Range": f"{off}-{off+999}"}, params={"select": sel, "order": "symbol,open_time"}, timeout=120); r.raise_for_status()
        b = r.json(); rows += b
        if len(b) < 1000: return pd.DataFrame(rows)
        off += 1000

def btc_daily():
    def mf(m):
        try:
            r = requests.get(f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{m}.zip", timeout=60)
            if r.status_code == 404: return None
            z = zipfile.ZipFile(io.BytesIO(r.content)); df = pd.read_csv(io.StringIO(z.read(z.namelist()[0]).decode()), header=None, usecols=[0, 4], names=["t", "close"])
            df = df[pd.to_numeric(df.t, errors="coerce").notna()].astype({"t": "int64"}); df["t"] = np.where(df.t > 10**14, df.t // 1000, df.t); return df
        except Exception as e: print("fail", m, e); return None
    months = [str(p) for p in pd.period_range("2022-12", "2026-10", freq="M")]
    with ThreadPoolExecutor(8) as ex: parts = [p for p in ex.map(mf, months) if p is not None]
    d = pd.concat(parts).drop_duplicates("t").sort_values("t"); d["open_time"] = pd.to_datetime(d.t, unit="ms", utc=True)
    d = d[d.open_time.dt.hour == 0].set_index("open_time").close
    return pd.DataFrame({"btc_ret_30d": d / d.shift(30) - 1, "btc_vs_sma200": d / d.rolling(200).mean() - 1})

def fit(tr, feats, t):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)); m.fit(tr[feats], tr[t].astype(int)); return m

def metrics(te, p, t, k=5):
    te = te.assign(p=p, y=te[t].astype(int)); base = te.y.mean()
    top = te.sort_values("p", ascending=False).groupby("open_time").head(k)
    try: auc = roc_auc_score(te.y, p)
    except Exception: auc = np.nan
    return {"n": len(te), "base": base, "auc": auc, "top5": top.y.mean(), "lift": top.y.mean() / base if base else np.nan}

def main():
    df = fetch()
    for c in BASE: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df.open_time, utc=True); df = df[~df.already_moved.astype(bool)].copy()
    df = df.join(btc_daily(), on="open_time"); df = df.dropna(subset=NEW)
    df["blk"] = ((df.open_time >= "2024-11-01").astype(int) + (df.open_time >= "2025-11-01").astype(int))
    df["trend"] = np.where(df.btc_ret_30d > 0.03, "BTC صاعد (30د>+3%)", np.where(df.btc_ret_30d < -0.03, "BTC هابط (30د<-3%)", "BTC جانبي"))
    df["sma"] = np.where(df.btc_vs_sma200 > 0, "فوق SMA200", "تحت SMA200")
    print("rows", len(df), df.open_time.min(), df.open_time.max())
    print("\nتوزيع الأيام حسب النظام:"); print(df.groupby("trend").agg(n=("symbol", "size"), b10=("y_10_24", "mean")).round(4).to_string()); print(df.groupby("sma").agg(n=("symbol", "size"), b10=("y_10_24", "mean")).round(4).to_string())
    for t in ("y_10_24", "y_25_48", "y_40_72"):
        P = {"base": pd.Series(index=df.index, dtype=float), "plus": pd.Series(index=df.index, dtype=float)}
        for b in (0, 1, 2):
            te = df[df.blk == b]; lo, hi = te.open_time.min(), te.open_time.max(); emb = pd.Timedelta(days=3)
            tr = df[(df.blk != b) & ~((df.open_time > lo - emb) & (df.open_time < hi + emb))]
            P["base"].loc[te.index] = fit(tr, BASE, t).predict_proba(te[BASE])[:, 1]; P["plus"].loc[te.index] = fit(tr, BASE + NEW, t).predict_proba(te[BASE + NEW])[:, 1]
        print(f"\n===== هدف {t}: leave-one-year-out =====")
        rows = []
        for name in ("base", "plus"):
            r = metrics(df, P[name], t); r["model"] = "الحالي" if name == "base" else "+اتجاه BTC طويل"; r["slice"] = "الكل"; rows.append(r)
        for col in ("trend", "sma"):
            for v, g in df.groupby(col):
                for name in ("base", "plus"):
                    r = metrics(g, P[name].loc[g.index], t); r["model"] = "الحالي" if name == "base" else "+اتجاه BTC طويل"; r["slice"] = v; rows.append(r)
        print(pd.DataFrame(rows)[["slice", "model", "n", "base", "auc", "top5", "lift"]].round(3).to_string(index=False))
        if t == "y_10_24":
            rows = []
            for b in (0, 1, 2):
                g = df[df.blk == b]
                for name in ("base", "plus"):
                    r = metrics(g, P[name].loc[g.index], t); r["model"] = name; r["year"] = f"Y{b+1}"; rows.append(r)
            print("\nلكل سنة:"); print(pd.DataFrame(rows)[["year", "model", "auc", "top5", "lift"]].round(3).to_string(index=False))
main()
