from pathlib import Path
import json
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
FEATURES_PATH = Path("data/features.csv")
THRESHOLD = 0.15

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(df, symbol):
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df = df.sort_values("open_time").drop_duplicates("open_time")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    out = pd.DataFrame({"timestamp": df["open_time"], "symbol": symbol, "close": c, "high": h, "low": l, "volume": v, "quote_volume": df["quote_volume"]})
    for n in [1, 3, 6, 12, 24, 48, 72]:
        out[f"ret_{n}h"] = c.pct_change(n)
    out["rsi_14"] = _rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    prev_close = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_close).abs(), (l-prev_close).abs()], axis=1).max(axis=1)
    out["atr_14_pct"] = tr.rolling(14).mean() / c.replace(0, np.nan)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    out["bb_width_20"] = (4 * std20) / sma20.replace(0, np.nan)
    out["bb_position_20"] = (c - (sma20 - 2*std20)) / (4*std20).replace(0, np.nan)
    out["sma_ratio_20"] = c / sma20.replace(0, np.nan) - 1
    out["sma_ratio_50"] = c / c.rolling(50).mean().replace(0, np.nan) - 1
    vol_mean20 = v.rolling(20).mean()
    out["volume_ratio_20"] = v / vol_mean20.replace(0, np.nan)
    out["quote_volume_log"] = np.log1p(df["quote_volume"].clip(lower=0))
    out["quote_volume_ratio_20"] = df["quote_volume"] / df["quote_volume"].rolling(20).mean().replace(0, np.nan)
    out["volatility_24h"] = out["ret_1h"].rolling(24).std()
    # Historical event detector: rolling close-to-close return across 24 hourly bars.
    out["historical_24h_return"] = c / c.shift(24) - 1
    out["explosion_15pct_24h"] = (out["historical_24h_return"] >= THRESHOLD).astype("int8")
    # Prediction label: at timestamp t, can the future high in t+1..t+24 reach +15% from close at t?
    future_high = pd.concat([h.shift(-i) for i in range(1, 25)], axis=1).max(axis=1)
    out["target_future_15pct_24h"] = (future_high / c - 1 >= THRESHOLD).astype("float")
    # Last 24 rows have incomplete future horizon and must not be labelled negative.
    out.loc[out.index[-min(24, len(out)):], "target_future_15pct_24h"] = np.nan
    return out

def analyze_symbols(symbols=None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_features = []
    files = sorted(RAW_DIR.glob("*_*.csv"))
    if symbols:
        allowed = set(symbols)
        files = [p for p in files if p.name.split("_")[0] in allowed]
    for path in files:
        symbol = path.name.split("_")[0]
        try:
            df = pd.read_csv(path)
            required = {"open_time", "open", "high", "low", "close", "volume", "quote_volume"}
            if not required.issubset(df.columns):
                print(f"[WARN] Missing columns in {path}; skipped")
                continue
            feats = build_features(df, symbol)
            all_features.append(feats)
            print(f"Analyzed {symbol}: {len(feats)} rows")
        except Exception as exc:
            print(f"[WARN] Failed {path}: {exc}")
    if not all_features:
        print("No raw data found. Run `python main.py collect ...` first.")
        return
    result = pd.concat(all_features, ignore_index=True)
    result.to_csv(FEATURES_PATH, index=False)
    explosions = result[result["explosion_15pct_24h"] == 1].copy()
    explosions.to_csv(REPORT_DIR / "explosions.csv", index=False)
    summary = {
        "symbols_analyzed": int(result["symbol"].nunique()),
        "rows": int(len(result)),
        "explosion_rows": int(len(explosions)),
        "date_min": str(result["timestamp"].min()),
        "date_max": str(result["timestamp"].max()),
        "definition": "24h close-to-close return >= 15%",
        "note": "Event rows can overlap; future labels use next 24 hourly highs. Research output only; not trading signals."
    }
    (REPORT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
