
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.technical_indicators import add_technical_indicators

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
FEATURES_PATH = Path("data/features.csv")

THRESHOLD = 0.15
FUTURE_HOURS = 24

REQUIRED_COLUMNS = {
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
}


def _rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def build_features(df, symbol):
    df = df.copy()

    # Normalize timestamps and numeric columns.
    raw_open_time = df["open_time"]
    numeric_open_time = pd.to_numeric(raw_open_time, errors="coerce")
    if numeric_open_time.notna().mean() >= 0.95:
        # Binance REST klines use Unix epoch milliseconds. Without unit="ms",
        # pandas interprets these integers as nanoseconds and shifts dates to 1970.
        df["open_time"] = pd.to_datetime(
            numeric_open_time, unit="ms", utc=True, errors="coerce"
        )
    else:
        df["open_time"] = pd.to_datetime(
            raw_open_time, utc=True, errors="coerce"
        )

    df = (
        df.dropna(subset=["open_time"])
        .sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Invalid prices cannot be used for reliable feature calculation.
    df = df[
        (df["close"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
    ].copy()

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid candle rows remain after cleaning.")

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    qv = df["quote_volume"]

    out = pd.DataFrame({
        "timestamp": df["open_time"],
        "symbol": symbol,
        "close": c,
        "high": h,
        "low": l,
        "volume": v,
        "quote_volume": qv,
    })

    # Price momentum.
    for n in [1, 3, 6, 12, 24, 48, 72]:
        out[f"ret_{n}h"] = c.pct_change(n)

    # RSI.
    out["rsi_14"] = _rsi(c, 14)

    # MACD.
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # ATR percentage.
    previous_close = c.shift(1)

    true_range = pd.concat(
        [
            h - l,
            (h - previous_close).abs(),
            (l - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["atr_14_pct"] = (
        true_range.rolling(14).mean()
        / c.replace(0, np.nan)
    )

    # Bollinger Bands.
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    band_width = (4 * std20).replace(0, np.nan)

    out["bb_width_20"] = band_width / sma20.replace(0, np.nan)
    out["bb_position_20"] = (
        c - (sma20 - 2 * std20)
    ) / band_width

    out["sma_ratio_20"] = (
        c / sma20.replace(0, np.nan) - 1
    )

    out["sma_ratio_50"] = (
        c / c.rolling(50).mean().replace(0, np.nan) - 1
    )

    # Volume indicators.
    volume_mean20 = v.rolling(20).mean()
    quote_volume_mean20 = qv.rolling(20).mean()

    out["volume_ratio_20"] = (
        v / volume_mean20.replace(0, np.nan)
    )

    out["quote_volume_log"] = np.log1p(qv.clip(lower=0))

    out["quote_volume_ratio_20"] = (
        qv / quote_volume_mean20.replace(0, np.nan)
    )

    # Recent volatility.
    out["volatility_24h"] = out["ret_1h"].rolling(24).std()

    # Broad causal technical indicators and current-state breakout phase labels.
    # This runs after timestamp sorting/cleaning, so output rows remain aligned.
    out = pd.concat([out, add_technical_indicators(df)], axis=1)

    # Historical event:
    # Did the closing price rise at least 15% over the preceding 24 bars?
    out["historical_24h_return"] = c / c.shift(24) - 1

    out["explosion_15pct_24h"] = (
        out["historical_24h_return"] >= THRESHOLD
    ).astype("int8")

    # Prediction target:
    # Can any high in the NEXT 24 bars reach 15% above the current close?
    future_high = pd.concat(
        [h.shift(-i) for i in range(1, FUTURE_HOURS + 1)],
        axis=1,
    ).max(axis=1)

    out["target_future_15pct_24h"] = (
        future_high / c - 1 >= THRESHOLD
    ).astype(float)

    # Do not incorrectly label incomplete future windows as negatives.
    incomplete_count = min(FUTURE_HOURS, len(out))

    out.loc[
        out.index[-incomplete_count:],
        "target_future_15pct_24h",
    ] = np.nan

    return out


def analyze_symbols(symbols=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove old outputs so a failed run cannot leave stale results
    # that might accidentally be used for model training.
    for path in [
        FEATURES_PATH,
        REPORT_DIR / "explosions.csv",
        REPORT_DIR / "analysis_summary.json",
    ]:
        if path.exists():
            path.unlink()

    files = sorted(RAW_DIR.glob("*_1h.csv"))

    if symbols:
        allowed = set(symbols)

        files = [
            path
            for path in files
            if path.stem.rsplit("_", 1)[0] in allowed
        ]

    if not files:
        raise FileNotFoundError(
            f"No matching raw CSV files found in {RAW_DIR.resolve()}. "
            f"Requested symbols: {symbols or 'all'}. "
            "Run the collect command and check its output."
        )

    print(f"Raw files selected: {len(files)}")

    all_features = []
    failed_files = []

    for path in files:
        symbol = path.stem.rsplit("_", 1)[0]

        try:
            df = pd.read_csv(path)

            # Normalize header whitespace.
            df.columns = df.columns.astype(str).str.strip()

            missing = REQUIRED_COLUMNS - set(df.columns)

            if missing:
                message = (
                    f"{path}: missing columns {sorted(missing)}"
                )
                failed_files.append(message)
                print(f"[ERROR] {message}")
                continue

            features = build_features(df, symbol)

            if features.empty:
                message = f"{path}: feature output is empty"
                failed_files.append(message)
                print(f"[ERROR] {message}")
                continue

            all_features.append(features)

            print(
                f"[OK] {symbol}: "
                f"{len(features)} rows analyzed"
            )

        except Exception as exc:
            message = f"{path}: {type(exc).__name__}: {exc}"
            failed_files.append(message)
            print(f"[ERROR] {message}")

    if not all_features:
        details = "\n".join(failed_files) or "No usable data."
        raise RuntimeError(
            "Analysis produced no usable features. "
            "Training must not continue.\n" + details
        )

    result = pd.concat(all_features, ignore_index=True)

    result = result.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    # Validate before writing output.
    if "target_future_15pct_24h" not in result.columns:
        raise RuntimeError("Prediction target column was not created.")

    result.to_csv(FEATURES_PATH, index=False)

    explosions = result[
        result["explosion_15pct_24h"] == 1
    ].copy()

    explosions.to_csv(
        REPORT_DIR / "explosions.csv",
        index=False,
    )

    valid_targets = result["target_future_15pct_24h"].dropna()

    summary = {
        "symbols_analyzed": int(result["symbol"].nunique()),
        "rows": int(len(result)),
        "explosion_rows": int(len(explosions)),
        "valid_future_labels": int(len(valid_targets)),
        "future_positive_rows": int(valid_targets.sum()),
        "date_min": str(result["timestamp"].min()),
        "date_max": str(result["timestamp"].max()),
        "definition": (
            "Historical event: 24-bar close-to-close return >= 15%"
        ),
        "prediction_target": (
            "Any high in the next 24 bars reaches >= 15% "
            "above the current close"
        ),
        "technical_feature_columns": int(len(result.columns)),
        "breakout_phase_rows": {
            str(phase): int(count)
            for phase, count in result["breakout_phase"].value_counts(dropna=False).items()
        },
        "failed_files": failed_files,
        "note": (
            "Research output only; not trading signals. "
            "Historical event detection and future prediction "
            "target are separate definitions."
        ),
    }

    summary_path = REPORT_DIR / "analysis_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if not FEATURES_PATH.exists() or FEATURES_PATH.stat().st_size == 0:
        raise RuntimeError(
            "Analysis finished without creating a valid features.csv."
        )

    print("\nAnalysis completed successfully.")
    print(f"Features file: {FEATURES_PATH}")
    print(f"Features rows: {len(result)}")
    print(f"Explosion rows: {len(explosions)}")

    if failed_files:
        print(
            f"[WARN] {len(failed_files)} raw file(s) were skipped. "
            "See the log above."
        )

    print(json.dumps(summary, indent=2))

    return summary
