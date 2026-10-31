"""Build a leakage-safe, chronological pre-breakout dataset from daily candle history."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_dataset(candles: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "open_time", "open", "high", "low", "close", "volume", "quote_volume"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    f = candles.copy()
    for col in ("open_time", "open", "high", "low", "close", "volume", "quote_volume"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
    f = f.dropna(subset=["symbol", "open_time", "open", "high", "low", "close"])
    f = f[(f.open > 0) & (f.high > 0) & (f.low > 0) & (f.close > 0)]
    f = f.sort_values(["symbol", "open_time"]).drop_duplicates(["symbol", "open_time"]).reset_index(drop=True)
    groups = f.groupby("symbol", sort=False, group_keys=False)
    g = f.groupby("symbol", sort=False)
    prev_close = g.close.shift(1)
    dt_utc = pd.to_datetime(f.open_time, unit="ms", utc=True)
    f["date_utc"] = dt_utc.dt.strftime("%Y-%m-%d")
    # Calendar features use only the signal candle timestamp; daily bars have no intraday timing.
    dow = dt_utc.dt.dayofweek
    month = dt_utc.dt.month - 1
    f["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    f["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    first_seen = g.open_time.transform("min")
    f["days_since_first_seen"] = ((f.open_time - first_seen) / 86400000).clip(lower=0)
    # All predictors use the signal-day candle and earlier data only.
    f["return_1d_pct"] = (f.close / prev_close - 1) * 100
    f["return_3d_pct"] = (f.close / g.close.shift(3) - 1) * 100
    f["return_7d_pct"] = (f.close / g.close.shift(7) - 1) * 100
    f["return_14d_pct"] = (f.close / g.close.shift(14) - 1) * 100
    f["range_pct"] = (f.high / f.low - 1) * 100
    f["body_pct"] = (f.close / f.open - 1) * 100
    f["close_location"] = (f.close - f.low) / (f.high - f.low).replace(0, np.nan)
    f["upper_wick_pct"] = (f.high - f[["open", "close"]].max(axis=1)) / f.open * 100
    f["lower_wick_pct"] = (f[["open", "close"]].min(axis=1) - f.low) / f.open * 100
    f["range_mean_7d_pct"] = g.range_pct.transform(lambda s: s.shift(1).rolling(7, min_periods=5).mean())
    f["range_mean_30d_pct"] = g.range_pct.transform(lambda s: s.shift(1).rolling(30, min_periods=15).mean())
    f["volatility_ratio_7d_30d"] = f.range_mean_7d_pct / f.range_mean_30d_pct.replace(0, np.nan)
    f["quote_volume_mean_7d"] = g.quote_volume.transform(lambda s: s.shift(1).rolling(7, min_periods=5).mean())
    f["quote_volume_mean_30d"] = g.quote_volume.transform(lambda s: s.shift(1).rolling(30, min_periods=15).mean())
    f["quote_volume_ratio_1d_7d"] = f.quote_volume / f.quote_volume_mean_7d.replace(0, np.nan)
    f["quote_volume_ratio_1d_30d"] = f.quote_volume / f.quote_volume_mean_30d.replace(0, np.nan)
    f["quote_volume_trend_3d_vs_7d"] = g.quote_volume.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean()) / f.quote_volume_mean_7d.replace(0, np.nan)
    # Technical momentum/trend features, calculated only from candles available at signal close.
    f["ema_9"] = g.close.transform(lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
    f["ema_21"] = g.close.transform(lambda s: s.ewm(span=21, adjust=False, min_periods=21).mean())
    f["ema_50"] = g.close.transform(lambda s: s.ewm(span=50, adjust=False, min_periods=50).mean())
    f["close_vs_ema9_pct"] = (f.close / f.ema_9 - 1) * 100
    f["ema9_vs_ema21_pct"] = (f.ema_9 / f.ema_21 - 1) * 100
    f["ema21_vs_ema50_pct"] = (f.ema_21 / f.ema_50 - 1) * 100
    f["macd_pct"] = (g.close.transform(lambda s: s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()) / f.close) * 100
    f["macd_signal_pct"] = (g.close.transform(lambda s: (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).ewm(span=9, adjust=False).mean()) / f.close) * 100
    macd_hist = f["macd_pct"] - f["macd_signal_pct"]
    prev_macd_hist = macd_hist.groupby(f.symbol, sort=False).shift(1)
    f["macd_histogram_pct"] = macd_hist
    f["macd_bullish_cross"] = ((macd_hist > 0) & (prev_macd_hist <= 0)).fillna(False).astype("int8")
    f["macd_bearish_cross"] = ((macd_hist < 0) & (prev_macd_hist >= 0)).fillna(False).astype("int8")
    ema_spread = f["ema_9"] - f["ema_21"]
    prev_ema_spread = ema_spread.groupby(f.symbol, sort=False).shift(1)
    f["ema9_21_bullish_cross"] = ((ema_spread > 0) & (prev_ema_spread <= 0)).fillna(False).astype("int8")
    f["ema9_21_bearish_cross"] = ((ema_spread < 0) & (prev_ema_spread >= 0)).fillna(False).astype("int8")
    f["_bar_index"] = g.cumcount()
    for event_col in ("macd_bullish_cross", "macd_bearish_cross", "ema9_21_bullish_cross", "ema9_21_bearish_cross"):
        last_event_bar = f["_bar_index"].where(f[event_col].eq(1)).groupby(f.symbol, sort=False).ffill()
        f[f"bars_since_{event_col}"] = f["_bar_index"] - last_event_bar
    f = f.drop(columns=["_bar_index"])
    delta = g.close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(f.symbol, sort=False).transform(lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
    avg_loss = loss.groupby(f.symbol, sort=False).transform(lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    f["rsi_14"] = 100 - (100 / (1 + rs))
    prev_rsi = f["rsi_14"].groupby(f.symbol, sort=False).shift(1)
    f["rsi_cross_up_30"] = ((f.rsi_14 >= 30) & (prev_rsi < 30)).fillna(False).astype("int8")
    f["rsi_cross_up_50"] = ((f.rsi_14 >= 50) & (prev_rsi < 50)).fillna(False).astype("int8")
    f["rsi_cross_down_70"] = ((f.rsi_14 <= 70) & (prev_rsi > 70)).fillna(False).astype("int8")
    f["rsi_overbought_70"] = (f.rsi_14 >= 70).fillna(False).astype("int8")
    f["rsi_oversold_30"] = (f.rsi_14 <= 30).fillna(False).astype("int8")
    f["ema_bullish_alignment"] = ((f.ema_9 > f.ema_21) & (f.ema_21 > f.ema_50)).fillna(False).astype("int8")
    f["price_above_ema50"] = (f.close > f.ema_50).fillna(False).astype("int8")
    previous_close = g.close.shift(1)
    true_range = pd.concat([f.high - f.low, (f.high - previous_close).abs(), (f.low - previous_close).abs()], axis=1).max(axis=1)
    f["atr_14_pct"] = true_range.groupby(f.symbol, sort=False).transform(lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean()) / f.close * 100
    rolling_mean = g.close.transform(lambda s: s.rolling(20, min_periods=20).mean())
    rolling_std = g.close.transform(lambda s: s.rolling(20, min_periods=20).std())
    f["bollinger_width_20_pct"] = 4 * rolling_std / rolling_mean * 100
    f["bollinger_position_20"] = (f.close - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std).replace(0, np.nan)
    f["close_vs_prior_20d_high_pct"] = (f.close / g.high.transform(lambda s: s.shift(1).rolling(20, min_periods=10).max()) - 1) * 100
    # Liquidity proxies from candle data only; not a substitute for order-book depth/spread.
    f["quote_volume_median_7d"] = g.quote_volume.transform(lambda s: s.shift(1).rolling(7, min_periods=5).median())
    f["quote_volume_median_30d"] = g.quote_volume.transform(lambda s: s.shift(1).rolling(30, min_periods=15).median())
    f["quote_volume_zscore_30d"] = (f.quote_volume - f.quote_volume_mean_30d) / g.quote_volume.transform(lambda s: s.shift(1).rolling(30, min_periods=15).std()).replace(0, np.nan)
    if "trade_count" in f.columns:
        f["trade_count_ratio_1d_7d"] = pd.to_numeric(f["trade_count"], errors="coerce") / g.trade_count.transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(7, min_periods=5).mean()).replace(0, np.nan)
    else:
        f["trade_count_ratio_1d_7d"] = np.nan
    if "taker_buy_quote_volume" in f.columns:
        taker_buy = pd.to_numeric(f["taker_buy_quote_volume"], errors="coerce")
        f["taker_buy_quote_ratio"] = taker_buy / pd.to_numeric(f["quote_volume"], errors="coerce").replace(0, np.nan)
    else:
        f["taker_buy_quote_ratio"] = np.nan
    # Labels start strictly after the signal-day close; no same-day high/close enters a target.
    for horizon, threshold in ((1, 25), (3, 40), (7, 75)):
        future_high = g.high.transform(lambda s: pd.concat([s.shift(-i) for i in range(1, horizon + 1)], axis=1).max(axis=1, skipna=False))
        f[f"future_max_high_{horizon}d_pct"] = (future_high / f.close - 1) * 100
        f[f"explosion_{horizon}d"] = np.where(future_high.notna(), (f[f"future_max_high_{horizon}d_pct"] >= threshold).astype("int8"), np.nan)
    f["split"] = np.select(
        [f.date_utc < "2025-10-28", f.date_utc < "2026-04-28"],
        ["train", "validation"], default="test"
    )
    return f


def summarize(data: pd.DataFrame) -> dict:
    splits = {}
    for name, part in data.groupby("split", dropna=False):
        splits[str(name)] = {
            "rows": int(len(part)),
            "symbols": int(part.symbol.nunique()),
            "label_counts": {
                col: {str(k): int(v) for k, v in part[col].value_counts(dropna=True).sort_index().items()}
                for col in ("explosion_1d", "explosion_3d", "explosion_7d")
            },
        }
    return {
        "rows": int(len(data)),
        "symbols": int(data.symbol.nunique()),
        "first_date_utc": data.date_utc.min() if len(data) else None,
        "last_date_utc": data.date_utc.max() if len(data) else None,
        "label_definition": "Future high after signal-day close: >=25% in 1 day, >=40% in 3 days, >=75% in 7 days",
        "predictor_timing": "Signal is after daily candle close; predictors use signal day and prior data only; labels begin the next day",
        "chronological_splits": splits,
        "limitations": [
            "Current Binance Spot USDT universe only; delisted symbols are missing (survivorship bias).",
            "Labels overlap across adjacent days and are not independent events.",
            "No profitability, fees, slippage, or execution assumptions are modeled.",
            "This is a research dataset, not a trading signal or financial advice.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/surge_scan/daily_candles.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/prebreakout_dataset"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candles = pd.read_csv(args.input)
    data = build_dataset(candles)
    output = args.out_dir / "prebreakout_dataset.csv"
    data.to_csv(output, index=False)
    report = summarize(data)
    report["input"] = str(args.input)
    report["output"] = str(output)
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
