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
    f["date_utc"] = pd.to_datetime(f.open_time, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
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
