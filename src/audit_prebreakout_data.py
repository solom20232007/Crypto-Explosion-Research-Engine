"""Audit historical candles and pre-breakout dataset quality before modeling."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FORBIDDEN_CANDIDATES = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def audit(candles: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    required = {"symbol", "open_time", "open", "high", "low", "close", "volume", "quote_volume"}
    missing = sorted(required - set(candles.columns))
    if missing:
        raise ValueError(f"Missing candle columns: {missing}")
    c = candles.copy()
    for col in ("open_time", "open", "high", "low", "close", "volume", "quote_volume"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    invalid_ohlc = (
        (c[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (c.high < c[["open", "low", "close"]].max(axis=1))
        | (c.low > c[["open", "high", "close"]].min(axis=1))
        | (c.volume < 0) | (c.quote_volume < 0)
    )
    duplicate_keys = int(c.duplicated(["symbol", "open_time"], keep=False).sum())
    valid = c.dropna(subset=["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    valid["open_time_dt"] = pd.to_datetime(valid.open_time, unit="ms", utc=True, errors="coerce")
    gap_rows = []
    for symbol, group in valid.groupby("symbol", sort=False):
        times = group.open_time.dropna().astype("int64").sort_values()
        if len(times) > 1:
            diffs = times.diff().dropna()
            gap_rows.append({
                "symbol": str(symbol),
                "rows": int(len(times)),
                "gaps_over_1d": int((diffs > 86_400_000).sum()),
                "largest_gap_days": float(diffs.max() / 86_400_000) if len(diffs) else 0.0,
            })
    d = dataset.copy()
    symbols = set(d.symbol.astype(str)) if "symbol" in d else set()
    future_leakage_checks = {}
    for col in ("explosion_1d", "explosion_3d", "explosion_7d"):
        future_leakage_checks[col] = bool(col in d.columns and d[col].dropna().isin([0, 1]).all())
    split_dates = {}
    if {"split", "date_utc"}.issubset(d.columns):
        for split, part in d.groupby("split"):
            split_dates[str(split)] = {
                "first": str(part.date_utc.min()), "last": str(part.date_utc.max()),
                "rows": int(len(part)),
            }
    suspicious = c[(c.high / c[["open", "close", "low"]].max(axis=1) >= 5) | (c[["open", "close", "high"]].min(axis=1) / c.low >= 5)]
    return {
        "candle_rows": int(len(c)),
        "dataset_rows": int(len(d)),
        "symbols": int(c.symbol.nunique()),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "duplicate_symbol_open_time_rows": duplicate_keys,
        "missing_quote_volume_rows": int(c.quote_volume.isna().sum()),
        "negative_or_zero_volume_rows": int(((c.volume <= 0) | (c.quote_volume <= 0)).sum()),
        "symbols_with_gaps_over_1d": int(sum(any(x["symbol"] == s and x["gaps_over_1d"] > 0 for x in gap_rows) for s in c.symbol.dropna().unique())),
        "gap_summary_top_20": sorted(gap_rows, key=lambda x: (x["gaps_over_1d"], x["largest_gap_days"]), reverse=True)[:20],
        "extreme_wick_rows_review_required": int(len(suspicious)),
        "forbidden_candidate_symbols_present": sorted(FORBIDDEN_CANDIDATES.intersection(symbols)),
        "split_date_ranges": split_dates,
        "label_values_binary_or_missing": future_leakage_checks,
        "limitations": [
            "Gap counts identify missing daily candles, not necessarily collector errors; listing age and delistings need separate lifecycle data.",
            "Extreme wick checks are review flags, not automatic proof of bad prices.",
            "This audit does not establish that the model is profitable or that all historical delisted symbols are covered.",
        ],
    }


def main():
    candle_path = Path("reports/surge_scan/daily_candles.csv")
    dataset_path = Path("reports/prebreakout_dataset/prebreakout_dataset.csv")
    out_path = Path("reports/prebreakout_dataset/data_quality_audit.json")
    report = audit(pd.read_csv(candle_path), pd.read_csv(dataset_path))
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["invalid_ohlc_rows"] or report["forbidden_candidate_symbols_present"]:
        raise SystemExit("Data quality audit found invalid OHLC rows or forbidden candidate symbols.")


if __name__ == "__main__":
    main()
