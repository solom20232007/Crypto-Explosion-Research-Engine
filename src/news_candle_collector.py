"""Fetch a recent daily-candle universe for timestamp-safe news features."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

from src.historical_surge_scanner import discover_target_symbols, fetch_klines, get_json, COLS


def collect_recent_daily_candles(output_path="reports/surge_scan/daily_candles.csv",
                                 days=30, max_symbols=None, pause_seconds=0.08):
    """Collect recent daily candles for currently active Binance Spot USDT pairs.

    This is a current-universe snapshot (survivorship bias applies); it is not a
    replacement for the full historical research scan.
    """
    if int(days) < 2 or int(days) > 365:
        raise ValueError("days must be between 2 and 365")
    symbols = discover_target_symbols(get_json("/api/v3/exchangeInfo"))
    if max_symbols is not None:
        if int(max_symbols) < 1:
            raise ValueError("max_symbols must be positive")
        symbols = symbols[:int(max_symbols)]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(days))
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    parts, failures = [], []
    for index, symbol in enumerate(symbols, 1):
        try:
            candles = fetch_klines(symbol, "1d", start_ms, end_ms, pause_seconds)
            if not candles.empty:
                candles.insert(0, "symbol", symbol)
                parts.append(candles)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        if index % 50 == 0:
            print(json.dumps({"daily_candle_progress": f"{index}/{len(symbols)}"}))
    columns = ["symbol"] + list(COLS)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if result.empty:
        raise RuntimeError("Binance daily-candle collection returned zero rows; refusing to create empty features")
    result = result.drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    summary = {
        "collected_at_utc": end.isoformat().replace("+00:00", "Z"),
        "requested_days": int(days),
        "universe": "currently active Binance Spot USDT pairs excluding BTC/ETH/SOL, stablecoins and leveraged tokens",
        "survivorship_bias_warning": "Current exchangeInfo omits delisted pairs; this recent snapshot is not a survivorship-bias-free historical universe.",
        "symbols_requested": len(symbols),
        "symbols_with_errors": len(failures),
        "candle_rows": len(result),
        "symbols_with_candles": int(result.symbol.nunique()),
        "failed_symbols": failures,
        "output": str(path),
    }
    summary_path = path.parent / "news_candle_collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/surge_scan/daily_candles.csv")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()
    collect_recent_daily_candles(args.output, args.days, args.max_symbols)


if __name__ == "__main__":
    main()
