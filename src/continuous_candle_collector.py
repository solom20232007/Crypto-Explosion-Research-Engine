"""Hourly Binance Spot candle ingestion into Supabase (research-only, no trading)."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.historical_surge_scanner import COLS, discover_target_symbols, fetch_klines, get_json
from src.news_monitor import _supabase_request

TIMEFRAME = "1h"
INTERVAL_MS = 3_600_000


def _utc(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def collect_hourly_candles(
    *,
    supabase_url=None,
    secret_key=None,
    lookback_hours=72,
    max_symbols=None,
    pause_seconds=0.08,
    now=None,
    upload=True,
    output_dir="reports/continuous_candles",
):
    """Fetch closed 1h candles for active Binance Spot USDT pairs and upsert them.

    Re-fetching a rolling 72h window makes the job restart-safe and repairs recent
    gaps. The candles primary key (symbol,timeframe,open_time) makes writes
    idempotent. This is a current-universe dataset and carries survivorship bias.
    """
    if not 2 <= int(lookback_hours) <= 720:
        raise ValueError("lookback_hours must be between 2 and 720")
    if max_symbols is not None and int(max_symbols) < 1:
        raise ValueError("max_symbols must be positive")

    end = _utc(now or datetime.now(timezone.utc))
    # Only request fully closed hourly candles. The current forming candle is excluded.
    end_exclusive = end.floor("h")
    start = end_exclusive - pd.Timedelta(hours=int(lookback_hours))
    start_ms, end_ms = int(start.timestamp() * 1000), int(end_exclusive.timestamp() * 1000)

    info = get_json("/api/v3/exchangeInfo")
    symbols = discover_target_symbols(info)
    # The historical scanner intentionally excludes BTC/ETH/SOL; continuous
    # collection includes them because they are important market-regime controls.
    by_symbol = {
        str(item.get("symbol", "")).upper(): item
        for item in info.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("quoteAsset") == "USDT"
        and item.get("isSpotTradingAllowed", True)
        and str(item.get("symbol", "")).upper().endswith("USDT")
    }
    excluded_bases = {"USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USDD"}
    universe = set(symbols)
    for symbol, item in by_symbol.items():
        base = str(item.get("baseAsset", "")).upper()
        if base in {"BTC", "ETH", "SOL"}:
            universe.add(symbol)
        if base in excluded_bases or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            universe.discard(symbol)
        if any(tag in base for tag in ("2L", "2S", "3L", "3S", "5L", "5S")):
            universe.discard(symbol)
    symbols = sorted(universe)
    if max_symbols is not None:
        symbols = symbols[: int(max_symbols)]
    if not symbols:
        raise RuntimeError("Binance returned no eligible Spot USDT symbols")

    frames, failures = [], []
    for index, symbol in enumerate(symbols, 1):
        try:
            frame = fetch_klines(symbol, TIMEFRAME, start_ms, end_ms, pause_seconds)
            if not frame.empty:
                frame.insert(0, "symbol", symbol)
                frames.append(frame)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        if index % 50 == 0:
            print(json.dumps({"progress": f"{index}/{len(symbols)}", "rows": sum(len(x) for x in frames)}))

    columns = ["symbol"] + list(COLS)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if data.empty:
        raise RuntimeError("Binance hourly candle collection returned zero rows; refusing empty upload")
    data["open_time"] = pd.to_numeric(data["open_time"], errors="coerce")
    data["close_time"] = pd.to_numeric(data["close_time"], errors="coerce")
    data = data.dropna(subset=["open_time", "close_time"])
    data = data[(data["open_time"] >= start_ms) & (data["close_time"] < int(end.timestamp() * 1000))]
    data = data.drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    if data.empty:
        raise RuntimeError("No fully closed hourly candles returned; refusing empty upload")

    records = []
    for row in data.to_dict("records"):
        open_time = pd.to_datetime(int(row["open_time"]), unit="ms", utc=True).isoformat()
        close_time = pd.to_datetime(int(row["close_time"]), unit="ms", utc=True).isoformat()
        records.append({
            "symbol": str(row["symbol"]),
            "timeframe": TIMEFRAME,
            "open_time": open_time,
            "close_time": close_time,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "quote_volume": float(row["quote_volume"]) if pd.notna(row.get("quote_volume")) else None,
            "trades": int(row["trade_count"]) if pd.notna(row.get("trade_count")) else None,
        })

    if upload:
        url = (supabase_url or os.getenv("SUPABASE_URL", "")).strip()
        key = (secret_key or os.getenv("SUPABASE_SECRET_KEY", "")).strip()
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required")
        for offset in range(0, len(records), 500):
            _supabase_request(
                url, key, "POST", "candles",
                params={"on_conflict": "symbol,timeframe,open_time"},
                rows=records[offset:offset + 500],
                prefer="resolution=merge-duplicates,return=minimal",
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "hourly_candles_latest.csv"
    data.to_csv(csv_path, index=False)
    summary = {
        "collected_at_utc": end.isoformat(),
        "timeframe": TIMEFRAME,
        "requested_lookback_hours": int(lookback_hours),
        "window_start_utc": start.isoformat(),
        "closed_candle_cutoff_utc": end_exclusive.isoformat(),
        "universe": "active Binance Spot USDT pairs, including BTC/ETH/SOL; stablecoins and leveraged tokens excluded",
        "survivorship_bias_warning": "This live current-universe feed cannot reconstruct delisted historical pairs.",
        "symbols_requested": len(symbols),
        "symbols_with_candles": int(data["symbol"].nunique()),
        "symbols_with_errors": len(failures),
        "candle_rows_fetched": len(records),
        "supabase_upserted": bool(upload),
        "idempotency_key": ["symbol", "timeframe", "open_time"],
        "failed_symbols": failures,
        "csv_output": str(csv_path),
    }
    (out / "collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-hours", type=int, default=72)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/continuous_candles")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    collect_hourly_candles(
        lookback_hours=args.lookback_hours,
        max_symbols=args.max_symbols,
        upload=not args.no_upload,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
