"""Download Binance Spot 1h klines for a configurable historical universe.

This collector uses Binance's public REST API. By default it discovers today's
active USDT Spot pairs; that is NOT a point-in-time historical universe and can
introduce survivorship bias. Supply --symbols-file with a curated historical
universe to include delisted symbols where data is available.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE_URLS = [
    "https://data-api.binance.vision",  # Same public API host used by V7 as primary.
    "https://api.binance.com",
]
KLINES_PATH = "/api/v3/klines"
EXCHANGE_INFO_PATH = "/api/v3/exchangeInfo"
LIMIT = 1000
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
]


def _request_json(path: str, params: dict | None = None, retries: int = 4):
    """Request public Binance data, trying the same alternate API host as V7."""
    session = requests.Session()
    last_error = None
    for base_url in BASE_URLS:
        url = f"{base_url}{path}"
        for attempt in range(retries):
            try:
                response = session.get(url, params=params, timeout=30)
                # A 451 is a host/access restriction; move to the next host immediately.
                if response.status_code == 451:
                    last_error = requests.HTTPError(
                        f"HTTP 451 from {base_url}; trying alternate Binance API host",
                        response=response,
                    )
                    break
                if response.status_code in (418, 429):
                    wait = int(response.headers.get("Retry-After", min(60, 2 ** attempt)))
                    time.sleep(max(wait, 1))
                    continue
                if response.status_code >= 500:
                    last_error = requests.HTTPError(
                        f"HTTP {response.status_code} from {base_url}", response=response
                    )
                    time.sleep(min(30, 2 ** attempt))
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    if exc.response.status_code == 451:
                        break
                if attempt < retries - 1:
                    time.sleep(min(30, 2 ** attempt))
        # Continue with the alternate host if this host failed.
    raise last_error or RuntimeError("No Binance API host returned data")


def discover_active_usdt_spot_symbols() -> list[str]:
    payload = _request_json(EXCHANGE_INFO_PATH)
    symbols = []
    for item in payload.get("symbols", []):
        if (item.get("status") == "TRADING"
                and item.get("quoteAsset") == "USDT"
                and item.get("isSpotTradingAllowed", True)):
            symbols.append(str(item["symbol"]))
    return sorted(set(symbols))


def read_symbols_file(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().upper()
        if not line or line.startswith("#"):
            continue
        values.append(line.split(",")[0].strip())
    return sorted(set(values))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def download_symbol(symbol: str, start_ms: int, end_ms: int, out_dir: Path,
                    pause_seconds: float = 0.15) -> dict:
    """Fetch one symbol in pages and persist each page for interruption recovery."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_1h.csv"
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            existing = pd.DataFrame(columns=COLUMNS)
    else:
        existing = pd.DataFrame(columns=COLUMNS)

    existing = existing.reindex(columns=COLUMNS)
    cursor = start_ms
    if not existing.empty:
        # Resume after the newest persisted candle. The separate coverage audit
        # is still needed to detect gaps inside the already-downloaded range.
        latest_existing = int(pd.to_numeric(existing["open_time"], errors="coerce").max())
        if latest_existing >= start_ms:
            cursor = max(start_ms, latest_existing + 60 * 60 * 1000)
    pages = 0
    fetched = 0
    while cursor < end_ms:
        payload = _request_json(KLINES_PATH, {
            "symbol": symbol, "interval": "1h", "startTime": cursor,
            "endTime": end_ms - 1, "limit": LIMIT,
        })
        if not payload:
            break
        page = pd.DataFrame(payload, columns=COLUMNS)
        if page.empty:
            break
        fetched += len(page)
        pages += 1
        existing = pd.concat([existing, page], ignore_index=True)
        existing["open_time"] = pd.to_numeric(existing["open_time"], errors="coerce")
        existing = (existing.dropna(subset=["open_time"])
                    .drop_duplicates(subset=["open_time"], keep="last")
                    .sort_values("open_time"))
        _atomic_write_csv(existing, path)
        next_cursor = int(page.iloc[-1]["open_time"]) + 60 * 60 * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(payload) < LIMIT:
            break
        time.sleep(max(0.0, pause_seconds))

    return {
        "symbol": symbol, "file": str(path), "pages_fetched": pages,
        "new_rows_received": fetched, "total_unique_rows": int(len(existing)),
        "first_open_time_ms": int(existing["open_time"].min()) if not existing.empty else None,
        "last_open_time_ms": int(existing["open_time"].max()) if not existing.empty else None,
        "status": "ok" if not existing.empty else "no_data",
    }


def collect(start: str, end: str, out_dir: Path, symbols_file: Path | None = None,
            max_symbols: int | None = None, pause_seconds: float = 0.15) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    if end_ts <= start_ts:
        raise ValueError("--end must be later than --start")
    if symbols_file:
        symbols = read_symbols_file(symbols_file)
        universe_source = f"symbols file: {symbols_file}"
    else:
        symbols = discover_active_usdt_spot_symbols()
        universe_source = "current exchangeInfo active USDT Spot pairs (survivorship-bias warning)"
    if max_symbols is not None:
        if max_symbols < 1:
            raise ValueError("--max-symbols must be positive")
        symbols = symbols[:max_symbols]

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, symbol in enumerate(symbols, start=1):
        try:
            result = download_symbol(symbol, start_ms, end_ms, out_dir, pause_seconds)
        except Exception as exc:
            result = {"symbol": symbol, "status": "error",
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(json.dumps({"progress": f"{index}/{len(symbols)}", **result}))
        # Persist a checkpoint after every symbol so interrupted runs are auditable.
        pd.DataFrame(results).to_csv(out_dir / "collection_manifest.csv", index=False)
        (out_dir / "collection_manifest.json").write_text(
            json.dumps({
                "requested_start_utc": start_ts.isoformat(),
                "requested_end_exclusive_utc": end_ts.isoformat(),
                "universe_source": universe_source,
                "symbols_requested": len(symbols),
                "symbols_ok": sum(r.get("status") == "ok" for r in results),
                "symbols_errored": sum(r.get("status") == "error" for r in results),
                "warning": (
                    "Active pairs discovered today do not include all historically listed or "
                    "delisted assets. Use a point-in-time symbol list to reduce survivorship bias."
                ),
                "results": results,
            }, indent=2), encoding="utf-8"
        )
        time.sleep(max(0.0, pause_seconds))
    summary = {
        "symbols_requested": len(symbols),
        "symbols_ok": sum(r.get("status") == "ok" for r in results),
        "symbols_errored": sum(r.get("status") == "error" for r in results),
        "output_dir": str(out_dir),
        "manifest": str(out_dir / "collection_manifest.json"),
        "universe_source": universe_source,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-10-28T00:00:00Z",
                        help="Inclusive UTC start, ISO-8601 (default: three years before 2026-10-28)")
    parser.add_argument("--end", default="2026-10-28T00:00:00Z",
                        help="Exclusive UTC end, ISO-8601")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--symbols-file", type=Path,
                        help="Optional newline-separated symbols list; can include delisted symbols")
    parser.add_argument("--max-symbols", type=int,
                        help="Optional cap for a small test run; omit to process the full selected universe")
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    args = parser.parse_args()
    summary = collect(args.start, args.end, args.out_dir, args.symbols_file,
                      args.max_symbols, args.pause_seconds)
    if summary["symbols_errored"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
