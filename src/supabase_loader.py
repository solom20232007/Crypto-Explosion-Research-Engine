"""Read historical candles from Supabase into the research engine's CSV format.

This module only reads from Supabase. It never inserts, updates, or deletes rows.
Required environment variables: SUPABASE_URL and SUPABASE_SECRET_KEY
(or SUPABASE_SERVICE_ROLE_KEY / SUPABASE_KEY).
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

OUTPUT_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "quote_volume"
]
SELECT_COLUMNS = ",".join([
    "symbol", "timeframe", "open_time", "close_time", "open", "high",
    "low", "close", "volume", "quote_volume", "trades"
])
PAGE_SIZE = 1000


def _credentials():
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = (
        os.environ.get("SUPABASE_SECRET_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        raise RuntimeError(
            "Supabase credentials missing. Set SUPABASE_URL and "
            "SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY / "
            "SUPABASE_KEY). Do not commit credentials to Git."
        )
    return url, key


def fetch_candles(symbol: str, timeframe: str = "1h", session=None) -> pd.DataFrame:
    """Fetch all rows for one symbol/timeframe with stable paginated ordering."""
    url, key = _credentials()
    client = session or requests.Session()
    endpoint = f"{url}/rest/v1/candles"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Range-Unit": "items",
    }
    rows = []
    start = 0
    while True:
        headers["Range"] = f"{start}-{start + PAGE_SIZE - 1}"
        response = client.get(
            endpoint,
            headers=headers,
            params={
                "select": SELECT_COLUMNS,
                "symbol": f"eq.{symbol}",
                "timeframe": f"eq.{timeframe}",
                "order": "open_time.asc",
            },
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(
                f"Supabase candle read failed for {symbol}/{timeframe}: "
                f"HTTP {response.status_code}: {response.text[:500]}. "
                "Check API permissions and Row Level Security."
            )
        page = response.json()
        if not isinstance(page, list):
            raise RuntimeError("Supabase returned an unexpected response format.")
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += len(page)

    if not rows:
        raise RuntimeError(
            f"No candles found for {symbol}/{timeframe} in Supabase."
        )

    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    for column in OUTPUT_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=OUTPUT_COLUMNS)
    df = (
        df[OUTPUT_COLUMNS]
        .sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        raise RuntimeError(f"Supabase rows for {symbol} contained no valid candles.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"Non-positive OHLC price found for {symbol}; refusing import.")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise RuntimeError(f"Invalid high price found for {symbol}; refusing import.")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise RuntimeError(f"Invalid low price found for {symbol}; refusing import.")
    return df


def load_from_supabase(
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    output_dir: str = "data/raw",
) -> list[Path]:
    """Import selected candles to local CSV files; no database writes."""
    if not timeframe or any(ch in timeframe for ch in "/\\"):
        raise ValueError("Invalid timeframe.")
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not symbols:
        raise ValueError("Provide at least one symbol.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for symbol in symbols:
        if not symbol.isalnum():
            raise ValueError(f"Invalid symbol: {symbol!r}")
        df = fetch_candles(symbol, timeframe)
        path = destination / f"{symbol}_{timeframe}.csv"
        df.to_csv(path, index=False)
        written.append(path)
        print(
            f"[OK] Supabase -> {path}: {len(df)} candles "
            f"({df['open_time'].min()} to {df['open_time'].max()})"
        )
    return written
