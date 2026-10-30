"""Upload collected Binance 1h candles to the existing Supabase candles table."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count",
]
BATCH_SIZE = 500


def credentials() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SECRET_KEY in GitHub Actions secrets.")
    return url, key


def iso_timestamp(value: object) -> str:
    # Binance CSV timestamps are Unix milliseconds.
    return pd.to_datetime(int(float(value)), unit="ms", utc=True).isoformat()


def upload_file(path: Path, session: requests.Session, endpoint: str, headers: dict) -> int:
    symbol = path.name.removesuffix("_1h.csv").upper()
    frame = pd.read_csv(path)
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")
    frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    rows = []
    for row in frame.to_dict(orient="records"):
        open_time = iso_timestamp(row["open_time"])
        rows.append({
            "symbol": symbol,
            "timeframe": "1h",
            "open_time": open_time,
            "close_time": iso_timestamp(row["close_time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "quote_volume": float(row["quote_volume"]),
            "trades": int(float(row["trade_count"])),
        })
    uploaded = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        response = session.post(
            endpoint,
            params={"on_conflict": "symbol,timeframe,open_time"},
            headers={**headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=batch,
            timeout=90,
        )
        if not response.ok:
            raise RuntimeError(
                f"Supabase upsert failed for {symbol} batch {start // BATCH_SIZE + 1}: "
                f"HTTP {response.status_code}: {response.text[:600]}"
            )
        uploaded += len(batch)
        print(f"[OK] {symbol}: uploaded/upserted {uploaded}/{len(rows)} rows")
    return uploaded


def main() -> None:
    url, key = credentials()
    endpoint = f"{url}/rest/v1/candles"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    files = [Path(f"data/raw/{symbol}_1h.csv") for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Expected collector output files not found: {missing}")
    total = 0
    with requests.Session() as session:
        for path in files:
            count = upload_file(path, session, endpoint, headers)
            total += count
    print(f"[DONE] Upserted {total} candle rows across {len(files)} symbols.")


if __name__ == "__main__":
    main()
