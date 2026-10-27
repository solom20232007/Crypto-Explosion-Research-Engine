
import os
import json
from pathlib import Path

import pandas as pd
import requests


def upload_results():
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SECRET_KEY"]

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    def upload(table, rows, conflict=None):
        if not rows:
            print(f"{table}: no rows to upload")
            return

        endpoint = f"{url}/rest/v1/{table}"
        params = {"on_conflict": conflict} if conflict else {}

        for start in range(0, len(rows), 500):
            batch = rows[start:start + 500]
            response = requests.post(
                endpoint,
                headers=headers,
                params=params,
                json=batch,
                timeout=60,
            )
            if not response.ok:
                raise RuntimeError(
                    f"{table} upload failed: HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )

        print(f"{table}: uploaded {len(rows)} rows")

    # Upload raw hourly candles
    candle_rows = []
    for path in sorted(Path("data/raw").glob("*.csv")):
        try:
            df = pd.read_csv(path)
            required = {"open_time", "open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                print(f"Skipping {path}: missing required columns")
                continue

            symbol = path.name.split("_")[0]
            timeframe = path.stem.split("_")[-1]

            df["open_time"] = pd.to_datetime(
                df["open_time"], utc=True, errors="coerce"
            )
            df = df.dropna(subset=["open_time"])

            for _, row in df.iterrows():
                candle_rows.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "open_time": row["open_time"].isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "quote_volume": (
                        float(row["quote_volume"])
                        if "quote_volume" in df.columns
                        and pd.notna(row["quote_volume"]) else None
                    ),
                    "trades": (
                        int(row["trades"])
                        if "trades" in df.columns and pd.notna(row["trades"])
                        else None
                    ),
                })
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    upload(
        "candles",
        candle_rows,
        "symbol,timeframe,open_time",
    )

    # Upload historical explosion events when the report exists
    report = Path("reports/explosions.csv")
    event_rows = []

    if report.exists():
        df = pd.read_csv(report)

        if {"timestamp", "symbol"}.issubset(df.columns):
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], utc=True, errors="coerce"
            )

            for _, row in df.dropna(subset=["timestamp"]).iterrows():
                event_rows.append({
                    "symbol": str(row["symbol"]),
                    "timeframe": "1h",
                    "event_time": row["timestamp"].isoformat(),
                    "close_price": (
                        float(row["close"])
                        if "close" in df.columns and pd.notna(row["close"])
                        else None
                    ),
                    "change_24h_pct": (
                        float(row["historical_24h_return"]) * 100
                        if "historical_24h_return" in df.columns
                        and pd.notna(row["historical_24h_return"])
                        else None
                    ),
                    "features": {},
                })

    upload(
        "explosion_events",
        event_rows,
        "symbol,timeframe,event_time",
    )

    # Store a summary of this upload
    summary_path = Path("reports/analysis_summary.json")
    summary = {}

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = {}

    run = {
        "status": "success",
        "symbols": sorted({r["symbol"] for r in candle_rows}),
        "timeframe": "1h",
        "candles_downloaded": len(candle_rows),
        "explosion_count": len(event_rows),
        "summary": summary,
    }

    response = requests.post(
        f"{url}/rest/v1/research_runs",
        headers={
            **headers,
            "Prefer": "return=minimal",
        },
        json=run,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            f"research_runs insert failed: HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    print("Supabase upload completed successfully.")
