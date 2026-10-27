
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

    # 1. Upload historical candles.
    candle_rows = []

    for path in sorted(Path("data/raw").glob("*.csv")):
        try:
            df = pd.read_csv(path)

            required = {
                "open_time", "open", "high", "low", "close", "volume"
            }

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
                        and pd.notna(row["quote_volume"])
                        else None
                    ),
                    "trades": (
                        int(row["trades"])
                        if "trades" in df.columns
                        and pd.notna(row["trades"])
                        else None
                    ),
                })

        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    upload("candles", candle_rows, "symbol,timeframe,open_time")

    # 2. Upload historical explosion events.
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
                        if "close" in df.columns
                        and pd.notna(row["close"])
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

    # 3. Read the analysis summary.
    summary_path = Path("reports/analysis_summary.json")
    summary = {}

    if summary_path.exists():
        try:
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            summary = {}

    # 4. Read model evaluation results.
    metrics_path = Path("reports/model_metrics.json")

    if not metrics_path.exists():
        raise RuntimeError(
            "Missing reports/model_metrics.json. "
            "Run `python main.py train` before uploading."
        )

    try:
        model_metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Could not read model metrics: {exc}"
        ) from exc

    required_metric_keys = [
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "roc_auc",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_score_at_0_5",
        "threshold",
    ]

    missing_keys = [
        name for name in required_metric_keys
        if name not in model_metrics
    ]

    if missing_keys:
        raise RuntimeError(
            "Model metrics file is missing required fields: "
            + ", ".join(missing_keys)
            + ". Run the updated training code first."
        )

    # 5. Create a research run and retrieve its database ID.
    run = {
        "status": "running",
        "symbols": sorted({r["symbol"] for r in candle_rows}),
        "timeframe": "1h",
        "candles_downloaded": len(candle_rows),
        "explosion_count": len(event_rows),
        "summary": summary,
    }

    run_response = requests.post(
        f"{url}/rest/v1/research_runs",
        headers={
            **headers,
            "Prefer": "return=representation",
        },
        json=run,
        timeout=30,
    )

    if not run_response.ok:
        raise RuntimeError(
            "research_runs insert failed: "
            f"HTTP {run_response.status_code}: "
            f"{run_response.text[:1000]}"
        )

    try:
        run_data = run_response.json()
        run_id = run_data[0]["id"]
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Could not retrieve the new research_runs ID. "
            "Check that the API returns the inserted row."
        ) from exc

    print(f"research_runs: created run {run_id}")

    # 6. Save model metrics and link them to this research run.
    trained_at = model_metrics.get("trained_at_utc")

    if not trained_at:
        trained_at = pd.Timestamp.now(tz="UTC").isoformat()

    metric_row = {
        "model_name": model_metrics.get(
            "model_name", "explosion_baseline"
        ),
        "trained_at": trained_at,
        "train_start": model_metrics["train_start"],
        "train_end": model_metrics["train_end"],
        "test_start": model_metrics["test_start"],
        "test_end": model_metrics["test_end"],
        "precision_score": model_metrics["precision_at_0_5"],
        "recall_score": model_metrics["recall_at_0_5"],
        "f1_score": model_metrics["f1_score_at_0_5"],
        "roc_auc": model_metrics["roc_auc"],
        "threshold": model_metrics["threshold"],
        "metrics": model_metrics,
        "run_id": run_id,
    }

    upload("model_metrics", [metric_row])

    # 7. Mark the research run successful only after all uploads finish.
    update_response = requests.patch(
        f"{url}/rest/v1/research_runs",
        headers={
            **headers,
            "Prefer": "return=minimal",
        },
        params={"id": f"eq.{run_id}"},
        json={"status": "success"},
        timeout=30,
    )

    if not update_response.ok:
        raise RuntimeError(
            "research_runs status update failed: "
            f"HTTP {update_response.status_code}: "
            f"{update_response.text[:1000]}"
        )

    print("Supabase upload completed successfully.")
