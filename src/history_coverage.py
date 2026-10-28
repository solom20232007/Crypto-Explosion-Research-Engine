"""Audit historical candle coverage before training the pre-breakout model."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
DEFAULT_START = "2023-10-01T00:00:00Z"
DEFAULT_END = "2026-10-28T00:00:00Z"


def audit_history(start: str = DEFAULT_START, end: str = DEFAULT_END,
                  raw_dir: Path = RAW_DIR, report_dir: Path = REPORT_DIR) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    if end_ts <= start_ts:
        raise ValueError("end must be later than start")

    files = sorted(raw_dir.glob("*_1h.csv"))
    rows = []
    for path in files:
        try:
            df = pd.read_csv(path, usecols=["open_time"])
            timestamps = pd.to_datetime(df["open_time"], utc=True, errors="coerce").dropna()
            timestamps = timestamps.drop_duplicates().sort_values()
            if timestamps.empty:
                rows.append({
                    "symbol": path.stem.rsplit("_", 1)[0],
                    "rows": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "expected_hourly_rows": int((end_ts - start_ts).total_seconds() // 3600),
                    "coverage_pct": 0.0,
                    "missing_hour_estimate": int((end_ts - start_ts).total_seconds() // 3600),
                    "three_year_coverage": False,
                    "status": "empty_or_invalid",
                })
                continue
            in_range = timestamps[(timestamps >= start_ts) & (timestamps < end_ts)]
            first = in_range.min() if not in_range.empty else pd.NaT
            last = in_range.max() if not in_range.empty else pd.NaT
            expected = int((end_ts - start_ts).total_seconds() // 3600)
            observed = int(in_range.nunique())
            coverage = min(100.0, observed / expected * 100.0) if expected else 0.0
            rows.append({
                "symbol": path.stem.rsplit("_", 1)[0],
                "rows": observed,
                "first_timestamp": None if pd.isna(first) else first.isoformat(),
                "last_timestamp": None if pd.isna(last) else last.isoformat(),
                "expected_hourly_rows": expected,
                "coverage_pct": round(coverage, 3),
                "missing_hour_estimate": max(0, expected - observed),
                "three_year_coverage": bool(
                    not in_range.empty
                    and first <= start_ts + pd.Timedelta(days=7)
                    and last >= end_ts - pd.Timedelta(days=7)
                    and coverage >= 95.0
                ),
                "status": "ok" if observed else "no_data_in_range",
            })
        except Exception as exc:
            rows.append({
                "symbol": path.stem.rsplit("_", 1)[0],
                "status": "read_error",
                "error": f"{type(exc).__name__}: {exc}",
            })

    table = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "history_coverage.csv"
    json_path = report_dir / "history_coverage.json"
    table.to_csv(csv_path, index=False)
    summary = {
        "requested_start_utc": start_ts.isoformat(),
        "requested_end_exclusive_utc": end_ts.isoformat(),
        "symbols_found": len(rows),
        "symbols_with_95pct_three_year_coverage": sum(
            bool(row.get("three_year_coverage")) for row in rows
        ),
        "symbols_with_any_data": sum(
            int(row.get("rows", 0) or 0) > 0 for row in rows
        ),
        "expected_hourly_rows_per_symbol": int(
            (end_ts - start_ts).total_seconds() // 3600
        ),
        "csv_report": str(csv_path),
        "json_report": str(json_path),
        "symbols": rows,
        "warning": (
            "Coverage is measured from files present locally. This audit does not "
            "prove that delisted symbols are included or that every gap is harmless."
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "symbols"}, indent=2))
    print(f"Detailed coverage report: {csv_path}")
    return summary


if __name__ == "__main__":
    audit_history()
