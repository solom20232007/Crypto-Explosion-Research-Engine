
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
]


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse ISO dates or Unix timestamps in milliseconds/microseconds."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()

    if len(non_null) > 0 and len(non_null) >= max(
        1, int(len(series.dropna()) * 0.8)
    ):
        median = non_null.abs().median()

        if median >= 1e14:
            unit = "us"
        elif median >= 1e11:
            unit = "ms"
        elif median >= 1e9:
            unit = "s"
        else:
            unit = None

        if unit:
            return pd.to_datetime(
                numeric,
                unit=unit,
                utc=True,
                errors="coerce",
            )

    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    )


def audit_file(path: Path, interval: str) -> dict:
    """Inspect one candle CSV and return a compact quality report."""
    result = {
        "file": str(path),
        "symbol": path.name.split("_")[0],
        "interval": interval,
        "status": "ERROR",
        "rows": 0,
        "valid_rows": 0,
        "first_candle_utc": "",
        "last_candle_utc": "",
        "duplicate_timestamps": 0,
        "invalid_timestamps": 0,
        "missing_values": 0,
        "invalid_ohlc_rows": 0,
        "negative_volume_rows": 0,
        "gap_count": 0,
        "estimated_missing_candles": 0,
        "largest_gap_hours": 0.0,
        "findings": "",
    }

    if not path.exists():
        result["findings"] = "File does not exist"
        return result

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        result["findings"] = f"CSV read failed: {exc}"
        return result

    result["rows"] = len(df)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        result["findings"] = (
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        return result

    if df.empty:
        result["findings"] = "File is empty"
        return result

    df = df.copy()

    df["open_time"] = parse_timestamps(df["open_time"])

    result["invalid_timestamps"] = int(
        df["open_time"].isna().sum()
    )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    value_columns = [
        column for column in REQUIRED_COLUMNS
        if column != "open_time"
    ]

    result["missing_values"] = int(
        df[value_columns].isna().sum().sum()
    )

    valid_time = df.dropna(subset=["open_time"]).copy()

    result["duplicate_timestamps"] = int(
        valid_time["open_time"].duplicated().sum()
    )

    valid_time = valid_time.sort_values("open_time")

    if not valid_time.empty:
        result["first_candle_utc"] = (
            valid_time["open_time"].iloc[0].isoformat()
        )
        result["last_candle_utc"] = (
            valid_time["open_time"].iloc[-1].isoformat()
        )

    complete = valid_time.dropna(
        subset=["open", "high", "low", "close", "volume"]
    ).copy()

    result["valid_rows"] = len(complete)

    # OHLC consistency checks:
    # prices must be positive, high must be the maximum,
    # and low must be the minimum of the candle.
    invalid_ohlc = (
        (complete["open"] <= 0)
        | (complete["high"] <= 0)
        | (complete["low"] <= 0)
        | (complete["close"] <= 0)
        | (complete["high"] < complete["low"])
        | (
            complete["high"]
            < complete[["open", "close"]].max(axis=1)
        )
        | (
            complete["low"]
            > complete[["open", "close"]].min(axis=1)
        )
    )

    result["invalid_ohlc_rows"] = int(invalid_ohlc.sum())

    result["negative_volume_rows"] = int(
        (complete["volume"] < 0).sum()
    )

    interval_delta = pd.Timedelta(interval)

    if len(valid_time) >= 2:
        differences = valid_time["open_time"].diff().dropna()

        # Allow small timestamp irregularities, but count gaps
        # larger than one candle interval.
        gap_differences = differences[
            differences > interval_delta * 1.5
        ]

        result["gap_count"] = int(len(gap_differences))

        missing_estimate = (
            gap_differences / interval_delta
        ).apply(lambda value: max(0, int(round(value)) - 1))

        result["estimated_missing_candles"] = int(
            missing_estimate.sum()
        )

        if not gap_differences.empty:
            result["largest_gap_hours"] = round(
                gap_differences.max().total_seconds() / 3600,
                2,
            )

    findings = []

    if result["invalid_timestamps"]:
        findings.append("invalid timestamps")

    if result["duplicate_timestamps"]:
        findings.append("duplicate timestamps")

    if result["missing_values"]:
        findings.append("missing numeric values")

    if result["invalid_ohlc_rows"]:
        findings.append("inconsistent OHLC prices")

    if result["negative_volume_rows"]:
        findings.append("negative volume")

    if result["gap_count"]:
        findings.append("time gaps detected")

    result["status"] = "REVIEW" if findings else "PASS"
    result["findings"] = "; ".join(findings) if findings else "No issues detected"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audit historical crypto candle data quality."
    )

    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to audit, e.g. BTCUSDT ETHUSDT SOLUSDT",
    )

    parser.add_argument(
        "--interval",
        default="1h",
        help="Candle interval, e.g. 1m, 15m, 1h, 4h, 1d",
    )

    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing symbol_interval.csv files",
    )

    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where audit reports will be written",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        files = [
            input_dir / f"{symbol.upper()}_{args.interval}.csv"
            for symbol in args.symbols
        ]
    else:
        files = sorted(
            input_dir.glob(f"*_{args.interval}.csv")
        )

    if not files:
        raise SystemExit(
            f"No CSV files found in {input_dir} "
            f"for interval {args.interval}"
        )

    reports = [
        audit_file(path, args.interval)
        for path in files
    ]

    report = pd.DataFrame(reports)

    csv_path = output_dir / "data_quality_report.csv"
    md_path = output_dir / "data_quality_report.md"

    report.to_csv(csv_path, index=False)

    summary_columns = [
        "symbol",
        "status",
        "rows",
        "valid_rows",
        "first_candle_utc",
        "last_candle_utc",
        "duplicate_timestamps",
        "invalid_timestamps",
        "missing_values",
        "invalid_ohlc_rows",
        "negative_volume_rows",
        "gap_count",
        "estimated_missing_candles",
        "largest_gap_hours",
        "findings",
    ]

    summary = report[summary_columns]

    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Historical Data Quality Audit\n\n")
        file.write(f"- Interval: `{args.interval}`\n")
        file.write(f"- Files checked: {len(report)}\n")
        file.write(
            f"- Passed: {(report['status'] == 'PASS').sum()}\n"
        )
        file.write(
            f"- Needs review: {(report['status'] == 'REVIEW').sum()}\n"
        )
        file.write(
            f"- Errors: {(report['status'] == 'ERROR').sum()}\n\n"
        )
        file.write(summary.to_markdown(index=False))
        file.write("\n")

    print("\n========== DATA QUALITY AUDIT ==========")
    print(summary.to_string(index=False))
    print(f"\nCSV report: {csv_path}")
    print(f"Markdown report: {md_path}")

    if (report["valid_rows"] == 0).all():
        raise SystemExit(
            "Audit failed: no valid candle rows were found."
        )


if __name__ == "__main__":
    main()
