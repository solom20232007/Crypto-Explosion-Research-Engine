
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


# Public Binance Spot archive endpoints.
MONTHLY_BASE = (
    "https://data.binance.vision/data/spot/monthly/klines"
)
DAILY_BASE = (
    "https://data.binance.vision/data/spot/daily/klines"
)
EXCHANGE_INFO = (
    "https://data-api.binance.vision/api/v3/exchangeInfo"
)

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

OUTPUT_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
]

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Crypto-Explosion-Research-Engine/1.0",
    }
)


def _utc_timestamp(value) -> pd.Timestamp:
    """Convert a date or timestamp to UTC."""
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def _request_with_retries(
    url: str,
    timeout: int = 60,
    attempts: int = 4,
):
    """
    Download a public archive.

    A 404 means the archive does not exist and returns None.
    No api.binance.com REST fallback is used.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = SESSION.get(url, timeout=timeout)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            print(
                f"[WARN] Request attempt {attempt}/{attempts} failed: "
                f"{url} | {exc}"
            )

            if attempt < attempts:
                time.sleep(min(2 ** attempt, 15))

    return None


def _normalise_timestamp(series: pd.Series) -> pd.Series:
    """
    Binance Spot archive timestamps can be milliseconds or
    microseconds. Detect the unit from the timestamp magnitude.
    """
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()

    if valid.empty:
        return pd.to_datetime(
            values,
            utc=True,
            errors="coerce",
        )

    median_value = valid.abs().median()
    unit = "us" if median_value >= 1e14 else "ms"

    return pd.to_datetime(
        values,
        unit=unit,
        utc=True,
        errors="coerce",
    )


def _read_kline_csv(content: bytes) -> pd.DataFrame:
    """Read and normalize a CSV contained in a Binance ZIP archive."""
    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_files = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                print("[WARN] ZIP archive contains no CSV file.")
                return empty

            with archive.open(csv_files[0]) as file:
                df = pd.read_csv(file, header=None)

    except (
        zipfile.BadZipFile,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as exc:
        print(f"[WARN] Could not read archive CSV: {exc}")
        return empty

    if df.empty or len(df.columns) < 6:
        return empty

    # Binance kline files contain 12 columns.
    df = df.iloc[:, :len(KLINE_COLUMNS)].copy()
    df.columns = KLINE_COLUMNS[:len(df.columns)]

    # Convert timestamps before filtering invalid rows.
    df["open_time"] = _normalise_timestamp(df["open_time"])

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )
        else:
            df[column] = pd.NA

    df = df.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    if df.empty:
        return empty

    return (
        df[OUTPUT_COLUMNS]
        .sort_values("open_time")
        .drop_duplicates(subset=["open_time"], keep="last")
        .reset_index(drop=True)
    )


def _download_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Download one monthly Spot kline archive."""
    month_text = f"{year:04d}-{month:02d}"

    url = (
        f"{MONTHLY_BASE}/{symbol}/{interval}/"
        f"{symbol}-{interval}-{month_text}.zip"
    )

    response = _request_with_retries(url)

    if response is None:
        print(
            f"[INFO] Monthly archive unavailable: "
            f"{symbol} {interval} {month_text}"
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = _read_kline_csv(response.content)

    if not df.empty:
        print(
            f"[OK] {symbol} {month_text}: "
            f"{len(df)} candles downloaded from monthly archive"
        )

    return df


def _download_day(
    symbol: str,
    interval: str,
    day,
) -> pd.DataFrame:
    """Download one daily Spot kline archive."""
    day_text = _utc_timestamp(day).strftime("%Y-%m-%d")

    url = (
        f"{DAILY_BASE}/{symbol}/{interval}/"
        f"{symbol}-{interval}-{day_text}.zip"
    )

    response = _request_with_retries(url)

    if response is None:
        print(
            f"[INFO] Daily archive unavailable: "
            f"{symbol} {interval} {day_text}"
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = _read_kline_csv(response.content)

    if not df.empty:
        print(
            f"[OK] {symbol} {day_text}: "
            f"{len(df)} candles downloaded from daily archive"
        )

    return df


def _load_existing(path: Path) -> pd.DataFrame:
    """Load previously collected candles, if available."""
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        df = pd.read_csv(path)

        if df.empty or "open_time" not in df.columns:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        for column in OUTPUT_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA

        df["open_time"] = pd.to_datetime(
            df["open_time"],
            utc=True,
            errors="coerce",
        )

        for column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(subset=["open_time"])

        return df[OUTPUT_COLUMNS]

    except Exception as exc:
        print(
            f"[WARN] Could not read existing file {path}: {exc}. "
            "It will be rebuilt from archives."
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _month_is_complete(
    existing: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    """
    Return True if existing data appears to cover the requested
    part of a month. 'end' is exclusive.
    """
    if existing.empty or end <= start:
        return False

    times = existing["open_time"].dropna()

    if times.empty:
        return False

    month_data = times[(times >= start) & (times < end)]

    if month_data.empty:
        return False

    # Allow a small margin for incomplete boundary candles.
    return (
        month_data.min() <= start + pd.Timedelta(hours=1)
        and month_data.max() >= end - pd.Timedelta(hours=2)
    )


def _collect_one(
    symbol: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Path:
    """Collect, merge, deduplicate, and save one symbol."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    output = raw_dir / f"{symbol}_{interval}.csv"
    existing = _load_existing(output)

    if end_ts <= start_ts:
        raise ValueError(
            f"Invalid date range: start={start_ts}, end={end_ts}"
        )

    # Work out which calendar months overlap [start_ts, end_ts).
    last_included = end_ts - pd.Timedelta(nanoseconds=1)

    periods = pd.period_range(
        start=start_ts.to_period("M"),
        end=last_included.to_period("M"),
        freq="M",
    )

    current_month = pd.Timestamp.now(tz="UTC").to_period("M")
    downloaded_frames = []
    newly_downloaded = 0

    print(f"\n[COLLECT] Symbol: {symbol}")
    print(f"[COLLECT] Interval: {interval}")
    print(f"[COLLECT] Start UTC: {start_ts}")
    print(f"[COLLECT] End UTC (exclusive): {end_ts}")

    for period in periods:
        month_start = period.start_time.tz_localize("UTC")
        month_end = (period + 1).start_time.tz_localize("UTC")

        clipped_start = max(start_ts, month_start)
        clipped_end = min(end_ts, month_end)

        if clipped_end <= clipped_start:
            continue

        # Do not repeatedly download historical months that already
        # appear complete. The current month is checked every run.
        if period != current_month and _month_is_complete(
            existing,
            clipped_start,
            clipped_end,
        ):
            print(
                f"[SKIP] {symbol} {period}: "
                "existing data appears complete"
            )
            continue

        if period == current_month:
            # The REST API may be blocked in some GitHub runners.
            # Use daily public archives for the current month instead.
            day_start = clipped_start.normalize()

            while day_start < clipped_end:
                try:
                    frame = _download_day(
                        symbol,
                        interval,
                        day_start,
                    )
                except Exception as exc:
                    print(
                        f"[WARN] Daily archive failed for "
                        f"{symbol} {day_start:%Y-%m-%d}: {exc}"
                    )
                    frame = pd.DataFrame(columns=OUTPUT_COLUMNS)

                if not frame.empty:
                    downloaded_frames.append(frame)
                    newly_downloaded += len(frame)

                day_start += pd.Timedelta(days=1)
                time.sleep(0.10)

        else:
            try:
                frame = _download_month(
                    symbol,
                    interval,
                    period.year,
                    period.month,
                )
            except Exception as exc:
                print(
                    f"[WARN] Monthly archive failed for "
                    f"{symbol} {period}: {exc}"
                )
                frame = pd.DataFrame(columns=OUTPUT_COLUMNS)

            if not frame.empty:
                downloaded_frames.append(frame)
                newly_downloaded += len(frame)

            time.sleep(0.15)

    chunks = []

    if not existing.empty:
        chunks.append(existing)

    chunks.extend(downloaded_frames)

    if not chunks:
        raise RuntimeError(
            f"No candle data available for {symbol}. "
            "Check the archive URLs, symbol, interval, and date range."
        )

    combined = pd.concat(chunks, ignore_index=True)

    combined["open_time"] = pd.to_datetime(
        combined["open_time"],
        utc=True,
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

    combined = combined.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # Respect the requested date range; end_ts is exclusive.
    combined = combined[
        (combined["open_time"] >= start_ts)
        & (combined["open_time"] < end_ts)
    ]

    combined = (
        combined[OUTPUT_COLUMNS]
        .drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    if combined.empty:
        raise RuntimeError(
            f"No candles remained for {symbol} after date filtering."
        )

    combined.to_csv(output, index=False)

    print(
        f"[SAVED] {symbol}: {len(combined)} total candles; "
        f"{newly_downloaded} downloaded this run -> {output}"
    )

    return output


def collect_symbols(
    symbols,
    interval="1h",
    start="2023-10-01",
    end="2026-10-27",
):
    """
    Collect Spot candles for one or more symbols.

    The end date/time is exclusive. For example, end='2026-10-27'
    collects candles strictly before 2026-10-27 00:00:00 UTC.
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    symbols = [
        str(symbol).strip().upper()
        for symbol in symbols
        if str(symbol).strip()
    ]

    if not symbols:
        raise ValueError("No symbols were supplied to collect_symbols().")

    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    failures = []
    saved_paths = []

    for index, symbol in enumerate(symbols, start=1):
        print(f"\n========== {index}/{len(symbols)}: {symbol} ==========")

        try:
            path = _collect_one(
                symbol=symbol,
                interval=interval,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            saved_paths.append(str(path))

        except Exception as exc:
            message = f"{symbol}: {exc}"
            failures.append(message)
            print(f"[ERROR] {message}")

    if failures:
        raise RuntimeError(
            "Collection failed for one or more symbols:\n"
            + "\n".join(failures)
        )

    print("\n[COMPLETE] All requested symbols were collected.")
    return saved_paths


def save_universe(output="data/universe.csv"):
    """
    Save Binance Spot USDT trading pairs to a CSV file.

    This function is used by: python main.py universe
    """
    response = _request_with_retries(
        EXCHANGE_INFO,
        timeout=30,
        attempts=4,
    )

    if response is None:
        raise RuntimeError(
            "Could not download Binance exchange information from "
            f"{EXCHANGE_INFO}"
        )

    payload = response.json()
    items = payload.get("symbols", [])

    rows = [
        {
            "symbol": item.get("symbol", ""),
            "base_asset": item.get("baseAsset", ""),
            "quote_asset": item.get("quoteAsset", ""),
            "status": item.get("status", ""),
            "is_spot_trading_allowed": item.get(
                "isSpotTradingAllowed",
                False,
            ),
        }
        for item in items
        if item.get("quoteAsset") == "USDT"
        and item.get("status") == "TRADING"
        and item.get("isSpotTradingAllowed", False)
    ]

    if not rows:
        raise RuntimeError(
            "Binance returned no eligible USDT Spot trading pairs."
        )

    df = pd.DataFrame(rows).sort_values("symbol")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[SAVED] {len(df)} symbols -> {output_path}")
    return str(output_path)
