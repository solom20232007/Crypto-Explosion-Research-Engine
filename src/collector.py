
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")

BINANCE_DATA_BASE = (
    "https://data.binance.vision/data/spot/monthly/klines"
)
BINANCE_API_BASE = "https://api.binance.com"

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
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

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Crypto-Explosion-Research-Engine/1.0"
})


def _utc_timestamp(value=None):
    if value is None:
        return pd.Timestamp.now(tz="UTC")

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def _parse_date(value):
    return _utc_timestamp(value)


def _request_with_retries(url, params=None, timeout=30, attempts=4):
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=timeout,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            last_error = exc

            print(
                f"[WARN] Request attempt {attempt}/{attempts} failed: "
                f"{url} - {exc}"
            )

            if attempt < attempts:
                time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(
        f"Request failed after {attempts} attempts: {url}. "
        f"Last error: {last_error}"
    )


def _read_kline_csv(content):
    """
    Read either a Binance Vision ZIP payload or CSV bytes.
    Binance Vision monthly files normally contain no header row.
    """
    if not content:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw_bytes = content

    if raw_bytes[:2] == b"PK":
        try:
            with zipfile.ZipFile(BytesIO(raw_bytes)) as archive:
                csv_names = [
                    name for name in archive.namelist()
                    if name.lower().endswith(".csv")
                ]

                if not csv_names:
                    raise ValueError(
                        "ZIP archive does not contain a CSV file."
                    )

                with archive.open(csv_names[0]) as file:
                    raw_bytes = file.read()

        except zipfile.BadZipFile as exc:
            raise ValueError("Downloaded ZIP file is invalid.") from exc

    try:
        df = pd.read_csv(
            BytesIO(raw_bytes),
            header=None,
            low_memory=False,
        )
    except Exception as exc:
        raise ValueError(f"Could not parse candle CSV: {exc}") from exc

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Binance Vision klines normally have 12 columns.
    if df.shape[1] < 7:
        raise ValueError(
            f"Expected at least 7 kline columns, found {df.shape[1]}."
        )

    df = df.iloc[:, :len(KLINE_COLUMNS)].copy()
    df.columns = KLINE_COLUMNS[:df.shape[1]]

    # Support files that unexpectedly include a header row.
    first_value = str(df.iloc[0]["open_time"]).strip().lower()

    if first_value in {"open_time", "open time", "timestamp"}:
        df = df.iloc[1:].copy()

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binance timestamps are milliseconds since the Unix epoch.
    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        ]
    )

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["open_time"])

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def _download_month(symbol, interval, year, month):
    url = (
        f"{BINANCE_DATA_BASE}/"
        f"{symbol}/{interval}/"
        f"{symbol}-{interval}-{year}-{month:02d}.zip"
    )

    response = _request_with_retries(url, timeout=60)

    if response is None:
        print(f"[INFO] Archive not available: {symbol} {year}-{month:02d}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        df = _read_kline_csv(response.content)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read monthly archive for "
            f"{symbol} {year}-{month:02d}: {exc}"
        ) from exc

    if not df.empty:
        print(
            f"[OK] {symbol} {year}-{month:02d}: "
            f"{len(df)} candles downloaded from archive"
        )

    return df


def _download_day(symbol, interval, day):
    day = pd.Timestamp(day).strftime("%Y-%m-%d")

    url = (
        f"https://data.binance.vision/data/spot/daily/klines/"
        f"{symbol}/{interval}/"
        f"{symbol}-{interval}-{day}.zip"
    )

    response = _request_with_retries(url, timeout=60)

    if response is None:
        print(f"[INFO] Daily archive not available: {symbol} {day}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = _read_kline_csv(response.content)

    if not df.empty:
        print(
            f"[OK] {symbol} {day}: "
            f"{len(df)} candles downloaded from daily archive"
        )

    return df

def _download_api(symbol, interval, start, end):
    """
    Retrieve candles from Binance REST API.
    end is exclusive for the final filtering step.
    """
    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)

    if end_ts <= start_ts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    url = f"{BINANCE_API_BASE}/api/v3/klines"

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    rows = []
    cursor_ms = start_ms
    page_count = 0

    while cursor_ms < end_ms:
        response = _request_with_retries(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor_ms,
                "endTime": end_ms - 1,
                "limit": 1000,
            },
            timeout=30,
        )

        if response is None:
            break

        try:
            batch = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Binance API returned invalid JSON for {symbol}: "
                f"{response.text[:200]}"
            ) from exc

        if isinstance(batch, dict):
            raise RuntimeError(
                f"Binance API error for {symbol}: {batch}"
            )

        if not batch:
            break

        rows.extend(batch)
        page_count += 1

        last_open_ms = int(batch[-1][0])
        next_cursor = last_open_ms + 1

        if next_cursor <= cursor_ms:
            raise RuntimeError(
                f"Pagination did not advance for {symbol}."
            )

        cursor_ms = next_cursor

        if len(batch) < 1000:
            break

        time.sleep(0.15)

    if not rows:
        print(f"[INFO] API returned no candles for {symbol}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)

    df = _read_api_dataframe(df)

    print(
        f"[OK] {symbol}: {len(df)} candles downloaded from API "
        f"across {page_count} page(s)"
    )

    return df


def _read_api_dataframe(df):
    df = df.copy()

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        ]
    )


timestamps = pd.to_numeric(df["open_time"], errors="coerce")

valid_timestamps = timestamps.dropna()

if valid_timestamps.empty:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)

# Binance Spot archive timestamps changed to microseconds in 2025.
unit = "us" if valid_timestamps.abs().median() >= 1e14 else "ms"


timestamps = pd.to_numeric(df["open_time"], errors="coerce")

valid_timestamps = timestamps.dropna()

if valid_timestamps.empty:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)

# Binance Spot archive timestamps changed to microseconds in 2025.
unit = "us" if valid_timestamps.abs().median() >= 1e14 else "ms"

df["open_time"] = pd.to_datetime(
    timestamps,
    unit=unit,
    utc=True,
    errors="coerce",
)



 return df.dropna(subset=["open_time"])[OUTPUT_COLUMNS].reset_index(
    drop=True
)

def _month_range(start, end):
    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)

    if end_ts <= start_ts:
        return []

    # Use timezone-naive month periods to avoid timezone warnings.
    start_period = start_ts.tz_localize(None).to_period("M")
    end_period = end_ts.tz_localize(None).to_period("M")

    return list(
        pd.period_range(start_period, end_period, freq="M")
    )


def _load_existing(path):
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.astype(str).str.strip()

        missing = set(OUTPUT_COLUMNS) - set(df.columns)

        if missing:
            print(
                f"[WARN] Existing file {path} is missing columns "
                f"{sorted(missing)}; it will not be used."
            )
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        df = df[OUTPUT_COLUMNS].copy()
        df["open_time"] = pd.to_datetime(
            df["open_time"],
            utc=True,
            errors="coerce",
        )

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(
            subset=["open_time", "open", "high", "low", "close"]
        )

    except Exception as exc:
        print(f"[WARN] Could not read existing file {path}: {exc}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _collect_one(symbol, interval, start, end):
    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)

    if end_ts <= start_ts:
        raise ValueError(
            f"Invalid date range for {symbol}: end must be after start."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DIR / f"{symbol}_{interval}.csv"
    existing = _load_existing(output_path)

    print(f"\nCollecting {symbol} ({interval})")
    print(f"Start UTC: {start_ts}")
    print(f"End UTC (exclusive): {end_ts}")
    print(f"Existing rows: {len(existing)}")

    downloaded_frames = []
    months = _month_range(start_ts, end_ts)
    current_period = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")

    for period in months:
        month_start = period.start_time.tz_localize("UTC")
        month_end = (period + 1).start_time.tz_localize("UTC")

        clipped_start = max(start_ts, month_start)
        clipped_end = min(end_ts, month_end)

        # Use the REST API for the current month; monthly archives may
        # not yet contain the most recent candles.

if period == current_period:
    # Use daily public archives instead of the blocked REST API.
    day_start = clipped_start.normalize()
    while day_start < clipped_end:
        day_end = day_start + pd.Timedelta(days=1)

        if day_start >= clipped_end:
            break

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

        day_start = day_end

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
            f"[WARN] Archive failed for {symbol} "
            f"{period}: {exc}"
        )
        frame = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if not frame.empty:
        downloaded_frames.append(frame)


                # Fall back to the API for this month if archive retrieval
                # fails, so a failed archive does not silently lose data.
                frame = _download_api(
                    symbol,
                    interval,
                    clipped_start,
                    clipped_end,
                )

        if not frame.empty:
            downloaded_frames.append(frame)

    if downloaded_frames:
        downloaded = pd.concat(downloaded_frames, ignore_index=True)
    else:
        downloaded = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if not downloaded.empty:
        downloaded["open_time"] = pd.to_datetime(
            downloaded["open_time"],
            utc=True,
            errors="coerce",
        )

    # Keep only the requested time range.
    if not downloaded.empty:
        downloaded = downloaded[
            (downloaded["open_time"] >= start_ts)
            & (downloaded["open_time"] < end_ts)
        ]

    if existing.empty and downloaded.empty:
        raise RuntimeError(
            f"No candle data was downloaded for {symbol}. "
            "Check Binance connectivity, symbol validity, and download logs."
        )

    combined = pd.concat(
        [existing, downloaded],
        ignore_index=True,
    )

    combined["open_time"] = pd.to_datetime(
        combined["open_time"],
        utc=True,
        errors="coerce",
    )

    combined = (
        combined.dropna(subset=["open_time"])
        .drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # Retain the full requested window rather than accidentally
    # truncating previously collected history.
    combined.to_csv(output_path, index=False)

    print(
        f"[SAVED] {symbol}: {len(combined)} total rows -> {output_path}"
    )

    if not combined.empty:
        print(
            f"[RANGE] {symbol}: "
            f"{combined['open_time'].min()} to "
            f"{combined['open_time'].max()}"
        )

    return output_path


def collect_symbols(
    symbols,
    interval="1h",
    start="2023-10-01",
    end=None,
):
    if not symbols:
        raise ValueError("At least one symbol is required.")

    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)

    if end_ts <= start_ts:
        raise ValueError("--end must be later than --start.")

    print(f"Interval: {interval}")
    print(f"Start UTC: {start_ts}")
    print(f"End UTC (exclusive): {end_ts}")
    print(f"Months to inspect: {len(_month_range(start_ts, end_ts))}")
    print(f"Symbols: {len(symbols)}")

    errors = []

    for index, symbol in enumerate(symbols, start=1):
        print(f"\n[{index}/{len(symbols)}] {symbol}")

        try:
            _collect_one(
                symbol=symbol.upper(),
                interval=interval,
                start=start_ts,
                end=end_ts,
            )
        except Exception as exc:
            message = f"{symbol}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"[ERROR] {message}")

    if errors:
        raise RuntimeError(
            "Collection failed for one or more symbols:\n"
            + "\n".join(errors)
        )

    print("\nCollection completed successfully.")


def save_universe():
    """
    Save Binance Spot USDT trading symbols to data/universe.csv.
    """
    url = f"{BINANCE_API_BASE}/api/v3/exchangeInfo"

    response = _request_with_retries(url, timeout=30)

    if response is None:
        raise RuntimeError("Binance exchangeInfo endpoint returned 404.")

    payload = response.json()

    symbols = []

    for item in payload.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", True)
        ):
            symbols.append(item["symbol"])

    if not symbols:
        raise RuntimeError(
            "Binance returned no active Spot USDT symbols."
        )

    path = Path("data/universe.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"symbol": sorted(set(symbols))}).to_csv(
        path,
        index=False,
    )

    print(f"Saved {len(symbols)} symbols to {path}")

    return path
