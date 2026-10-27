
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Binance public market data
BASE = "https://data.binance.vision/data/spot/monthly/klines"
KLINES_API = "https://data-api.binance.vision/api/v3/klines"
EXCHANGE_INFO = "https://data-api.binance.vision/api/v3/exchangeInfo"

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

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

# Supported intervals and their approximate lengths.
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Crypto-Explosion-Research-Engine/0.2"}
)

retry_strategy = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)

adapter = HTTPAdapter(max_retries=retry_strategy)
SESSION.mount("https://", adapter)


def _utc_timestamp(value) -> pd.Timestamp:
    """Convert a date or timestamp to UTC."""
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def _months(start: str, end: str) -> list[str]:
    start_period = _utc_timestamp(start).to_period("M")
    end_period = _utc_timestamp(end).to_period("M")

    return [
        str(period)
        for period in pd.period_range(
            start_period,
            end_period,
            freq="M",
        )
    ]


def _normalise_timestamp(series: pd.Series) -> pd.Series:
    """
    Binance archive timestamps may be in milliseconds or microseconds.
    Detect the unit from the numeric magnitude.
    """
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()

    if valid.empty:
        return pd.to_datetime(values, utc=True, errors="coerce")

    median = valid.median()
    unit = "us" if median > 10**14 else "ms"

    return pd.to_datetime(
        values,
        unit=unit,
        utc=True,
        errors="coerce",
    )


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Binance candle columns and remove invalid rows."""
    if df.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = df.copy()

    # Archive CSVs normally have 12 columns, but validate safely.
    if len(df.columns) < 6:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = df.iloc[:, :len(KLINE_COLUMNS)]

    columns = KLINE_COLUMNS[:len(df.columns)]
    df.columns = columns

    if "open_time" not in df.columns:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df["open_time"] = _normalise_timestamp(df["open_time"])
    df = df.dropna(subset=["open_time"])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure the output schema stays consistent even if an archive
    # contains fewer columns than expected.
    for col in KLINE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[KLINE_COLUMNS]
    df = df.drop_duplicates(subset=["open_time"])
    df = df.sort_values("open_time")

    return df.reset_index(drop=True)


def save_universe(output="data/universe.csv"):
    """Save currently trading Binance Spot USDT pairs."""
    response = SESSION.get(EXCHANGE_INFO, timeout=30)
    response.raise_for_status()

    items = response.json().get("symbols", [])

    rows = [
        {
            "symbol": item["symbol"],
            "base_asset": item.get("baseAsset", ""),
            "quote_asset": item.get("quoteAsset", ""),
            "status": item.get("status", ""),
            "is_spot_trading_allowed": item.get(
                "isSpotTradingAllowed", False
            ),
        }
        for item in items
        if item.get("quoteAsset") == "USDT"
        and item.get("status") == "TRADING"
        and item.get("isSpotTradingAllowed", False)
    ]

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("symbol")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Universe saved: {len(df)} symbols -> {output_path}")
    return str(output_path)


def _download_month(
    symbol: str,
    interval: str,
    month: str,
) -> pd.DataFrame | None:
    """Download and parse one monthly Binance archive."""
    url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"

    try:
        response = SESSION.get(url, timeout=60)

        # A missing archive can be normal for a newly listed coin.
        if response.status_code == 404:
            return None

        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_names:
                print(f"[WARN] No CSV inside archive: {symbol} {month}")
                return None

            with archive.open(csv_names[0]) as file:
                df = pd.read_csv(file, header=None)

        if df.empty:
            return None

        return _prepare_dataframe(df)

    except (
        requests.RequestException,
        zipfile.BadZipFile,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        # Do not silently treat a network failure as a successful download.
        raise RuntimeError(
            f"Failed downloading {symbol} {interval} {month}: {exc}"
        ) from exc


def _download_recent_api(
    symbol: str,
    interval: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch recent candles using Binance API.

    end_ts is exclusive. API requests are paginated so that the function
    can retrieve more than 1,000 candles when necessary.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(
            f"Unsupported interval for API pagination: {interval}"
        )

    if start_ts >= end_ts:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    interval_ms = INTERVAL_MS[interval]
    cursor_ms = int(start_ts.timestamp() * 1000)
    end_ms_exclusive = int(end_ts.timestamp() * 1000)

    rows = []

    while cursor_ms < end_ms_exclusive:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor_ms,
            "endTime": end_ms_exclusive - 1,
            "limit": 1000,
        }

        response = SESSION.get(
            KLINES_API,
            params=params,
            timeout=45,
        )
        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        rows.extend(batch)

        last_open_ms = int(batch[-1][0])
        next_cursor = last_open_ms + interval_ms

        # Safety guard against a pagination loop.
        if next_cursor <= cursor_ms:
            raise RuntimeError(
                f"API pagination did not advance for {symbol}."
            )

        cursor_ms = next_cursor

        if len(batch) < 1000:
            break

        time.sleep(0.10)

    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = _prepare_dataframe(df)

    df = df[
        (df["open_time"] >= start_ts)
        & (df["open_time"] < end_ts)
    ]

    return df.reset_index(drop=True)


def _month_is_complete(
    existing: pd.DataFrame,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
    interval_ms: int,
) -> bool:
    """Check whether saved candles adequately cover the requested period."""
    if existing.empty:
        return False

    month_data = existing.loc[
        (existing["open_time"] >= required_start)
        & (existing["open_time"] < required_end),
        "open_time",
    ]

    if month_data.empty:
        return False

    interval = pd.Timedelta(milliseconds=interval_ms)

    # A partial month, a newly listed symbol, or a missing section
    # should be downloaded again instead of being skipped.
    if month_data.min() > required_start + interval:
        return False

    if month_data.max() < required_end - 2 * interval:
        return False

    expected = max(
        1,
        int((required_end - required_start) / interval),
    )

    return len(month_data) >= expected * 0.95


def collect_symbols(
    symbols,
    interval="1h",
    start="2023-10-01",
    end=None,
):
    """
    Initial backfill plus incremental collection.

    - Downloads historical closed months from Binance archives.
    - Skips historical months that already have adequate coverage.
    - Refreshes the current month through Binance's klines API.
    - Merges old and new candles without duplicate open_time values.
    - Never stores candles later than the requested end/current time.

    The end date is inclusive when passed as a date string.
    """
    if not symbols:
        raise ValueError("No symbols supplied to collect_symbols().")

    if interval not in INTERVAL_MS:
        raise ValueError(
            f"Unsupported interval: {interval}. "
            f"Supported: {', '.join(INTERVAL_MS)}"
        )

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    start_ts = _utc_timestamp(start)
    now = pd.Timestamp.now(tz="UTC")

    if end is None:
        end_ts = now
    else:
        requested_end = _utc_timestamp(end)

        # A date-only end is inclusive through that UTC day.
        if len(str(end).strip()) == 10:
            requested_end += pd.Timedelta(days=1)

        end_ts = min(requested_end, now)

    if start_ts >= end_ts:
        raise ValueError(
            f"Invalid date range: start={start_ts}, end={end_ts}"
        )

    interval_ms = INTERVAL_MS[interval]
    interval_delta = pd.Timedelta(milliseconds=interval_ms)

    months = _months(
        start_ts.strftime("%Y-%m-%d"),
        end_ts.strftime("%Y-%m-%d"),
    )

    current_month = now.strftime("%Y-%m")

    print(f"Interval: {interval}")
    print(f"Start UTC: {start_ts}")
    print(f"End UTC (exclusive): {end_ts}")
    print(f"Months to inspect: {len(months)}")
    print(f"Symbols: {len(symbols)}")

    failures = []

    for index, symbol in enumerate(symbols, start=1):
        symbol = str(symbol).strip().upper()

        if not symbol:
            continue

        output = raw_dir / f"{symbol}_{interval}.csv"

        if output.exists():
            try:
                existing = pd.read_csv(output)
                existing = _prepare_dataframe(existing)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read existing file {output}: {exc}"
                ) from exc
        else:
            existing = pd.DataFrame(columns=KLINE_COLUMNS)

        chunks = []
        downloaded_rows = 0
        missing_archives = 0

        print(
            f"\n[{index}/{len(symbols)}] {symbol}: "
            f"existing rows={len(existing)}"
        )

        for month in months:
            month_start = pd.Timestamp(
                f"{month}-01",
                tz="UTC",
            )
            month_end = month_start + pd.offsets.MonthBegin(1)

            required_start = max(start_ts, month_start)
            required_end = min(end_ts, month_end)

            if required_start >= required_end:
                continue

            # Current month is always refreshed from the API.
            # This avoids relying on an archive that may not be ready yet.
            if month == current_month:
                continue

            if _month_is_complete(
                existing=existing,
                month_start=month_start,
                month_end=month_end,
                required_start=required_start,
                required_end=required_end,
                interval_ms=interval_ms,
            ):
                continue

            try:
                month_df = _download_month(symbol, interval, month)
            except Exception as exc:
                failures.append(str(exc))
                print(f"[ERROR] {exc}")
                continue

            if month_df is None or month_df.empty:
                missing_archives += 1
                continue

            month_df = month_df[
                (month_df["open_time"] >= required_start)
                & (month_df["open_time"] < required_end)
            ]

            if not month_df.empty:
                chunks.append(month_df)
                downloaded_rows += len(month_df)

            time.sleep(0.12)

        # Fetch the entire requested portion of the current month.
        # Usually this is at most a few hundred hourly candles.
        current_month_start = pd.Timestamp(
            f"{current_month}-01",
            tz="UTC",
        )
        recent_start = max(start_ts, current_month_start)
        recent_end = min(end_ts, now)

        if recent_start < recent_end:
            try:
                recent_df = _download_recent_api(
                    symbol=symbol,
                    interval=interval,
                    start_ts=recent_start,
                    end_ts=recent_end,
                )

                if not recent_df.empty:
                    chunks.append(recent_df)
                    downloaded_rows += len(recent_df)

            except Exception as exc:
                failures.append(
                    f"{symbol} recent API data: {exc}"
                )
                print(
                    f"[ERROR] Could not refresh recent data "
                    f"for {symbol}: {exc}"
                )

        if not existing.empty:
            chunks.insert(0, existing)

        if not chunks:
            print(
                f"[WARN] {symbol}: no usable data available. "
                f"Missing archives encountered: {missing_archives}"
            )
            continue

        combined = pd.concat(chunks, ignore_index=True)
        combined = _prepare_dataframe(combined)

        combined = combined[
            (combined["open_time"] >= start_ts)
            & (combined["open_time"] < end_ts)
        ]

        combined = (
            combined.drop_duplicates(subset=["open_time"], keep="last")
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        # Do not replace a valid saved file with an empty result.
        if combined.empty:
            print(f"[WARN] {symbol}: merged data is empty; old file retained.")
            continue

        combined.to_csv(output, index=False)

        latest = combined["open_time"].max()

        print(
            f"[OK] {symbol}: saved {len(combined)} rows; "
            f"{downloaded_rows} rows fetched; "
            f"latest={latest}; "
            f"missing monthly archives={missing_archives}; "
            f"file={output}"
        )

    if failures:
        message = (
            f"Collection finished with {len(failures)} error(s). "
            "Some symbols or months may be incomplete. "
            "Check the ERROR lines above before trusting the research results."
        )
        raise RuntimeError(message)

    print("\nCollection completed without reported download errors.")
