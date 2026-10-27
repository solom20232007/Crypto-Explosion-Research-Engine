from __future__ import annotations
import io
import time
import zipfile
from pathlib import Path
from datetime import date
import requests
import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines"
EXCHANGE_INFO = "https://data-api.binance.vision/api/v3/exchangeInfo"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Crypto-Explosion-Research-Engine/0.1"})

def save_universe(output="data/universe.csv"):
    response = SESSION.get(EXCHANGE_INFO, timeout=30)
    response.raise_for_status()
    items = response.json().get("symbols", [])
    rows = [
        {
            "symbol": x["symbol"],
            "base_asset": x.get("baseAsset", ""),
            "quote_asset": x.get("quoteAsset", ""),
            "status": x.get("status", ""),
            "is_spot_trading_allowed": x.get("isSpotTradingAllowed", False),
        }
        for x in items
        if x.get("quoteAsset") == "USDT"
        and x.get("status") == "TRADING"
        and x.get("isSpotTradingAllowed", False)
    ]
    df = pd.DataFrame(rows).sort_values("symbol")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output

def _months(start: str, end: str):
    s = pd.Timestamp(start).to_period("M")
    e = pd.Timestamp(end).to_period("M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]

def _normalise_timestamp(series):
    # Binance public archive uses milliseconds historically and microseconds
    # from 2025-01-01 onward for Spot datasets.
    vals = pd.to_numeric(series, errors="coerce")
    median = vals.dropna().median()
    unit = "us" if median > 10**14 else "ms"
    return pd.to_datetime(vals, unit=unit, utc=True, errors="coerce")

def _download_month(symbol: str, interval: str, month: str):
    url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    try:
        r = SESSION.get(url, timeout=45)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f, header=None)
        if df.empty:
            return None
        # Some archive files may contain a header; discard it by numeric conversion.
        df = df.iloc[:, :12]
        df.columns = KLINE_COLUMNS[:len(df.columns)]
        df["open_time"] = _normalise_timestamp(df["open_time"])
        df = df.dropna(subset=["open_time"])
        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                    "trades", "taker_buy_base", "taker_buy_quote"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except requests.RequestException as exc:
        print(f"[WARN] {symbol} {month}: {exc}")
        return None
    except (zipfile.BadZipFile, ValueError, pd.errors.ParserError) as exc:
        print(f"[WARN] Could not parse {symbol} {month}: {exc}")
        return None

def collect_symbols(symbols, interval="1h", start="2023-10-01", end="2026-10-27"):
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    start_ts = pd.Timestamp(start, tz="UTC")
    months = _months(start, end)
    for i, symbol in enumerate(symbols, 1):
        output = raw_dir / f"{symbol}_{interval}.csv"
        existing = pd.DataFrame()
        if output.exists():
            try:
                existing = pd.read_csv(output, parse_dates=["open_time"])
                existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
            except Exception:
                existing = pd.DataFrame()
        known = set(existing["open_time"].astype(str)) if not existing.empty else set()
        chunks = [existing] if not existing.empty else []
        downloaded = 0
        for month in months:
            # Skip a month if its first and last hours already exist in the saved data.
            if not existing.empty:
                month_start = pd.Timestamp(month, tz="UTC")
                month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
                have = existing.loc[
                    (existing["open_time"] >= month_start) &
                    (existing["open_time"] <= month_end),
                    "open_time"
                ]
                # Skip only when the saved month appears to cover both ends.
                if (not have.empty and
                    have.min() <= month_start + pd.Timedelta(hours=1) and
                    have.max() >= month_end - pd.Timedelta(hours=2)):
                    continue
            df = _download_month(symbol, interval, month)
            if df is not None and not df.empty:
                chunks.append(df)
                downloaded += len(df)
            time.sleep(0.12)  # polite archive access; adjust only with care
        if chunks:
            combined = pd.concat(chunks, ignore_index=True)
            combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True, errors="coerce")
            combined = combined.dropna(subset=["open_time"])
            combined = combined[(combined["open_time"] >= start_ts) & (combined["open_time"] < end_ts)]
            combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time")
            combined.to_csv(output, index=False)
            print(f"[{i}/{len(symbols)}] {symbol}: saved {len(combined)} rows ({downloaded} newly downloaded) -> {output}")
        else:
            print(f"[{i}/{len(symbols)}] {symbol}: no data downloaded")
