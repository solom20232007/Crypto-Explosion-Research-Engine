"""Live Binance order-book liquidity snapshots for research; never places orders."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASES = ("https://data-api.binance.vision", "https://api.binance.com")
EXCLUDED = {"BTC", "ETH", "SOL", "USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USDD", "EUR", "TRY", "BRL", "GBP", "AUD", "BIDR", "IDRT", "UAH", "RUB", "ZAR", "PLN", "RON", "ARS", "JPY"}
LEVERAGED = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "5L", "5S")
OUT = Path("reports/orderbook_liquidity")


def get_json(path: str, params=None, session=requests):
    last = None
    for base in BASES:
        for attempt in range(4):
            try:
                response = session.get(base + path, params=params, timeout=20)
                if response.status_code == 451:
                    last = requests.HTTPError("HTTP 451", response=response)
                    break
                if response.status_code in (418, 429):
                    time.sleep(max(int(response.headers.get("Retry-After", 2 ** attempt)), 1))
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)
    raise last or RuntimeError("Binance API unavailable")


def eligible_symbols(exchange_info):
    symbols = []
    for item in exchange_info.get("symbols", []):
        base = str(item.get("baseAsset", "")).upper()
        symbol = str(item.get("symbol", "")).upper()
        if item.get("status") != "TRADING" or item.get("quoteAsset") != "USDT":
            continue
        if not item.get("isSpotTradingAllowed", True) or base in EXCLUDED:
            continue
        if base.endswith(LEVERAGED) or any(token in base for token in ("2L", "2S", "3L", "3S", "5L", "5S")):
            continue
        if symbol.endswith("USDT"):
            symbols.append(symbol)
    return sorted(set(symbols))


def _depth_usdt(levels, mid, band_pct, side):
    boundary = mid * (1 + band_pct / 100) if side == "asks" else mid * (1 - band_pct / 100)
    total = 0.0
    for price, quantity in levels:
        price, quantity = float(price), float(quantity)
        if (side == "asks" and price <= boundary) or (side == "bids" and price >= boundary):
            total += price * quantity
    return total


def _slippage_bps(levels, notional_usdt, mid, side):
    remaining = float(notional_usdt)
    filled_qty = 0.0
    spent = 0.0
    for price, quantity in levels:
        price, quantity = float(price), float(quantity)
        available_quote = price * quantity
        take_quote = min(remaining, available_quote)
        if take_quote > 0:
            spent += take_quote
            filled_qty += take_quote / price
            remaining -= take_quote
        if remaining <= 1e-8:
            break
    if remaining > max(0.01, notional_usdt * 1e-6) or filled_qty <= 0:
        return None
    average_price = spent / filled_qty
    move = (average_price / mid - 1) if side == "asks" else (1 - average_price / mid)
    return round(move * 10000, 4)


def calculate_metrics(symbol, book, captured_at, notionals=(1000, 5000)):
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        raise ValueError(f"Empty order book for {symbol}")
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        raise ValueError(f"Invalid best prices for {symbol}")
    mid = (best_bid + best_ask) / 2
    bid_qty = sum(float(level[1]) for level in bids)
    ask_qty = sum(float(level[1]) for level in asks)
    total_qty = bid_qty + ask_qty
    row = {
        "symbol": symbol,
        "captured_at_utc": captured_at,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_bps": round((best_ask - best_bid) / mid * 10000, 4),
        "depth_bid_usdt_0_5pct": round(_depth_usdt(bids, mid, 0.5, "bids"), 2),
        "depth_ask_usdt_0_5pct": round(_depth_usdt(asks, mid, 0.5, "asks"), 2),
        "depth_bid_usdt_1pct": round(_depth_usdt(bids, mid, 1.0, "bids"), 2),
        "depth_ask_usdt_1pct": round(_depth_usdt(asks, mid, 1.0, "asks"), 2),
        "bid_ask_qty_imbalance": round((bid_qty - ask_qty) / total_qty, 6) if total_qty else None,
        "book_limit": len(bids),
        "data_kind": "live_snapshot_not_historical",
    }
    for notional in notionals:
        suffix = str(int(notional))
        row[f"buy_slippage_bps_{suffix}_usdt"] = _slippage_bps(asks, notional, mid, "asks")
        row[f"sell_slippage_bps_{suffix}_usdt"] = _slippage_bps(bids, notional, mid, "bids")
    return row


def collect(max_symbols=100, min_quote_volume=100000, pause=0.08, output_dir=OUT):
    info = get_json("/api/v3/exchangeInfo")
    eligible = set(eligible_symbols(info))
    tickers = get_json("/api/v3/ticker/24hr")
    liquid = []
    for ticker in tickers:
        symbol = ticker.get("symbol")
        if symbol not in eligible:
            continue
        try:
            quote_volume = float(ticker.get("quoteVolume", 0))
        except (TypeError, ValueError):
            continue
        if quote_volume >= min_quote_volume:
            liquid.append((symbol, quote_volume))
    liquid.sort(key=lambda item: item[1], reverse=True)
    selected = liquid[:max(1, int(max_symbols))]
    captured = datetime.now(timezone.utc).isoformat()
    rows, errors = [], []
    for index, (symbol, quote_volume) in enumerate(selected):
        try:
            book = get_json("/api/v3/depth", {"symbol": symbol, "limit": 100})
            row = calculate_metrics(symbol, book, captured)
            row["quote_volume_24h_usdt"] = round(quote_volume, 2)
            rows.append(row)
        except Exception as exc:  # retain per-symbol failures for auditability
            errors.append({"symbol": symbol, "error": str(exc)[:300]})
        if index + 1 < len(selected):
            time.sleep(max(0.0, pause))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"orderbook_snapshot_{stamp}.csv"
    json_path = output_dir / f"orderbook_snapshot_{stamp}.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = {
        "captured_at_utc": captured,
        "symbols_eligible": len(eligible),
        "symbols_meeting_volume_floor": len(liquid),
        "symbols_requested": len(selected),
        "symbols_captured": len(rows),
        "symbols_with_errors": len(errors),
        "min_quote_volume_usdt": min_quote_volume,
        "book_limit": 100,
        "historical_depth_available": False,
        "warning": "Order-book snapshots are live observations and cannot be used as historical pre-breakout features unless separately archived with trustworthy timestamps.",
        "errors": errors,
        "csv_file": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"} | {"errors_sample": errors[:10]}, indent=2))
    if not rows:
        raise RuntimeError("No order-book snapshots collected; inspect API access and errors")
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("MAX_SYMBOLS", "100")))
    parser.add_argument("--min-quote-volume", type=float, default=float(os.getenv("MIN_QUOTE_VOLUME", "100000")))
    parser.add_argument("--pause", type=float, default=0.08)
    args = parser.parse_args()
    collect(args.max_symbols, args.min_quote_volume, args.pause)


if __name__ == "__main__":
    main()
