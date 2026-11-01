"""Build leakage-safe pre-breakout and pump-like event research rows.

Features use only Binance aggTrades strictly before each anchor.
Labels use future 1h klines and are stored separately from features.
This is research, not proof of coordinated manipulation or a trading signal.
"""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.aggtrade_prebreakout_study import (
    HOUR_MS, build_cases, extract_features, load_window_trades,
)

KLINES_URL = "https://api.binance.com/api/v3/klines"
LEAD_HOURS = (24, 12, 6)


def add_burst_features(trades, anchor_ms):
    if trades.empty:
        return {}
    t = trades.copy()
    t["timestamp"] = pd.to_numeric(t["timestamp"], errors="coerce")
    t["quote"] = pd.to_numeric(t["quote"], errors="coerce")
    t["aggressive_buy"] = ~t["is_buyer_maker"].astype(bool)
    t = t.dropna(subset=["timestamp", "quote"])
    if t.empty:
        return {}
    t["minute"] = (t["timestamp"] // 60_000).astype("int64")
    t["five_minute"] = (t["timestamp"] // 300_000).astype("int64")
    t["buy_quote"] = t["quote"].where(t["aggressive_buy"], 0.0)
    t["sell_quote"] = t["quote"].where(~t["aggressive_buy"], 0.0)

    def bins(key):
        g = t.groupby(key).agg(
            quote=("quote", "sum"),
            buy=("buy_quote", "sum"),
            sell=("sell_quote", "sum"),
            trades=("quote", "size"),
        )
        g["pressure"] = np.where(g.quote > 0, (g.buy - g.sell) / g.quote, 0.0)
        return g

    m1 = bins("minute")
    m5 = bins("five_minute")
    recent_minute = (anchor_ms - 1) // 60_000
    recent_five = (anchor_ms - 1) // 300_000
    last1 = m1[m1.index == recent_minute]
    last5 = m5[m5.index == recent_five]
    prior60 = m1[m1.index < recent_minute].tail(60)
    total72 = float(t.quote.sum())
    large_threshold = float(t.quote.quantile(0.99))
    large = t[t.quote >= large_threshold]
    large_buy = large[large.aggressive_buy]
    large_buy_5m = large_buy.groupby("five_minute").quote.sum()

    def val(frame, col):
        return float(frame.iloc[-1][col]) if len(frame) else None

    positive_pressure_bins = m5[(m5.quote >= max(float(m5.quote.median()), 1.0)) & (m5.pressure >= 0.60)]
    return {
        "burst_last_1m_quote_usdt": val(last1, "quote"),
        "burst_last_1m_buy_pressure": val(last1, "pressure"),
        "burst_last_5m_quote_usdt": val(last5, "quote"),
        "burst_last_5m_buy_pressure": val(last5, "pressure"),
        "burst_last_5m_trade_count": val(last5, "trades"),
        "burst_1m_vs_prior60m_median_ratio": (
            float(val(last1, "quote") / prior60.quote.median())
            if len(last1) and len(prior60) and prior60.quote.median() > 0 else None
        ),
        "burst_top_1m_volume_share_72h": float(m1.quote.max() / total72) if total72 > 0 else None,
        "burst_top_5m_volume_share_72h": float(m5.quote.max() / total72) if total72 > 0 else None,
        "burst_5m_bins_buy_pressure_ge_0_60": int(len(positive_pressure_bins)),
        "burst_max_5m_buy_pressure": float(m5.pressure.max()) if len(m5) else None,
        "burst_max_large_buy_usdt_5m": float(large_buy_5m.max()) if len(large_buy_5m) else 0.0,
        "burst_large_buy_active_5m_bins": int(len(large_buy_5m)),
    }


def _read_kline_archive(session, symbol, interval, period, cache_dir):
    """Read one Binance Vision monthly or daily 1h-kline archive."""
    if interval == "monthly":
        suffix = period.strftime("%Y-%m")
    else:
        suffix = period.strftime("%Y-%m-%d")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}-1h-{suffix}.zip"
    if cache_path.exists():
        payload = cache_path.read_bytes()
    else:
        url = (
            f"https://data.binance.vision/data/spot/{interval}/klines/"
            f"{symbol}/1h/{symbol}-1h-{suffix}.zip"
        )
        response = session.get(url, timeout=60)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.content
        cache_path.write_bytes(payload)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(csv_name) as handle:
            frame = pd.read_csv(handle, header=None, names=cols)
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["open_time", "open", "high", "low", "close"])


def fetch_future_klines(session, symbol, anchor_ms, cache_dir, pause=0.1):
    """Load future 1h candles from Binance Vision archives, not api.binance.com.

    GitHub-hosted runners can receive HTTP 451 from the Binance REST API.
    Monthly public archives are preferred; daily archives are used when a
    monthly archive is not published.
    """
    start_ms = int(anchor_ms - HOUR_MS)
    end_ms = int(anchor_ms + 72 * HOUR_MS)
    start = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")
    frames = []
    missing_months = []

    for month in months:
        frame = _read_kline_archive(
            session, symbol, "monthly", month.start_time, cache_dir
        )
        if frame is None:
            missing_months.append(month)
        else:
            frames.append(frame)
        if pause:
            time.sleep(pause)

    # Daily archive fallback only for months whose monthly file is unavailable.
    if missing_months:
        days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D", tz="UTC")
        needed = [day for day in days if day.to_period("M") in missing_months]
        for day in needed:
            frame = _read_kline_archive(session, symbol, "daily", day, cache_dir)
            if frame is not None:
                frames.append(frame)
            if pause:
                time.sleep(pause)

    if not frames:
        raise RuntimeError(f"No Binance Vision 1h kline archives found for {symbol}")
    result = pd.concat(frames, ignore_index=True)
    result = (
        result.drop_duplicates("open_time", keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    result = result[(result.open_time >= start_ms) & (result.open_time <= end_ms)]
    if result.empty:
        raise RuntimeError(f"Binance Vision archives contained no requested candles for {symbol}")
    return result

def label_future(klines, anchor_ms):
    if klines.empty:
        return {"label": "incomplete", "label_reason": "no_future_klines"}
    prior = klines[klines.open_time <= anchor_ms]
    future = klines[(klines.open_time >= anchor_ms) & (klines.open_time < anchor_ms + 72 * HOUR_MS)]
    if prior.empty or len(future) < 48:
        return {"label": "incomplete", "label_reason": "insufficient_hourly_history"}
    base = float(prior.iloc[-1].close)
    if not np.isfinite(base) or base <= 0:
        return {"label": "incomplete", "label_reason": "invalid_anchor_price"}

    out = {}
    for hours in (24, 48, 72):
        w = future[future.open_time < anchor_ms + hours * HOUR_MS]
        if w.empty:
            out[f"future_max_high_return_{hours}h_pct"] = None
            out[f"future_close_return_{hours}h_pct"] = None
            continue
        out[f"future_max_high_return_{hours}h_pct"] = round((float(w.high.max()) / base - 1) * 100, 5)
        out[f"future_close_return_{hours}h_pct"] = round((float(w.iloc[-1].close) / base - 1) * 100, 5)

    w24 = future[future.open_time < anchor_ms + 24 * HOUR_MS]
    peak_idx = w24.high.idxmax() if not w24.empty else None
    if peak_idx is not None:
        peak = float(w24.loc[peak_idx, "high"])
        after = future[(future.open_time > float(w24.loc[peak_idx, "open_time"])) &
                       (future.open_time <= float(w24.loc[peak_idx, "open_time"]) + 12 * HOUR_MS)]
        drawdown = (float(after.low.min()) / peak - 1) * 100 if not after.empty else None
    else:
        peak, drawdown = None, None
    out["future_24h_peak_price"] = peak
    out["future_24h_peak_to_next12h_drawdown_pct"] = round(drawdown, 5) if drawdown is not None else None

    high24 = out["future_max_high_return_24h_pct"]
    high48 = out["future_max_high_return_48h_pct"]
    high72 = out["future_max_high_return_72h_pct"]
    close24 = out["future_close_return_24h_pct"]
    close72 = out["future_close_return_72h_pct"]
    pump = (
        high24 is not None and high24 >= 10
        and drawdown is not None and drawdown <= -8
        and close24 is not None and close24 < 5
    )
    sustained = (
        high72 is not None and high72 >= 10
        and close72 is not None and close72 >= 8
    )
    if pump:
        label, reason = "pump_like_reversal", "24h high >=10%, 12h post-peak drawdown <=-8%, 24h close <5%"
    elif sustained:
        label, reason = "sustained_breakout", "72h high >=10% and 72h close >=8%"
    elif high72 is not None and high72 < 10:
        label, reason = "no_10pct_breakout", "72h future high remained below +10%"
    else:
        label, reason = "failed_or_ambiguous_breakout", "10% high occurred without sustained-close or strict reversal criteria"
    out["label"] = label
    out["label_reason"] = reason
    out["target_10pct_24h"] = bool(high24 is not None and high24 >= 10)
    out["target_10pct_48h"] = bool(high48 is not None and high48 >= 10)
    out["target_10pct_72h"] = bool(high72 is not None and high72 >= 10)
    return out


def run(input_dir, out_dir, cases_per_tier=20, pause=0.1):
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cases = build_cases(input_dir, cases_per_tier=cases_per_tier)
    if base_cases.empty:
        raise RuntimeError("No matched historical event/control cases were found")
    rows, errors = [], []
    session = requests.Session()
    archive_cache = out_dir / "aggtrade_archive_cache"
    kline_cache = out_dir / "future_kline_cache"

    for case in base_cases.to_dict("records"):
        for lead in LEAD_HOURS:
            anchor = int(case["anchor_ms"]) - lead * HOUR_MS
            row = {
                **case,
                "source_anchor_ms": int(case["anchor_ms"]),
                "source_anchor_time_utc": case["anchor_time_utc"],
                "lead_hours_before_source_trigger": lead,
                "anchor_ms": anchor,
                "anchor_time_utc": pd.to_datetime(anchor, unit="ms", utc=True).isoformat(),
                "case_id": f'{case["case_id"]}_lead{lead}h',
                "pair_id": f'{case["pair_id"]}_lead{lead}h',
            }
            try:
                trades, statuses, expected = load_window_trades(
                    session, case["symbol"], anchor, archive_cache, pause
                )
                feats = extract_features(trades, anchor)
                feats.update(add_burst_features(trades, anchor))
                klines = fetch_future_klines(session, case["symbol"], anchor, kline_cache, pause)
                labels = label_future(klines, anchor)
                missing = sum(s == "missing_archive" for s in statuses)
                parse_errors = sum(str(s).startswith("parse_error") for s in statuses)
                row.update(feats)
                row.update(labels)
                row.update({
                    "archive_days_expected": expected,
                    "archive_days_missing": missing,
                    "archive_parse_errors": parse_errors,
                    "features_complete": bool(not trades.empty and missing == 0 and parse_errors == 0),
                    "row_complete": bool(not trades.empty and missing == 0 and parse_errors == 0
                                         and labels.get("label") != "incomplete"),
                })
            except Exception as exc:
                row.update({
                    "label": "incomplete",
                    "label_reason": f"{type(exc).__name__}: {exc}",
                    "features_complete": False,
                    "row_complete": False,
                })
                errors.append({"case_id": row["case_id"], "error": row["label_reason"]})
            rows.append(row)
            if len(rows) % 10 == 0:
                print(json.dumps({"prebreakout_progress": f"{len(rows)}/{len(base_cases) * len(LEAD_HOURS)}",
                                  "errors": len(errors)}))

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "true_prebreakout_dataset.csv", index=False)
    complete = frame[frame.row_complete == True]
    summary = {
        "design": "Each event and matched control is sampled 24h, 12h, and 6h before its source anchor. Features use only preceding 72h aggTrades; labels use subsequent 1h klines.",
        "cases_per_tier_requested": int(cases_per_tier),
        "base_manifest_rows": int(len(base_cases)),
        "dataset_rows": int(len(frame)),
        "complete_rows": int(len(complete)),
        "complete_event_rows": int((complete.role == "event").sum()) if not complete.empty else 0,
        "complete_control_rows": int((complete.role == "control").sum()) if not complete.empty else 0,
        "lead_hours": list(LEAD_HOURS),
        "labels": {str(k): int(v) for k, v in complete.label.value_counts().items()},
        "labels_by_lead": {
            str(lead): {str(k): int(v) for k, v in g.label.value_counts().items()}
            for lead, g in complete.groupby("lead_hours_before_source_trigger")
        } if not complete.empty else {},
        "features": [
            "preceding 72h/24h/6h/1h trade flow, buy/sell pressure, large-trade metrics",
            "last 1m and 5m buy pressure and quote volume",
            "last 1m volume versus prior 60-minute median",
            "top 1m/5m volume concentration",
            "count of high-buy-pressure 5m bins",
            "maximum 5m buy pressure and large-buy burst concentration",
        ],
        "label_definitions": {
            "pump_like_reversal": "24h future high >= +10%, 12h after-peak low <= -8% from peak, and 24h close < +5%.",
            "sustained_breakout": "72h future high >= +10% and 72h close >= +8%.",
            "no_10pct_breakout": "72h future high remains below +10%.",
            "failed_or_ambiguous_breakout": "10% high occurred but neither strict pump-like reversal nor sustained-breakout criteria were met.",
        },
        "caveats": [
            "A pump-like label is a price/volume pattern, not proof of coordinated buying or market manipulation.",
            "Public aggTrades cannot identify traders, accounts, beneficial owners, or off-exchange coordination.",
            "The event catalogue is selected from the existing historical surge scan and has survivorship/discovery bias.",
            "The 1h labels have up to one-hour timing granularity; the anchor price is the last hourly close at or before the anchor.",
            "This dataset is exploratory; chronological grouped validation is required before model training or scanner integration.",
        ],
        "errors": errors,
    }
    (out_dir / "true_prebreakout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("input-artifact/surge_scan"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/true_prebreakout"))
    parser.add_argument("--cases-per-tier", type=int, default=20)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    args = parser.parse_args()
    run(args.input_dir, args.out_dir, args.cases_per_tier, args.pause_seconds)


if __name__ == "__main__":
    main()
