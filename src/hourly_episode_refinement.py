"""Refine daily surge episodes to hourly, leakage-safe forward-move timestamps."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASES = ("https://data-api.binance.vision", "https://api.binance.com")
HOUR_MS = 3_600_000


def _get_hourly(symbol, start_ms, end_ms, pause=0.08):
    """Fetch at most a compact event-centered hourly window."""
    last_error = None
    for base in BASES:
        for attempt in range(4):
            try:
                response = requests.get(
                    base + "/api/v3/klines",
                    params={"symbol": symbol, "interval": "1h",
                            "startTime": int(start_ms), "endTime": int(end_ms) - 1,
                            "limit": 200},
                    timeout=30,
                )
                if response.status_code in (418, 429):
                    time.sleep(max(int(response.headers.get("Retry-After", 2 ** attempt)), 1))
                    continue
                response.raise_for_status()
                rows = response.json()
                if not rows:
                    return pd.DataFrame()
                cols = ["open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trade_count",
                        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]
                frame = pd.DataFrame(rows, columns=cols)
                for col in ("open_time", "high", "low", "close", "quote_volume",
                            "taker_buy_quote_volume", "trade_count"):
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
                return frame.dropna(subset=["open_time", "high", "close"]).sort_values("open_time")
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)
        time.sleep(max(0, pause))
    raise last_error or RuntimeError("Binance hourly endpoint unavailable")


def refine_episode(episode, hourly):
    """Find earliest hourly close whose *future* window reaches each target."""
    symbol = str(episode["symbol"])
    day = pd.Timestamp(episode["episode_start"])
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    start = int(day.timestamp() * 1000)
    result = {
        "symbol": symbol,
        "episode_start_utc": day.isoformat(),
        "episode_id": episode.get("episode_id"),
        "episode_tier": episode.get("episode_tier"),
        "daily_trigger_days": episode.get("trigger_days"),
        "hourly_bars_returned": int(len(hourly)),
        "hourly_data_gap_count": 0,
    }
    if hourly.empty:
        result["status"] = "no_hourly_data"
        return result
    h = hourly.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    times = h.open_time.astype("int64").to_numpy()
    gaps = pd.Series(times).diff().dropna()
    result["hourly_data_gap_count"] = int((gaps != HOUR_MS).sum())
    # Signals are formed after a candle closes. Search from 72h before episode
    # start through the first 24h of its UTC trigger day.
    lower, upper = start - 72 * HOUR_MS, start + 24 * HOUR_MS
    search_indices = [i for i, ts in enumerate(times) if lower <= ts < upper]
    targets = [
        ("early_10pct_24h", 24, 10.0),
        ("confirmed_15pct_24h", 24, 15.0),
        ("confirmed_25pct_48h", 48, 25.0),
        ("confirmed_40pct_72h", 72, 40.0),
    ]
    for name, horizon, threshold in targets:
        found = None
        for i in search_indices:
            j = i + horizon
            if j >= len(h):
                continue
            # Exclude a signal if its forward window contains missing hourly bars.
            if times[j] - times[i] != horizon * HOUR_MS:
                continue
            base = float(h.iloc[i].close)
            future_high = float(h.iloc[i + 1:j + 1].high.max())
            if base > 0:
                move = (future_high / base - 1.0) * 100
                if move >= threshold:
                    found = {
                        "signal_time_utc": pd.to_datetime(times[i] + HOUR_MS, unit="ms", utc=True).isoformat(),
                        "signal_close": round(base, 12),
                        "future_max_high_pct": round(move, 4),
                        "hours_to_episode_start": round((start - times[i] - HOUR_MS) / HOUR_MS, 2),
                    }
                    break
        result[name + "_signal_time_utc"] = found["signal_time_utc"] if found else None
        result[name + "_signal_close"] = found["signal_close"] if found else None
        result[name + "_future_max_high_pct"] = found["future_max_high_pct"] if found else None
        result[name + "_lead_hours_to_episode_start"] = found["hours_to_episode_start"] if found else None
    result["status"] = "ok"
    return result


def run(input_dir: Path, out_dir: Path, pause=0.08, max_episodes=None):
    episodes = pd.read_csv(input_dir / "surge_episodes_lifecycle.csv")
    if max_episodes:
        episodes = episodes.head(max_episodes)
    records, failures = [], []
    for n, episode in enumerate(episodes.to_dict("records"), 1):
        day = pd.Timestamp(episode["episode_start"])
        day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
        start = int(day.timestamp() * 1000)
        try:
            # 72h pre-event history plus 96h after UTC episode start, so a signal
            # during the first 24h can still have a complete 72h outcome window.
            hourly = _get_hourly(str(episode["symbol"]), start - 72 * HOUR_MS,
                                 start + 96 * HOUR_MS + 1, pause)
            records.append(refine_episode(episode, hourly))
        except Exception as exc:
            failures.append({"symbol": episode["symbol"], "episode_start": episode["episode_start"],
                             "error": f"{type(exc).__name__}: {exc}"})
            records.append({"symbol": episode["symbol"], "episode_start_utc": str(episode["episode_start"]),
                            "episode_id": episode.get("episode_id"), "status": "fetch_error"})
        if n % 100 == 0:
            print(json.dumps({"hourly_refinement_progress": f"{n}/{len(episodes)}",
                              "failures": len(failures)}))
        if pause:
            time.sleep(pause)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "hourly_episode_refinement.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    summary = {
        "episodes_requested": int(len(episodes)),
        "episodes_ok": int(sum(x.get("status") == "ok" for x in records)),
        "episodes_without_data": int(sum(x.get("status") == "no_hourly_data" for x in records)),
        "episodes_fetch_failed": int(len(failures)),
        "hourly_data_gap_rows": int(sum(x.get("hourly_data_gap_count", 0) for x in records)),
        "label_definition": "At each hourly candle close, use only its close as the baseline and the next 24/48/72 complete hourly highs as the future outcome. Earliest qualifying signal close is retained for +10%/24h, +15%/24h, +25%/48h and +40%/72h.",
        "search_window": "From 72 hours before the daily episode start through the first 24 hours of the episode UTC day; fetch includes 72 future hours.",
        "leakage_rule": "Signal timestamp is after the baseline candle closes; all outcome highs are strictly in subsequent hourly candles. These are forward labels, not live buy signals.",
        "limitations": [
            "Episodes originate from a daily-trigger catalog and currently active Binance Spot USDT symbols; this is not a complete delisted-symbol universe.",
            "Hourly refinement is centered on detected daily episodes, so events that never triggered the daily discovery thresholds are not represented.",
            "Lead hours are measured relative to the UTC daily episode start, not a claim of true market accumulation onset."
        ],
        "failures": failures,
        "output": str(output),
    }
    (out_dir / "hourly_episode_refinement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("reports/surge_scan"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/surge_scan"))
    parser.add_argument("--pause-seconds", type=float, default=0.08)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    run(args.input_dir, args.out_dir, args.pause_seconds, args.max_episodes)


if __name__ == "__main__":
    main()
