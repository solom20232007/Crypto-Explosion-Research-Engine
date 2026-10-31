"""Descriptive lifecycle analysis of historical surge episodes (research only)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_episodes(events: pd.DataFrame, candles: pd.DataFrame, cluster_gap_days: int = 7):
    required_events = {"symbol", "event_date_utc"}
    required_candles = {"symbol", "open_time", "high", "low", "close"}
    if required_events - set(events.columns):
        raise ValueError(f"events missing {sorted(required_events - set(events.columns))}")
    if required_candles - set(candles.columns):
        raise ValueError(f"candles missing {sorted(required_candles - set(candles.columns))}")
    e = events.copy()
    e["event_date"] = pd.to_datetime(e["event_date_utc"], utc=True).dt.floor("D")
    e = e.sort_values(["symbol", "event_date"])
    e["_gap"] = e.groupby("symbol").event_date.diff().dt.days
    e["_episode_no"] = e.groupby("symbol")["_gap"].transform(
        lambda s: s.gt(cluster_gap_days).fillna(True).cumsum()
    )
    grouped = e.groupby(["symbol", "_episode_no"], sort=False)
    episodes = grouped.agg(
        episode_start=("event_date", "min"),
        episode_last_trigger=("event_date", "max"),
        trigger_days=("event_date", "nunique"),
        raw_trigger_rows=("event_date", "size"),
        trigger_types=("trigger_types", lambda s: ",".join(sorted({v for x in s.dropna() for v in str(x).split(",")})) if "trigger_types" in e else ""),
    ).reset_index().drop(columns=["_episode_no"])
    episodes["episode_id"] = episodes["symbol"] + "_" + episodes["episode_start"].dt.strftime("%Y%m%d")

    c = candles.copy()
    c["date"] = pd.to_datetime(c["open_time"], unit="ms", utc=True).dt.floor("D")
    for col in ("high", "low", "close"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["symbol", "date", "high", "low", "close"])
    c = c[(c.high > 0) & (c.low > 0) & (c.close > 0)]
    by_symbol = {s: g.sort_values("date").drop_duplicates("date").set_index("date") for s, g in c.groupby("symbol")}
    records = []
    for ep in episodes.to_dict("records"):
        symbol, start = ep["symbol"], ep["episode_start"]
        bars = by_symbol.get(symbol)
        row = dict(ep)
        if bars is None or start not in bars.index:
            records.append(row)
            continue
        loc = bars.index.get_loc(start)
        if not isinstance(loc, (int, np.integer)):
            loc = int(np.asarray(loc).flat[0])
        prev_close = float(bars.iloc[loc - 1].close) if loc > 0 else np.nan
        start_close = float(bars.iloc[loc].close)
        forward = bars.iloc[loc:loc + 45]
        for horizon in (1, 3, 7, 14, 30):
            w = forward.iloc[:horizon]
            complete = len(w) == horizon
            row[f"during_{horizon}d_max_high_pct"] = round((float(w.high.max()) / prev_close - 1) * 100, 4) if complete and prev_close > 0 else np.nan
            row[f"during_{horizon}d_close_return_pct"] = round((float(w.iloc[-1].close) / prev_close - 1) * 100, 4) if complete and prev_close > 0 else np.nan
        if len(forward) >= 14:
            peak_pos = int(np.argmax(forward.high.to_numpy()[:14]))
            peak = forward.iloc[peak_pos]
            peak_high = float(peak.high)
            row["peak_day_offset"] = peak_pos
            row["peak_high_from_pre_event_close_pct"] = round((peak_high / prev_close - 1) * 100, 4) if prev_close > 0 else np.nan
            row["start_close_to_peak_high_pct"] = round((peak_high / start_close - 1) * 100, 4) if start_close > 0 else np.nan
            after_peak = forward.iloc[peak_pos + 1:]
            for horizon in (7, 14, 30):
                post = after_peak.iloc[:horizon]
                complete = len(post) == horizon
                row[f"after_peak_{horizon}d_max_drawdown_pct"] = round((float(post.low.min()) / peak_high - 1) * 100, 4) if complete and peak_high > 0 else np.nan
                row[f"after_peak_{horizon}d_close_return_pct"] = round((float(post.iloc[-1].close) / peak_high - 1) * 100, 4) if complete and peak_high > 0 else np.nan
        else:
            row["peak_day_offset"] = np.nan
            row["peak_high_from_pre_event_close_pct"] = np.nan
            row["start_close_to_peak_high_pct"] = np.nan
            for horizon in (7, 14, 30):
                row[f"after_peak_{horizon}d_max_drawdown_pct"] = np.nan
                row[f"after_peak_{horizon}d_close_return_pct"] = np.nan
        records.append(row)
    result = pd.DataFrame(records)
    return result.sort_values(["episode_start", "symbol"]).reset_index(drop=True)


def summarize_lifecycle(episodes: pd.DataFrame):
    metric_cols = [c for c in episodes.columns if any(k in c for k in (
        "max_high_pct", "close_return_pct", "max_drawdown_pct", "peak_high_from", "start_close_to_peak"
    ))]
    stats = {}
    for col in metric_cols:
        values = pd.to_numeric(episodes[col], errors="coerce").dropna()
        if len(values):
            stats[col] = {
                "count": int(len(values)),
                "median_pct": round(float(values.median()), 4),
                "p25_pct": round(float(values.quantile(.25)), 4),
                "p75_pct": round(float(values.quantile(.75)), 4),
                "p90_pct": round(float(values.quantile(.90)), 4),
            }
    return {
        "episode_count": int(len(episodes)),
        "symbols": int(episodes.symbol.nunique()) if len(episodes) else 0,
        "cluster_gap_days": 7,
        "episode_definition": "Per symbol, trigger days separated by at most 7 calendar days are grouped; episode start is the first trigger day.",
        "during_definition": "Fixed forward windows from first trigger day; max high and final close are measured against the prior day's close.",
        "after_peak_definition": "Peak is the highest daily high in the first 14 days from episode start; post-peak windows begin on the following day.",
        "metrics": stats,
        "limitations": [
            "Descriptive lifecycle analysis, not a predictive model or trading return simulation.",
            "Episodes are identified from threshold-trigger days, so episode start is the first detected surge day, not necessarily the earliest pre-breakout point.",
            "Metrics require complete forward windows; incomplete windows are left blank.",
            "Universe contains currently active Binance Spot USDT symbols only; delisted-symbol survivorship bias remains.",
        ],
    }


def run(input_dir: Path, out_dir: Path):
    events = pd.read_csv(input_dir / "surge_events.csv")
    candles = pd.read_csv(input_dir / "daily_candles.csv")
    episodes = build_episodes(events, candles)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out_dir / "surge_episodes_lifecycle.csv", index=False)
    report = summarize_lifecycle(episodes)
    (out_dir / "surge_lifecycle_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("reports/surge_scan"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/surge_scan"))
    args = p.parse_args()
    run(args.input_dir, args.out_dir)


if __name__ == "__main__":
    main()
