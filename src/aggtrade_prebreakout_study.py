"""Matched pre-breakout trade-flow study using Binance public daily aggTrades archives.

No wallet identity is inferred: buyer/seller fields represent aggressive taker direction.
"""
import argparse
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id",
        "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]


def _utc_ms(value):
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _event_anchor(row):
    names = ("confirmed_15pct_24h", "confirmed_25pct_48h", "confirmed_40pct_72h")
    candidates = [row.get(name + "_signal_time_utc") for name in names]
    candidates = [x for x in candidates if pd.notna(x) and str(x).strip()]
    if candidates:
        return min(candidates, key=_utc_ms), "confirmed"
    early = row.get("early_10pct_24h_signal_time_utc")
    if pd.notna(early) and str(early).strip():
        return str(early), "early_only"
    return None, None


def build_cases(input_dir, cases_per_tier=25, seed=20261101):
    episodes = pd.read_csv(input_dir / "surge_episodes_lifecycle.csv")
    refined = pd.read_csv(input_dir / "hourly_episode_refinement.csv")
    daily = pd.read_csv(input_dir / "daily_candles.csv")
    for frame in (episodes, refined):
        frame["symbol"] = frame.symbol.astype(str)
    daily["symbol"] = daily.symbol.astype(str)
    daily["open_time"] = pd.to_numeric(daily.open_time, errors="coerce").astype("Int64")
    for col in ("close", "high", "quote_volume", "volume"):
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    merged = episodes.merge(refined, on=["symbol", "episode_id"], how="inner",
                            suffixes=("_episode", "_hourly"))
    candidates = []
    for row in merged.to_dict("records"):
        anchor, tier = _event_anchor(row)
        if not anchor:
            continue
        candidates.append({
            "symbol": row["symbol"], "episode_id": row["episode_id"],
            "episode_start": row.get("episode_start"),
            "anchor_time_utc": pd.Timestamp(anchor).isoformat(),
            "anchor_ms": _utc_ms(anchor), "case_tier": tier,
        })
    rng = np.random.default_rng(seed)
    daily_by_symbol = {s: g.sort_values("open_time").reset_index(drop=True)
                       for s, g in daily.groupby("symbol")}
    episode_dates = {}
    for c in candidates:
        episode_dates.setdefault(c["symbol"], []).append(_utc_ms(c["episode_start"]))
    # Search candidate controls at the same UTC hour, in a nearby period, with
    # no daily surge trigger nearby and no +10% daily high over the next 3 days.
    pairs, used_controls = [], set()
    for tier in ("confirmed", "early_only"):
        pool = [x for x in candidates if x["case_tier"] == tier]
        if pool:
            order = rng.permutation(len(pool))
            pool = [pool[i] for i in order]
        selected = 0
        for event in pool:
            if selected >= cases_per_tier:
                break
            g = daily_by_symbol.get(event["symbol"])
            if g is None or len(g) < 5:
                continue
            event_day = (event["anchor_ms"] // DAY_MS) * DAY_MS
            event_rows = g[g.open_time == event_day]
            if event_rows.empty:
                event_rows = g.iloc[(g.open_time.astype("int64") - event_day).abs().argsort()[:1]]
            event_row = event_rows.iloc[0]
            event_qv = float(event_row.quote_volume) if pd.notna(event_row.quote_volume) else 0
            hour_offset = (event["anchor_ms"] - event_day) // HOUR_MS
            eligible = []
            times = g.open_time.astype("int64").to_numpy()
            highs = g.high.to_numpy()
            closes = g.close.to_numpy()
            qvs = g.quote_volume.to_numpy()
            for i in range(3, len(g) - 3):
                day_ms = int(times[i])
                control_ms = day_ms + int(hour_offset) * HOUR_MS
                key = (event["symbol"], control_ms)
                if key in used_controls or abs(control_ms - event["anchor_ms"]) < 7 * DAY_MS:
                    continue
                if abs(day_ms - event_day) > 180 * DAY_MS:
                    continue
                if any(abs(control_ms - d) < 8 * DAY_MS for d in episode_dates.get(event["symbol"], [])):
                    continue
                prev_close = closes[i - 1]
                future_high = np.nanmax(highs[i:i + 4])
                if not np.isfinite(prev_close) or prev_close <= 0 or not np.isfinite(future_high):
                    continue
                if (future_high / prev_close - 1) >= 0.10:
                    continue
                if event_qv > 0 and np.isfinite(qvs[i]) and not (0.2 <= qvs[i] / event_qv <= 5.0):
                    continue
                # Require a full 72h candle-history footprint before control time.
                if day_ms - 3 * DAY_MS < int(times[0]):
                    continue
                volume_penalty = abs(math.log(max(qvs[i], 1) / max(event_qv, 1))) if event_qv > 0 and np.isfinite(qvs[i]) else 2.0
                date_penalty = abs(control_ms - event["anchor_ms"]) / DAY_MS
                eligible.append((volume_penalty + 0.002 * date_penalty, control_ms))
            if not eligible:
                continue
            eligible.sort(key=lambda x: x[0])
            control_ms = eligible[0][1]
            pair_id = f"{event['symbol']}_{event['anchor_ms']}"
            pairs.append({**event, "pair_id": pair_id, "case_id": pair_id + "_event",
                          "role": "event"})
            pairs.append({"symbol": event["symbol"], "episode_id": None,
                          "episode_start": None,
                          "anchor_time_utc": pd.to_datetime(control_ms, unit="ms", utc=True).isoformat(),
                          "anchor_ms": control_ms, "case_tier": "control",
                          "pair_id": pair_id, "case_id": pair_id + "_control",
                          "role": "control"})
            used_controls.add((event["symbol"], control_ms))
            selected += 1
    return pd.DataFrame(pairs)


def _download_day(session, symbol, day, cache_dir, pause=0.1):
    cache_dir.mkdir(parents=True, exist_ok=True)
    date = pd.to_datetime(day, unit="ms", utc=True).strftime("%Y-%m-%d")
    path = cache_dir / f"{symbol}-aggTrades-{date}.zip"
    if path.exists():
        return path.read_bytes(), "cache"
    url = f"{BASE}/{symbol}/{symbol}-aggTrades-{date}.zip"
    last = None
    for attempt in range(4):
        try:
            response = session.get(url, timeout=60)
            if response.status_code == 404:
                return None, "missing_archive"
            if response.status_code in (418, 429, 500, 502, 503, 504):
                time.sleep(max(float(response.headers.get("Retry-After", 2 ** attempt)), 1))
                continue
            response.raise_for_status()
            path.write_bytes(response.content)
            if pause:
                time.sleep(pause)
            return response.content, "downloaded"
        except requests.RequestException as exc:
            last = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise last or RuntimeError("aggTrades archive download failed")


def load_window_trades(session, symbol, anchor_ms, cache_dir, pause=0.1):
    start_ms = anchor_ms - 72 * HOUR_MS
    first_day = (start_ms // DAY_MS) * DAY_MS
    last_day = ((anchor_ms - 1) // DAY_MS) * DAY_MS
    day_starts = list(range(first_day, last_day + 1, DAY_MS))
    chunks, statuses = [], []
    for day in day_starts:
        raw, status = _download_day(session, symbol, day, cache_dir, pause)
        statuses.append(status)
        if raw is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
                with archive.open(csv_name) as handle:
                    part = pd.read_csv(handle, header=None, names=COLS)
            part["timestamp"] = pd.to_numeric(part.timestamp, errors="coerce")
            part["price"] = pd.to_numeric(part.price, errors="coerce")
            part["quantity"] = pd.to_numeric(part.quantity, errors="coerce")
            part["is_buyer_maker"] = part.is_buyer_maker.astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False})
            part = part.dropna(subset=["timestamp", "price", "quantity", "is_buyer_maker"])
            part = part[(part.timestamp >= start_ms) & (part.timestamp < anchor_ms)]
            chunks.append(part)
        except (zipfile.BadZipFile, StopIteration, ValueError, pd.errors.ParserError) as exc:
            statuses.append("parse_error:" + type(exc).__name__)
    if chunks:
        trades = pd.concat(chunks, ignore_index=True)
        trades = trades.drop_duplicates("agg_trade_id").sort_values("timestamp")
        trades["quote"] = trades.price * trades.quantity
    else:
        trades = pd.DataFrame(columns=COLS + ["quote"])
    return trades, statuses, len(day_starts)


def extract_features(trades, anchor_ms, large_threshold=None):
    if trades.empty:
        return {"trade_rows_72h": 0, "large_trade_threshold_usdt": None}
    t = trades.sort_values("timestamp").copy()
    if large_threshold is None:
        large_threshold = float(t.quote.quantile(0.99))
    t["aggressive_buy"] = ~t.is_buyer_maker.astype(bool)
    t["large"] = t.quote >= large_threshold
    result = {"trade_rows_72h": int(len(t)),
              "large_trade_threshold_usdt": round(large_threshold, 4)}
    for hours in (72, 24, 6, 1):
        w = t[t.timestamp >= anchor_ms - hours * HOUR_MS]
        prefix = f"h{hours}"
        buy = float(w.loc[w.aggressive_buy, "quote"].sum())
        sell = float(w.loc[~w.aggressive_buy, "quote"].sum())
        total = buy + sell
        lb = w[w.large & w.aggressive_buy]
        ls = w[w.large & ~w.aggressive_buy]
        result.update({
            prefix + "_trade_count": int(len(w)),
            prefix + "_quote_volume_usdt": round(total, 4),
            prefix + "_aggressive_buy_usdt": round(buy, 4),
            prefix + "_aggressive_sell_usdt": round(sell, 4),
            prefix + "_buy_ratio": round(buy / total, 5) if total else None,
            prefix + "_buy_pressure": round((buy - sell) / total, 5) if total else None,
            prefix + "_median_trade_usdt": round(float(w.quote.median()), 4) if len(w) else None,
            prefix + "_p95_trade_usdt": round(float(w.quote.quantile(.95)), 4) if len(w) else None,
            prefix + "_large_buy_count": int(len(lb)),
            prefix + "_large_sell_count": int(len(ls)),
            prefix + "_large_buy_usdt": round(float(lb.quote.sum()), 4),
            prefix + "_large_sell_usdt": round(float(ls.quote.sum()), 4),
            prefix + "_large_buy_active_hours": int(pd.to_datetime(lb.timestamp, unit="ms", utc=True).dt.floor("h").nunique()),
            prefix + "_large_sell_active_hours": int(pd.to_datetime(ls.timestamp, unit="ms", utc=True).dt.floor("h").nunique()),
        })
        if len(w) > 1:
            result[prefix + "_trade_price_change_pct"] = round(
                (float(w.iloc[-1].price) / float(w.iloc[0].price) - 1) * 100, 5)
        else:
            result[prefix + "_trade_price_change_pct"] = None
    # Acceleration compares the latest 6h with the preceding 6h.
    recent = t[t.timestamp >= anchor_ms - 6 * HOUR_MS].quote.sum()
    prior = t[(t.timestamp >= anchor_ms - 12 * HOUR_MS) &
              (t.timestamp < anchor_ms - 6 * HOUR_MS)].quote.sum()
    result["volume_acceleration_6h_vs_prior6h"] = round(float(recent / prior), 5) if prior > 0 else None
    return result


def run(input_dir, out_dir, cases_per_tier=25, pause=0.1):
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(input_dir, cases_per_tier=cases_per_tier)
    cases.to_csv(out_dir / "aggtrade_case_manifest.csv", index=False)
    cache = out_dir / "archive_cache"
    session = requests.Session()
    outputs, errors, archive_counts = [], [], {"downloaded": 0, "cache": 0, "missing_archive": 0}
    for idx, case in enumerate(cases.to_dict("records"), 1):
        try:
            trades, statuses, expected_days = load_window_trades(
                session, case["symbol"], int(case["anchor_ms"]), cache, pause)
            for status in statuses:
                if status in archive_counts:
                    archive_counts[status] += 1
            features = extract_features(trades, int(case["anchor_ms"]))
            missing = sum(s == "missing_archive" for s in statuses)
            parse_errors = sum(str(s).startswith("parse_error") for s in statuses)
            outputs.append({**case, **features,
                            "archive_days_expected": expected_days,
                            "archive_days_missing": missing,
                            "archive_parse_errors": parse_errors,
                            "data_complete": missing == 0 and parse_errors == 0 and not trades.empty,
                            "status": "ok" if missing == 0 and parse_errors == 0 and not trades.empty else "incomplete"})
        except Exception as exc:
            errors.append({"case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"})
            outputs.append({**case, "data_complete": False, "status": "fetch_error"})
        if idx % 10 == 0:
            print(json.dumps({"aggtrade_progress": f"{idx}/{len(cases)}",
                              "errors": len(errors), "archive_counts": archive_counts}))
    frame = pd.DataFrame(outputs)
    frame.to_csv(out_dir / "aggtrade_prebreakout_features.csv", index=False)
    complete = frame[frame.data_complete == True] if not frame.empty else frame
    feature_cols = [c for c in complete.columns if c.startswith(("h72_", "h24_", "h6_", "h1_")) or c == "volume_acceleration_6h_vs_prior6h"]
    group_summary = {}
    if not complete.empty:
        for label, group in complete.groupby("case_tier"):
            group_summary[label] = {"cases": int(len(group)),
                "features": {c: {"median": round(float(group[c].median()), 5) if group[c].notna().any() else None,
                                 "mean": round(float(group[c].mean()), 5) if group[c].notna().any() else None}
                             for c in feature_cols}}
    pair_rows = []
    if not complete.empty:
        indexed = complete.set_index(["pair_id", "role"])
        for pair_id in complete.pair_id.unique():
            if (pair_id, "event") not in indexed.index or (pair_id, "control") not in indexed.index:
                continue
            e, c = indexed.loc[(pair_id, "event")], indexed.loc[(pair_id, "control")]
            row = {"pair_id": pair_id, "event_tier": e.case_tier}
            for col in feature_cols:
                if pd.notna(e.get(col)) and pd.notna(c.get(col)):
                    row[col + "_event_minus_control"] = float(e[col]) - float(c[col])
            pair_rows.append(row)
    pd.DataFrame(pair_rows).to_csv(out_dir / "aggtrade_matched_pair_deltas.csv", index=False)
    summary = {
        "study_design": "Matched event/control pilot; each event has a same-symbol control timestamp chosen within 180 days, similar daily quote volume, at least 8 days from any detected episode, and no +10% daily-high move in the following 3 days.",
        "sample_target_per_tier": int(cases_per_tier),
        "manifest_rows": int(len(cases)),
        "event_cases": int((cases.role == "event").sum()) if not cases.empty else 0,
        "matched_controls": int((cases.role == "control").sum()) if not cases.empty else 0,
        "cases_with_complete_aggtrades": int(len(complete)),
        "complete_event_cases": int((complete.role == "event").sum()) if not complete.empty else 0,
        "complete_controls": int((complete.role == "control").sum()) if not complete.empty else 0,
        "matched_complete_pairs": int(len(pair_rows)),
        "archive_counts": archive_counts,
        "fetch_errors": errors,
        "group_summary": group_summary,
        "feature_windows_hours": [72, 24, 6, 1],
        "large_trade_definition": "Top 1% of quote-notional aggTrades within each case's preceding 72-hour window; descriptive size proxy, not wallet/whale identity.",
        "aggressor_definition": "isBuyerMaker=false means aggressive buyer; true means aggressive seller.",
        "limitations": [
            "Pilot sample only; not a full-universe statistical conclusion.",
            "Events originate from the existing daily-trigger catalog and currently listed Binance Spot USDT pairs, retaining survivorship and discovery-selection bias.",
            "Control matching is observational and does not establish causality.",
            "aggTrades does not identify wallets or beneficial owners.",
            "Archive files unavailable for any required UTC day make that case incomplete and exclude it from complete-case group summaries."
        ],
    }
    (out_dir / "aggtrade_prebreakout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("input-artifact/surge_scan"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/aggtrade_study"))
    parser.add_argument("--cases-per-tier", type=int, default=25)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    args = parser.parse_args()
    run(args.input_dir, args.out_dir, args.cases_per_tier, args.pause_seconds)


if __name__ == "__main__":
    main()
