# Scanner signal-time news features

The news monitor can now create `news_symbol_signal_features.csv` from the actual scanner signal timestamps.

## Input

Pass a CSV using `--signals`. It must contain:

- `symbol`: Binance symbol such as `ABCUSDT`
- `signal_time_utc`: timezone-aware UTC timestamp, preferably ISO 8601 such as `2026-10-30T19:42:00Z`

One row represents one symbol at one signal timestamp. Repeated symbols at different timestamps are expected.

## Run

```bash
python -m src.news_monitor \
  --feed-config config/news_feeds.json \
  --candles reports/surge_scan/daily_candles.csv \
  --signals reports/surge_scan/scanner_signals.csv
```

The input path above is an example; provide the real CSV emitted by the scanner. Without `--signals`, the monitor continues to produce the existing daily feature file only.

## Point-in-time rule

News availability uses `available_at_utc`, falling back to `first_seen_utc` for legacy event files. `published_utc` is never used as a substitute for availability. A story is included only when its availability timestamp is strictly earlier than the signal timestamp; equal timestamps are excluded conservatively.

The generated file contains the same 24-hour / 7-day counts, announcement / unlock / security categories, and positive / negative keyword hints as the daily feature file.

## Important limitation

A current RSS or Binance snapshot cannot reconstruct when old stories first became visible historically. Historical backtests must use archived first-seen timestamps, not today's collection time or the publisher's potentially older publication timestamp.
