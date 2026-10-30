# Historical Binance Spot data collection

## What this step does

The collector downloads Binance Spot 1-hour OHLCV klines to one CSV per symbol under `data/raw/`. It writes a manifest after each symbol and persists each downloaded page, so a rerun can continue after the latest saved candle. Existing files are not overwritten from scratch.

## Quick smoke test

Run a small request first to confirm networking and file output:

```bash
python collect_history.py --start 2026-10-20T00:00:00Z --end 2026-10-28T00:00:00Z --max-symbols 3
python audit_history.py
```

## Three-year collection

```bash
python collect_history.py --start 2023-10-28T00:00:00Z --end 2026-10-28T00:00:00Z
python audit_history.py
```

The default universe is discovered from Binance's current `exchangeInfo` and contains active USDT Spot pairs only. This is convenient for a first broad collection, but it has survivorship bias: it does not automatically include all pairs that were listed and later delisted. To collect additional known historical symbols, provide a newline-separated list (comments beginning with `#` are ignored):

```bash
python collect_history.py --symbols-file data/historical_symbols.txt --start 2023-10-28T00:00:00Z --end 2026-10-28T00:00:00Z
```

Only symbols whose historical candles are available from Binance will return data. Do not claim a complete three-year universe until the manifest, coverage report, and a point-in-time listing/delisting inventory have been reviewed.

## Operational notes

- The full universe can take a long time and make many API requests. Start with the smoke test, then run the full collection on a stable machine or a controlled scheduled job. Do not run a large collection on every GitHub push.
- HTTP 429/418 and server errors are retried with backoff. Errors are recorded in `data/raw/collection_manifest.json`; a non-zero exit code is returned if any symbols fail.
- The collector writes market candles only; it does not collect historical news, does not generate features, does not train the advisor, and does not alter V7.
- The coverage audit measures candle continuity, but cannot establish historical listing eligibility by itself.
- Check applicable Binance API terms and local data-storage limits before large-scale collection.
