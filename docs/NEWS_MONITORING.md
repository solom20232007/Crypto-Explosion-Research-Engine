# News and Announcements Monitoring

The module ingests RSS/Atom feeds and creates auditable article records plus per-symbol daily features.

## Run

Pass comma-separated RSS/Atom URLs using --feeds. The default candle input is reports/surge_scan/daily_candles.csv. The GitHub Actions workflow is manual and stores output as an artifact.

## Timestamp safety

- published_utc is the publisher-provided publication/update time.
- first_seen_utc is the UTC time this collector fetched the feed.
- available_at_utc is conservatively set to first_seen_utc, not the potentially earlier publication time.
- An item is included only when first_seen_utc is strictly earlier than the candle close/signal timestamp.
- Items without a publication time are retained and flagged.

## Outputs

- reports/news_monitor/news_events.csv
- reports/news_monitor/news_symbol_daily_features.csv
- reports/news_monitor/news_monitor_summary.json

## Caveats

A current RSS snapshot cannot reconstruct when a historical article first became visible. Do not use it as historical news training data unless archived first-seen timestamps exist. Ticker matching may miss project names; optional JSON aliases map base tickers to names. Keyword categories and sentiment hints are weak triage labels, not model-grade NLP. Review feed availability and licensing before production scheduling.

## Curated starter source registry

The default registry is `config/news_feeds.json`. It starts with established independent publishers (CoinDesk, Cointelegraph, The Block, Decrypt, CryptoSlate) plus primary official feeds (Coinbase Blog and Ethereum Foundation Blog). Each record stores source name, class/tier, type, and enabled state. The tier is provenance metadata, not a guarantee that an individual claim is correct.

Run with the registry:

`python -m src.news_monitor --feed-config config/news_feeds.json --candles reports/surge_scan/daily_candles.csv`

The registry can be edited to add project-specific official blogs and chain foundations. Exchange newsroom pages that do not expose RSS/Atom should be integrated through a dedicated official API/page adapter rather than pretending they are RSS feeds. Source metadata is retained in every article row.
