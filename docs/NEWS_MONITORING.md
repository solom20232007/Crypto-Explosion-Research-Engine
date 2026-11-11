# News and Announcements Monitoring

The module ingests RSS/Atom feeds and Binance public announcements, then creates auditable article records plus per-symbol daily features. No Binance API credentials are required for the public-page collector.

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


## Public Binance announcements (no API key)

The default workflow also fetches the latest English announcements from Binance's public announcement CMS. It uses the public page's underlying JSON response, normalizes each article into the same schema as RSS items, and stores the official announcement URL, category, publisher timestamp, and local first-seen/available-at timestamp. No API key, secret, trading permission, or withdrawal permission is needed.

The collector is best-effort: Binance does not document this CMS endpoint as a stable public developer API, so its response shape or availability may change. If the request fails, the workflow records the error and continues with configured RSS/Atom sources. It does not fabricate announcement records or historical first-seen times.

Run with the default 50-item public announcement fetch:

`python -m src.news_monitor --feed-config config/news_feeds.json --candles reports/surge_scan/daily_candles.csv --binance-public-limit 50`

The older authenticated WebSocket adapter remains an optional manual CLI capability only; it is not used by the default GitHub Actions workflow and is not required for the research pipeline. Its documented stream needs credentials, while this public-page collector does not.
