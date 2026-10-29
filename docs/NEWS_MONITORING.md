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


## Official Binance announcements WebSocket

Binance documents an official English announcements topic, `com_announcement_en`, delivered over `wss://api.binance.com/sapi/wss`. The documented stream requires an API key in the `X-MBX-APIKEY` header and an HMAC-signed connection URL. This collector is optional and bounded by a user-selected duration; it does not place orders or request trading permissions.

Set repository Actions secrets `BINANCE_API_KEY` and `BINANCE_API_SECRET` (API key permissions should be limited to what Binance requires for this public announcements service; never enable withdrawals or trading). In the manual workflow, set `binance_ws_seconds` to a short collection window such as 60–300 seconds. Leave it at 0 to skip the WebSocket connection.

The adapter stores Binance's `publishDate` as `published_utc`, but uses the local receipt time as `first_seen_utc` and `available_at_utc` for leakage-safe research. The documented payload contains catalog ID/name, title, body and publish timestamp, but no article URL; therefore the URL is left blank rather than fabricated. The official API only pushes English announcements. This live stream is not a historical archive and must not be used to backfill historical first-seen times.

Reference: Binance's [official WebSocket API overview](https://developers.binance.com/en/docs/products/announcements/general-info) and [announcement payload schema](https://developers.binance.com/en/docs/products/announcements/announcement).
