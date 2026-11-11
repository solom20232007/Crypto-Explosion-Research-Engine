# Comprehensive Pre-Breakout Research Roadmap

Scope: research branch only. Do not merge to `main` without explicit approval. The independent V7 system is out of scope and must not be modified.

## Research principle

Build a time-aligned, auditable feature table where every feature has a source timestamp and an availability timestamp. At prediction time, a feature is usable only if it was publicly available by that timestamp. Never use future candles, revised news, later wallet labels, or post-event data as predictors.

## Signal groups

1. **Market structure and technical analysis (in progress)**
   - Returns and candle ranges over 1/3/7/14 days.
   - RSI, EMA alignment, MACD, ATR, Bollinger width/position, prior-high proximity.
   - Breakout/consolidation features, candle body/wicks, volatility expansion.
   - Hourly features for 4h/12h/24h lead times once hourly coverage is audited.

2. **Volume and liquidity (partially available)**
   - Quote volume versus rolling median/mean, volume z-score, trade-count ratio, taker-buy quote share when supplied by candles.
   - Add live/historical order-book spread, depth within 0.5%/1%, depth imbalance, estimated slippage for fixed notional, and liquidity persistence.
   - Treat candle volume as a proxy, not actual executable liquidity. Flag thin books and suspicious wicks.

3. **News and catalysts (planned)**
   - Ingest timestamped headlines and official announcements from exchange/project feeds and reputable news APIs.
   - Store original URL, publisher, published_at, first_seen_at, symbol/project mapping, language, duplicate cluster, event category, and sentiment/novelty scores.
   - Separate official listings/delistings, token unlocks, partnerships, exploits, regulatory events, and general commentary.
   - Do not rely on unlicensed scraping or assume a news item was available before its publication/first-seen time.

4. **Whale wallets and on-chain flows (planned, chain-specific)**
   - Identify chain/token contract first; track large transfers, exchange inflows/outflows, concentration, top-holder changes, and known exchange/bridge/treasury wallets.
   - Record provider, chain, contract address, block time, observed time, entity-label confidence, and USD valuation source.
   - Do not label a wallet as a whale or exchange without evidence. Transfers alone do not prove buying/selling; exchange deposits are contextual, not deterministic.
   - Use public data or explicitly configured provider credentials; never commit API keys.

5. **Token supply and event calendar (planned)**
   - Circulating/total/max supply, market cap and FDV, unlock/vesting schedule, emissions, burns, mint authority, contract changes, exchange listing status.
   - Keep snapshots with source and timestamp; these values change and providers may disagree.

6. **Market regime and cross-asset context (planned)**
   - BTC/ETH market regime only as context variables, not as candidate coins; broad market returns/volatility, BTC dominance if sourced, sector-relative strength, and correlation.
   - Candidate universe remains small USDT altcoins; BTC, ETH, SOL are excluded as candidates per project requirements.

7. **Quality, manipulation, and risk controls**
   - Minimum dollar turnover and order-book depth; spread/slippage limits; abnormal wick/price-gap flags; missing-data and stale-feed flags; suspicious volume spikes.
   - Track listing age, delisting/survivorship bias, token migrations, redenominations, and contract swaps.

## Evaluation protocol

- Targets are future-only and defined separately by horizon; keep 4h/12h/24h labels when complete hourly candles exist and daily 1/3/7-day labels for the current historical scan.
- Split by time, not random rows. Add purging/embargo around overlapping target windows and deduplicate consecutive trigger days into episodes.
- Compare against prevalence and simple rules; report precision, recall, average precision, ROC-AUC, false alerts per symbol/day, lead time, event capture, and calibration.
- Evaluate top-k alerts and simulated entry after signal availability, including fees, slippage, spread, and adverse selection. Report results by liquidity bucket and market regime.
- Keep a final untouched out-of-time test set; never tune thresholds on that test set.
- Track data coverage and feature availability per split. Missing news/on-chain coverage must not silently become a neutral signal.

## Implementation sequence

1. Validate historical candle continuity, symbol lifecycle, labels, and the current daily technical/liquidity feature pipeline.
2. Produce baseline/out-of-time evaluation with episode-level metrics and realistic execution costs.
3. Add order-book/liquidity snapshots and hourly indicators.
4. Add timestamp-safe news ingestion.
5. Add chain-specific whale/on-chain sources after choosing supported chains and data providers.
6. Add tokenomics/unlock calendar and market-regime context.
7. Combine features only after each source is independently validated; run ablation tests to measure incremental value.

## Current limitations

- Current Binance universe is based on active pairs, so delisted historical coins may be missing.
- Candle-derived liquidity proxies are not order-book depth.
- News, whale, tokenomics, and live order-book integrations are not yet implemented in this phase.
- Research outputs are not a guarantee of future performance or trading recommendations.
