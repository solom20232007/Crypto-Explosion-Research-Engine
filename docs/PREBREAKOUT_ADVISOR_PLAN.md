# Binance Pre-Breakout Advisor — implementation plan

## Goal
Research which Binance Spot USDT pairs are likely to become the most profitable opportunities over the next 4, 12, and 24 hours, before the move happens. Do not require a fixed absolute return such as 15%, 20%, or 100%: rank candidates relative to the other eligible symbols at the same timestamp. Evaluate realized entry-to-exit returns, fees/slippage, and downside risk as well as future highs. The output is an evidence-backed review of indicator/rule changes for V7. It must never silently change V7 or production configuration.

## First baseline implemented in this branch
- Uses a relative ranking target: labels the top 10% of eligible symbols at each timestamp by maximum future high return, with no fixed percentage threshold.
- The future-high target is only a discovery label, not proof of a tradable profit. The next evaluation step must simulate entry after the signal, realistic exits, fees, slippage, and adverse excursion.
- Uses only features calculated at the signal timestamp.
- Uses chronological train/validation/test splits with a horizon embargo.
- Reports precision among the highest-ranked candidates, lift over the base rate, average precision, and validation permutation importance.
- Writes reports/prebreakout_advisor.json.
- Requires at least 10 symbols; it intentionally refuses to treat BTC/ETH/SOL-only data as a top-gainer universe.

## Data requirements
1. Build a historical symbol universe, not just today's active pairs. Record listing/delisting dates when available and avoid survivorship bias.
2. Collect 1h OHLCV candles for all eligible USDT Spot pairs across the three-year period; document missing periods and symbols with insufficient history.
3. Keep point-in-time snapshots of symbol eligibility and liquidity. A current universe must not be projected backward as if every pair existed then.
4. Add historical news/announcements only when timestamped sources are available. Features must be aligned to when the information became public, not the later article crawl time.
5. Add taker-buy/sell, spread/order-book, market-wide and BTC/ETH context where historical data supports it.

## Evaluation rules
- Use rolling/forward-chaining tests after the baseline, and never tune against the final test set.
- Report top-5/top-10 precision, lift, average precision, forward returns after fees/slippage, adverse excursion, and risk-adjusted results. Rank by expected opportunity rather than assuming a single return threshold fits every market regime.
- Compare with simple baselines (past momentum, volume spike, and random/top-volume selection).
- Check symbol-level and time-level leakage, duplicate candles, delisted coins, and sparse labels.
- News features are a separate experiment; they must show incremental out-of-time value over price/volume-only models.

## V7 recommendation contract
Every recommendation should include: current rule/indicator, proposed candidate value or range, reason, validation evidence, out-of-time test evidence, expected alert-volume change, risks, and whether manual approval is required. A feature-importance ranking alone is not permission to change a threshold. No automatic deployment.

## Current known limitation
The existing historical dataset was previously observed to contain only BTCUSDT, ETHUSDT, and SOLUSDT for about one year. The baseline will stop with a clear message until a wider historical universe is collected. The three-year goal is not considered complete until coverage is verified in the generated data report.
