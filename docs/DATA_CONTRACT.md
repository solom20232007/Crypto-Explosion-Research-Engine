# Data contract for consumer programs

**These are research indicators, not buy/sell signals.** The project ranks Binance Spot (USDT) small-cap
candidates by how likely a large upward move is in the next 24-72 hours, and publishes the indicators behind the ranking.

## Access
Connect with the read-only role `research_reader` (see "Access setup") and read the `api` schema. Do not read `public.*`.

## Views (schema `api`)
| View | Content | Refresh |
|---|---|---|
| `latest_recommendations` | Top 30 candidates of the latest decision bar | daily ~01:25 UTC |
| `recommendation_history` | All past recommendations + realized outcomes | daily |
| `recommendation_performance` | Realized hit rate of top 5/10/30 vs the universe base rate | daily |
| `market_context_daily` | Breadth, BTC returns, `market_wide` inputs at each 00:00 UTC | hourly |
| `experimental_*` | News / on-chain (Ethereum only) / futures context. Not used by the model | varies |

### `latest_recommendations` columns
- `decision_bar_utc`: the 00:00 UTC hourly candle the ranking is based on (its close is known at 01:00 UTC).
- `rank`, `rank_score`: order by `rank_score` (uncalibrated logistic score, trained on +10%/24h).
- `p_up10_24h`, `p_up25_48h`, `p_up40_72h`: isotonic-calibrated probabilities that the max high in the next 24/48/72h exceeds +10% / +25% / +40% over the bar close.
- `market_wide`: true when BTC 24h return >= 3% or >= 12 research symbols are up >= 10% in 24h. The model has near-chance skill on such days.
- `already_moved`: always false in published rows (symbols that already moved are excluded; outside the training domain).
- Indicators: `ret_24h, ret_72h, rv24` (24h realized volatility of hourly returns), `range_24h`, `vol_ratio_24` (hour volume vs previous 24h mean), `vol_compression` (rv24/rv168), `close_vs_sma20`, `dist_from_high_168`, `dist_from_low_168`, `btc_ret_24h, breadth_24h, pump_count_24h`.

## Honest performance (temporal hold-out, Nov 2025 - Nov 2026 data)
- Ranking AUC about 0.68 (0.72 excluding `market_wide` days). Top-5% of rows have about 3x the base rate.
- Listing the top 5 of ~114 symbols per day gave about 40% hit rate for +10%/24h vs about 10% base (precision on the weighted hold-out).
- Forward (live) results accumulate in `recommendation_performance`; judge them only after 2-3 months.
- Universe is the current Binance Spot list (survivorship bias): delisted coins are absent, so results are optimistic.
- Majors and stable/wrapped assets (BTC, ETH, SOL, EURI, WBETH, WBTC) are excluded as candidates.

## Not guaranteed
No liquidity/slippage model is included. Small-cap spreads can erase a statistical edge; check order-book depth before acting.
