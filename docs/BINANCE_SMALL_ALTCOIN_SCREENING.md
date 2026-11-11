# Binance Small-Altcoin Candidate Screening

## Objective
Find Binance Spot USDT altcoins with enough real trading activity and two-sided liquidity to justify deeper pre-breakout monitoring. BTC, ETH, BNB, SOL and other major assets are excluded from the candidate universe; they remain market-regime controls elsewhere.

## Two-stage screen

1. Universe: active Binance Spot USDT symbols; exclude stablecoins, fiat pairs, leveraged tokens, and a maintained list of major assets.
2. Low-cost 24h ticker gate:
   - quote volume >= 250,000 USDT
   - >= 500 trades
3. Spread gate:
   - spread <= 75 basis points (0.75%)
4. Deep order-book check:
   - inspect up to 60 highest-volume survivors
   - capture 100 levels per side
   - require at least 2,500 USDT bid depth and 2,500 USDT ask depth within 1% of mid-price
5. Save qualified symbols and timestamped book metrics as workflow artifacts for 30 days.

## Important limitations

Binance Spot exchange metadata does not expose market capitalization. This is a liquidity-screened small/mid-altcoin proxy based on excluding major assets, not a verified market-cap ranking. Thresholds are initial research defaults and should be evaluated against subsequent candidate coverage and false-positive rates before tightening.

Order-book snapshots are live observations, not historical depth. They must be archived with timestamps and must never be used as historical pre-breakout features unless the actual observation time is preserved. The hourly candle collector remains the broad historical market dataset; BTC/ETH/SOL remain context controls, not target coins.

## Workflow

GitHub Actions workflow: `.github/workflows/live-orderbook-liquidity.yml`
- Intended cadence: every 15 minutes (GitHub scheduled workflows execute only from the repository default branch).
- Manual dispatch and research-branch marker trigger are available when the workflow is run from the research branch.
- No trade execution or V7 scanner changes are performed.
