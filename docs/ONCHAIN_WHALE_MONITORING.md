# On-chain whale monitoring — Phase 1

This module is a research data layer, not a trading signal. It does not infer that every large transfer is a buy or sell.

## Initial chain scope
- Ethereum (EVM)
- BNB Smart Chain (EVM)
- Solana

Base and Arbitrum can be added after validating provider coverage and mapping tokens to canonical contract/mint addresses.

## Data model
- `onchain_transfers`: immutable, deduplicated transfer events keyed by chain + transaction hash + event index. Stores both block time and first observed time to prevent look-ahead leakage.
- `onchain_wallet_labels`: provenance-aware labels. A wallet is never called an exchange, bridge, treasury, or whale without a source and confidence.
- `onchain_whale_features`: rolling token-level flow aggregates, separate from raw events.

## Critical interpretation rules
1. A wallet-to-wallet transfer is not automatically a buy or sell.
2. Exchange deposits and withdrawals are separate features; do not interpret them as deterministic price direction.
3. Token contract/mint address is the canonical identity. Symbols alone are not unique and must never be used to join assets.
4. USD value must have a recorded valuation source/time in the provider payload. Missing valuation remains NULL, never zero.
5. `block_time_utc` is chain event time; `observed_at_utc` is when our collector first saw it. A feature is usable at a prediction timestamp only when it was observed by then.
6. Deduplicate replays using the primary key and upsert. Keep provider and raw payload for audit.
7. Do not train on live snapshots as if they were historical unless they were archived at the time.

## Provider setup and current collector
- Bitquery V2 GraphQL is the first provider for Ethereum, BNB Chain and Solana. Its transfer datasets expose token movements and provider-valued USD amounts where available. Docs: https://docs.bitquery.io/docs/blockchain/Ethereum/transfers/erc20-token-transfer-api/ and https://docs.bitquery.io/docs/blockchain/Solana/solana-transfers/
- GitHub secret required: `BITQUERY_ACCESS_TOKEN`. Never commit or print this value.
- Collector: `src/onchain_whale_collector.py`; workflow: `.github/workflows/onchain-whale-collector.yml`.
- Initial scope: transfers with provider `AmountInUSD` above USD 100,000, 3-hour overlapping lookback, every 30 minutes. The overlap plus primary-key upsert makes replays safe.
- The collector writes immutable transfer rows only. Wallet labels and directional buy/sell interpretation are intentionally not inferred. Feature aggregation and verified wallet-label enrichment are separate follow-up phases.

## Operational validation
1. Run the workflow manually after configuring the secret.
2. Confirm successful API responses and non-zero/zero counts per chain without assuming zero means no whales (it may mean no provider-valued transfers above threshold).
3. Verify Supabase inserts and duplicate replay behavior; `observed_at_utc` is preserved on duplicate replays.
4. Review provider plan limits and coverage before extending historical backfill or increasing frequency.
5. Build point-in-time flow features only after the raw feed is validated.
