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

## Provider setup
- EVM: Etherscan API V2 supports a shared API endpoint and chain IDs, but account/transfer endpoints are address-scoped; it does not by itself discover all large transfers globally. See https://docs.etherscan.io/
- Solana: Helius provides address transaction history and transfer-oriented APIs. It is useful after candidate wallets or mints are known; it is not a universal global whale feed. See https://www.helius.dev/docs/getting-data
- A provider with global large-transfer discovery or an explicit monitored-wallet/token universe is required before scheduled ingestion can be enabled.

## First operational milestones
1. Apply the Supabase migration.
2. Configure provider credentials as GitHub Actions secrets (never commit keys).
3. Implement provider-specific backfill/live adapters and token-address registry.
4. Validate coverage, replay idempotency, observed-time integrity, and label provenance.
5. Only then schedule collection and build features for the historical event study.
