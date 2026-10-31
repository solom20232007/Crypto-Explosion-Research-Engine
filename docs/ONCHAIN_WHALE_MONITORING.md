# On-chain whale monitoring — Phase 1

This module is a research data layer, not a trading signal. It does not infer that every large transfer is a buy or sell.

## Initial chain scope

- Ethereum (EVM)
- BNB Smart Chain (EVM)
- Solana

## Data model

- `onchain_transfers`: immutable, deduplicated transfer events keyed by chain + transaction hash + event index. Stores both block time and first observed time to prevent look-ahead leakage.
- `onchain_wallet_labels`: provenance-aware labels. A wallet is never called an exchange, bridge, treasury, or whale without a source and confidence.
- `onchain_whale_features`: rolling token-level flow aggregates, separate from raw events.

## Provider coverage and limits

### Routescan — active alternative for global EVM ERC-20 transfers

- Collector: `src/routescan_whale_collector.py`.
- Endpoint: Routescan High-speed API `/v2/network/mainnet/evm/{chainId}/erc20-transfers`.
- Unlike address-history endpoints, the token-level endpoint supports chain-wide ERC-20 transfer retrieval without a wallet address.
- Current configured chains: Ethereum (1) and BNB Smart Chain (56).
- Free keyless tier: 2 requests/second and 10,000 calls/day. A free registered key raises this to 5 requests/second and 100,000 calls/day. A key is optional; use `ROUTESCAN_API_KEY` if configured.
- This feed is ERC-20 only. It does not collect native coin transfers, Solana transfers, or every possible EVM chain in this initial configuration.
- The collector has no USD threshold, stores the raw payload, paginates, deduplicates, retries transient errors, and fails rather than silently truncating at the page cap.
- Official docs: https://docs.routescan.io/api/tokens and https://docs.routescan.io/plans-and-limits/rate-limits

### Bitquery — global EVM and Solana transfer feed (temporarily paused)

- Collector: `src/onchain_whale_collector.py`; workflow: `.github/workflows/onchain-whale-collector.yml`.
- Secret: `BITQUERY_ACCESS_TOKEN`.
- The all-token workflow is manual-only while the current plan quota is exhausted. Do not re-enable it until the user confirms the quota has reset and the query budget is reviewed.
- Bitquery remains the intended broad Solana/global transfer source; its current free quota is exhausted.

### Ankr — wallet-address history, not global chain-wide transfers

- Method: `ankr_getTokenTransfers`.
- Secret: `ANKR_API_TOKEN`.
- Requires one or more wallet addresses. It is not a chain-wide all-token feed.
- Advanced API access and account plan eligibility must be verified in the Ankr dashboard before enabling.
- Official docs: https://www.ankr.com/docs/advanced-api/token-methods/ and https://www.ankr.com/docs/rpc-service/service-plans/

### Alchemy — wallet/contract-filtered EVM transfers

- Method: `alchemy_getAssetTransfers`.
- Secret: `ALCHEMY_API_KEY`.
- Requires a target address, token contract, or bounded block range. It is useful for enriching tracked wallets/tokens, not a substitute for an unbounded chain-wide feed.
- Free plan currently advertises 30M compute units/month. Supported Transfers API networks vary; confirm in Alchemy docs before adding a chain.
- Official docs: https://www.alchemy.com/docs/data/transfers-api/transfers-endpoints/alchemy-get-asset-transfers and https://www.alchemy.com/pricing

### Helius — Solana wallet-address history

- Method: Enhanced Transactions / parsed transaction APIs.
- Secret: `HELIUS_API_KEY`.
- Address-scoped history is suitable for enriching tracked Solana wallets; it does not provide a complete global all-wallet token transfer feed on the free tier.
- Free plan currently advertises 1M credits/month and 10 requests/second.
- Official docs: https://www.helius.dev/docs and https://www.helius.dev/pricing

## Operating rules

1. Bitquery remains paused until the user confirms reactivation.
2. Routescan is the first free alternative to validate for chain-wide EVM ERC-20 transfers. Its separate workflow runs tests before collection.
3. Ankr, Alchemy, Routescan address history, and Helius are address/contract-scoped enrichment sources. Do not label their results as global coverage.
4. Optional providers remain disabled until the corresponding API secret is configured and a smoke test confirms access.
5. All providers write to the same raw transfer table using chain + tx hash + event index as the event identity. USD values are optional metadata only; never filter transfers by USD value.
6. Token contract/mint address is the canonical identity. Symbols alone are not unique.
7. `block_time_utc` is chain event time; `observed_at_utc` is when our collector first saw it. A feature is usable at a prediction timestamp only when it was observed by then.
8. Keep provider payloads for audit. Do not infer buy/sell direction from transfers alone.
9. Do not train on live snapshots as if they were historical unless they were archived at the time.

## Validation sequence

1. Run the Routescan workflow manually and confirm successful API responses.
2. Verify row counts by chain, latest block time, and duplicate event keys in Supabase.
3. Compare a sample of Routescan events against an independent explorer.
4. Measure pages/calls consumed per run before enabling a recurring schedule.
5. Add the wallet-scoped providers after tracked wallet candidates have been curated from observed data and their secrets are configured.
6. Build point-in-time flow features only after raw feed coverage and valuation quality are validated.
