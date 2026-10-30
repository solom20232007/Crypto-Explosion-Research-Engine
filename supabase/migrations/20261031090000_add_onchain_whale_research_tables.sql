create table if not exists public.onchain_transfers (
  chain text not null,
  tx_hash text not null,
  event_index integer not null default 0,
  block_number bigint,
  block_time_utc timestamptz not null,
  observed_at_utc timestamptz not null default now(),
  token_address text not null,
  token_symbol text,
  token_name text,
  from_address text not null,
  to_address text not null,
  amount_raw text,
  token_decimals integer,
  amount_token numeric,
  amount_usd numeric,
  transfer_kind text not null default 'token',
  provider text not null,
  provider_event_id text,
  is_finalized boolean not null default false,
  source_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  primary key (chain, tx_hash, event_index),
  constraint onchain_transfers_chain_nonempty check (length(trim(chain)) > 0),
  constraint onchain_transfers_hash_nonempty check (length(trim(tx_hash)) > 0),
  constraint onchain_transfers_amount_nonnegative check (amount_token is null or amount_token >= 0),
  constraint onchain_transfers_usd_nonnegative check (amount_usd is null or amount_usd >= 0),
  constraint onchain_transfers_decimals_nonnegative check (token_decimals is null or token_decimals >= 0),
  constraint onchain_transfers_kind_check check (transfer_kind in ('native','token','wrapped','bridge','unknown'))
);

create index if not exists onchain_transfers_block_time_idx
  on public.onchain_transfers (block_time_utc desc);
create index if not exists onchain_transfers_token_time_idx
  on public.onchain_transfers (chain, token_address, block_time_utc desc);
create index if not exists onchain_transfers_from_time_idx
  on public.onchain_transfers (chain, from_address, block_time_utc desc);
create index if not exists onchain_transfers_to_time_idx
  on public.onchain_transfers (chain, to_address, block_time_utc desc);
create index if not exists onchain_transfers_usd_time_idx
  on public.onchain_transfers (amount_usd desc, block_time_utc desc)
  where amount_usd is not null;

create table if not exists public.onchain_wallet_labels (
  chain text not null,
  wallet_address text not null,
  entity_type text not null,
  entity_name text,
  confidence numeric not null default 0.5,
  label_source text not null,
  evidence_url text,
  valid_from_utc timestamptz not null default now(),
  valid_to_utc timestamptz,
  observed_at_utc timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  primary key (chain, wallet_address, entity_type, label_source),
  constraint onchain_wallet_labels_confidence_check check (confidence >= 0 and confidence <= 1),
  constraint onchain_wallet_labels_entity_check check (entity_type in ('exchange','bridge','project_treasury','market_maker','fund','whale_candidate','burn','unknown'))
);
create index if not exists onchain_wallet_labels_address_idx
  on public.onchain_wallet_labels (chain, wallet_address);
create index if not exists onchain_wallet_labels_entity_idx
  on public.onchain_wallet_labels (entity_type, entity_name);

create table if not exists public.onchain_whale_features (
  chain text not null,
  token_address text not null,
  token_symbol text,
  window_minutes integer not null,
  window_end_utc timestamptz not null,
  feature_version text not null default 'onchain-whale-v1',
  transfers_count integer not null default 0,
  whale_transfers_count integer not null default 0,
  whale_inflow_usd numeric not null default 0,
  whale_outflow_usd numeric not null default 0,
  net_whale_flow_usd numeric not null default 0,
  exchange_inflow_usd numeric not null default 0,
  exchange_outflow_usd numeric not null default 0,
  unique_whale_senders integer not null default 0,
  unique_whale_receivers integer not null default 0,
  largest_transfer_usd numeric,
  median_transfer_usd numeric,
  provider_coverage jsonb not null default '{}'::jsonb,
  computed_at_utc timestamptz not null default now(),
  primary key (chain, token_address, window_minutes, window_end_utc, feature_version),
  constraint onchain_whale_features_window_check check (window_minutes > 0),
  constraint onchain_whale_features_counts_check check (transfers_count >= 0 and whale_transfers_count >= 0)
);
create index if not exists onchain_whale_features_time_idx
  on public.onchain_whale_features (window_end_utc desc);
create index if not exists onchain_whale_features_token_time_idx
  on public.onchain_whale_features (token_symbol, window_end_utc desc);

alter table public.onchain_transfers enable row level security;
alter table public.onchain_wallet_labels enable row level security;
alter table public.onchain_whale_features enable row level security;

drop policy if exists "service role full access onchain_transfers" on public.onchain_transfers;
create policy "service role full access onchain_transfers"
  on public.onchain_transfers for all to service_role using (true) with check (true);
drop policy if exists "service role full access onchain_wallet_labels" on public.onchain_wallet_labels;
create policy "service role full access onchain_wallet_labels"
  on public.onchain_wallet_labels for all to service_role using (true) with check (true);
drop policy if exists "service role full access onchain_whale_features" on public.onchain_whale_features;
create policy "service role full access onchain_whale_features"
  on public.onchain_whale_features for all to service_role using (true) with check (true);
