create table if not exists public.onchain_collection_checkpoints (
  provider text not null,
  chain text not null,
  cursor_kind text not null check (cursor_kind in ('unix_seconds', 'block_number')),
  cursor_value bigint not null,
  initial_cursor_value bigint not null,
  last_completed_from bigint,
  last_completed_to bigint,
  status text not null default 'ready' check (status in ('ready', 'running', 'paused', 'complete', 'error')),
  last_error text,
  updated_at timestamptz not null default now(),
  primary key (provider, chain),
  constraint onchain_checkpoint_cursor_nonnegative check (cursor_value >= 0),
  constraint onchain_checkpoint_initial_nonnegative check (initial_cursor_value >= 0)
);
alter table public.onchain_collection_checkpoints enable row level security;
drop policy if exists "service role full access onchain_collection_checkpoints" on public.onchain_collection_checkpoints;
create policy "service role full access onchain_collection_checkpoints"
  on public.onchain_collection_checkpoints for all to service_role using (true) with check (true);
grant all on public.onchain_collection_checkpoints to service_role;
create or replace function public.advance_onchain_collection_checkpoint(
  p_provider text, p_chain text, p_cursor_kind text, p_cursor_value bigint,
  p_initial_cursor_value bigint default null, p_last_completed_from bigint default null,
  p_last_completed_to bigint default null, p_status text default 'ready',
  p_last_error text default null
)
returns public.onchain_collection_checkpoints
language plpgsql security invoker set search_path = ''
as $$
declare saved public.onchain_collection_checkpoints;
begin
  insert into public.onchain_collection_checkpoints (
    provider, chain, cursor_kind, cursor_value, initial_cursor_value,
    last_completed_from, last_completed_to, status, last_error, updated_at
  ) values (
    p_provider, p_chain, p_cursor_kind, p_cursor_value,
    coalesce(p_initial_cursor_value, p_cursor_value),
    p_last_completed_from, p_last_completed_to, p_status, p_last_error, now()
  )
  on conflict (provider, chain) do update set
    cursor_kind = excluded.cursor_kind,
    cursor_value = greatest(onchain_collection_checkpoints.cursor_value, excluded.cursor_value),
    initial_cursor_value = onchain_collection_checkpoints.initial_cursor_value,
    last_completed_from = case when excluded.cursor_value >= onchain_collection_checkpoints.cursor_value
      then excluded.last_completed_from else onchain_collection_checkpoints.last_completed_from end,
    last_completed_to = case when excluded.cursor_value >= onchain_collection_checkpoints.cursor_value
      then excluded.last_completed_to else onchain_collection_checkpoints.last_completed_to end,
    status = excluded.status, last_error = excluded.last_error, updated_at = now()
  returning * into saved;
  return saved;
end;
$$;
revoke all on function public.advance_onchain_collection_checkpoint(text,text,text,bigint,bigint,bigint,bigint,text,text) from public, anon, authenticated;
grant execute on function public.advance_onchain_collection_checkpoint(text,text,text,bigint,bigint,bigint,bigint,text,text) to service_role;
