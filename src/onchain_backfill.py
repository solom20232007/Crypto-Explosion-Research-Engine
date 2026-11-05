"""Checkpointed, bounded historical backfill for Ethereum and BSC token transfers."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from src.bsc_rpc_whale_collector import BscRpcClient, CHAIN as BSC_CHAIN, PROVIDER as BSC_PROVIDER
from src.onchain_checkpoint import SupabaseCheckpointStore
from src.routescan_whale_collector import (
    DEFAULT_CHAINS,
    PROVIDER as ROUTESCAN_PROVIDER,
    RoutescanClient,
    upload_transfers,
)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _epoch(value: str) -> int:
    return int(_parse_utc(value).timestamp())


def _checkpoint_cursor(checkpoint: dict[str, Any], expected_kind: str) -> int:
    if checkpoint.get("cursor_kind") != expected_kind:
        raise RuntimeError(
            f"Checkpoint cursor kind mismatch: expected {expected_kind}, "
            f"found {checkpoint.get('cursor_kind')}"
        )
    return int(checkpoint["cursor_value"])


def _find_first_block_at_or_after(
    client: BscRpcClient, target_timestamp: int, latest_number: int, low_hint: int = 0
) -> int:
    """Return the first block whose timestamp is >= target, or latest+1."""
    low = max(0, low_hint)
    high = latest_number
    while low < high:
        middle = (low + high) // 2
        block = client._block(middle)
        if client._hex_int(block["timestamp"]) < target_timestamp:
            low = middle + 1
        else:
            high = middle
    if low > latest_number:
        return latest_number + 1
    block = client._block(low)
    if client._hex_int(block["timestamp"]) < target_timestamp:
        return latest_number + 1
    return low


def run_routescan_backfill(
    start_utc: str,
    end_utc: str | None = None,
    chunk_minutes: int = 30,
    max_chunks: int = 12,
    max_pages: int = 50,
    max_total_requests: int = 1000,
    store: Any | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    if chunk_minutes < 1 or max_chunks < 1 or max_pages < 1 or max_total_requests < 1:
        raise ValueError("chunk_minutes, max_chunks, max_pages and max_total_requests must be positive")

    store = store or SupabaseCheckpointStore()
    provider, chain = ROUTESCAN_PROVIDER, "ethereum"
    start_value = _epoch(start_utc)
    end_value = _epoch(end_utc) if end_utc else int((datetime.now(timezone.utc) - timedelta(seconds=120)).timestamp())
    if end_value < start_value:
        raise ValueError("end_utc must be later than start_utc")

    checkpoint = store.get(provider, chain)
    if checkpoint is None:
        checkpoint = store.advance(
            provider=provider, chain=chain, cursor_kind="unix_seconds",
            cursor_value=start_value, initial_cursor_value=start_value, status="ready",
        )
    cursor = _checkpoint_cursor(checkpoint, "unix_seconds")
    initial = int(checkpoint.get("initial_cursor_value", start_value))
    last_completed_from = checkpoint.get("last_completed_from")
    last_completed_to = checkpoint.get("last_completed_to")
    if client is None:
        api_key = __import__("os").getenv("ROUTESCAN_API_KEY")
        client = RoutescanClient(api_key)

    budget = [0]
    chunks_done = 0
    attempted_rows = 0
    try:
        while cursor < end_value and chunks_done < max_chunks:
            chunk_end = min(cursor + chunk_minutes * 60, end_value)
            rows = client._fetch_interval(
                chain, DEFAULT_CHAINS[chain],
                datetime.fromtimestamp(cursor, timezone.utc),
                datetime.fromtimestamp(chunk_end, timezone.utc),
                max_pages, budget, max_total_requests,
            )
            observed = datetime.now(timezone.utc).isoformat()
            for row in rows:
                row["observed_at_utc"] = observed
            attempted = upload_transfers(rows) if rows else 0
            attempted_rows += attempted

            next_status = "complete" if chunk_end >= end_value else "ready"
            completed_from = cursor
            store.advance(
                provider=provider, chain=chain, cursor_kind="unix_seconds",
                cursor_value=chunk_end, initial_cursor_value=initial,
                completed_from=completed_from, completed_to=chunk_end,
                status=next_status, last_error=None,
            )
            last_completed_from, last_completed_to = completed_from, chunk_end
            cursor = chunk_end
            chunks_done += 1
            print(json.dumps({
                "provider": provider, "chain": chain,
                "completed_from_utc": datetime.fromtimestamp(completed_from, timezone.utc).isoformat(),
                "completed_to_utc": datetime.fromtimestamp(cursor, timezone.utc).isoformat(),
                "checkpoint_cursor": cursor, "chunk_rows_attempted": attempted,
                "uploaded_attempted_total": attempted_rows, "requests_used": budget[0],
            }, sort_keys=True), flush=True)
    except Exception as exc:
        store.advance(
            provider=provider, chain=chain, cursor_kind="unix_seconds",
            cursor_value=cursor, initial_cursor_value=initial,
            completed_from=last_completed_from, completed_to=last_completed_to,
            status="paused", last_error=str(exc)[:2000],
        )
        raise

    final_status = "complete" if cursor >= end_value else "paused"
    if cursor >= end_value:
        store.advance(
            provider=provider, chain=chain, cursor_kind="unix_seconds",
            cursor_value=cursor, initial_cursor_value=initial,
            completed_from=last_completed_from, completed_to=last_completed_to,
            status="complete", last_error=None,
        )
    result = {
        "provider": provider, "chain": chain, "status": final_status,
        "cursor_value": cursor, "target_end": end_value,
        "chunks_completed_this_run": chunks_done,
        "uploaded_attempted": attempted_rows, "requests_used": budget[0],
        "max_total_requests": max_total_requests,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run_bsc_backfill(
    start_utc: str,
    end_utc: str | None = None,
    chunk_blocks: int = 5000,
    max_chunks: int = 12,
    max_total_requests: int = 1000,
    store: Any | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    if chunk_blocks < 1 or max_chunks < 1 or max_total_requests < 1:
        raise ValueError("chunk_blocks, max_chunks and max_total_requests must be positive")

    store = store or SupabaseCheckpointStore()
    if client is None:
        import os
        client = BscRpcClient(
            rpc_url=os.getenv("BSC_RPC_URL") or None,
            max_total_requests=max_total_requests,
        )

    provider, chain = BSC_PROVIDER, BSC_CHAIN
    start_value = _epoch(start_utc)
    requested_end = _epoch(end_utc) if end_utc else None
    latest_number = client._hex_int(client._rpc("eth_blockNumber", []))
    latest_block = client._block(latest_number)
    latest_timestamp = client._hex_int(latest_block["timestamp"])
    safe_end = min(
        requested_end if requested_end is not None else latest_timestamp - 60,
        latest_timestamp - 60,
    )
    if safe_end < start_value:
        raise ValueError("BSC safe end time is earlier than start_utc")

    checkpoint = store.get(provider, chain)
    if checkpoint is None:
        first_block = _find_first_block_at_or_after(client, start_value, latest_number)
        checkpoint = store.advance(
            provider=provider, chain=chain, cursor_kind="block_number",
            cursor_value=first_block, initial_cursor_value=first_block, status="ready",
        )
    cursor = _checkpoint_cursor(checkpoint, "block_number")
    initial = int(checkpoint.get("initial_cursor_value", cursor))
    last_completed_from = checkpoint.get("last_completed_from")
    last_completed_to = checkpoint.get("last_completed_to")
    target_end_exclusive = _find_first_block_at_or_after(
        client, safe_end, latest_number, low_hint=min(cursor, latest_number)
    )
    chunks_done = 0
    attempted_rows = 0
    try:
        while cursor < target_end_exclusive and chunks_done < max_chunks:
            chunk_end_exclusive = min(cursor + chunk_blocks, target_end_exclusive)
            rows = client._fetch_block_range(cursor, chunk_end_exclusive - 1)
            observed = datetime.now(timezone.utc).isoformat()
            for row in rows:
                row["observed_at_utc"] = observed
            attempted = upload_transfers(rows) if rows else 0
            attempted_rows += attempted

            next_status = "complete" if chunk_end_exclusive >= target_end_exclusive else "ready"
            store.advance(
                provider=provider, chain=chain, cursor_kind="block_number",
                cursor_value=chunk_end_exclusive, initial_cursor_value=initial,
                completed_from=cursor, completed_to=chunk_end_exclusive - 1,
                status=next_status, last_error=None,
            )
            last_completed_from, last_completed_to = cursor, chunk_end_exclusive - 1
            print(json.dumps({
                "provider": provider, "chain": chain,
                "completed_from_block": cursor,
                "completed_to_block": chunk_end_exclusive - 1,
                "checkpoint_next_block": chunk_end_exclusive,
                "chunk_rows_attempted": attempted,
                "uploaded_attempted_total": attempted_rows,
                "rpc_requests_used": client.request_count,
            }, sort_keys=True), flush=True)
            cursor = chunk_end_exclusive
            chunks_done += 1
    except Exception as exc:
        store.advance(
            provider=provider, chain=chain, cursor_kind="block_number",
            cursor_value=cursor, initial_cursor_value=initial,
            completed_from=last_completed_from, completed_to=last_completed_to,
            status="paused", last_error=str(exc)[:2000],
        )
        raise

    final_status = "complete" if cursor >= target_end_exclusive else "paused"
    if cursor >= target_end_exclusive:
        store.advance(
            provider=provider, chain=chain, cursor_kind="block_number",
            cursor_value=cursor, initial_cursor_value=initial,
            completed_from=last_completed_from, completed_to=last_completed_to,
            status="complete", last_error=None,
        )
    result = {
        "provider": provider, "chain": chain, "status": final_status,
        "next_block": cursor, "target_end_exclusive_block": target_end_exclusive,
        "chunks_completed_this_run": chunks_done,
        "uploaded_attempted": attempted_rows,
        "rpc_requests_used": client.request_count,
        "max_total_requests": max_total_requests,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume historical on-chain transfer backfill from Supabase checkpoints")
    parser.add_argument("--provider", choices=("both", "routescan", "bsc"), default="both")
    parser.add_argument("--start-utc", required=True, help="UTC start time, e.g. 2026-11-04T00:00:00Z; used only when a checkpoint is first created")
    parser.add_argument("--end-utc", help="Optional fixed UTC end time; otherwise use a safe recent cutoff")
    parser.add_argument("--chunk-minutes", type=int, default=30)
    parser.add_argument("--chunk-blocks", type=int, default=5000)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--max-total-requests", type=int, default=1000)
    args = parser.parse_args()

    failures = []
    if args.provider in {"both", "routescan"}:
        try:
            run_routescan_backfill(
                args.start_utc, args.end_utc, args.chunk_minutes, args.max_chunks,
                args.max_pages, args.max_total_requests,
            )
        except Exception as exc:
            failures.append({"provider": "routescan", "error": str(exc)})
            print(json.dumps(failures[-1], sort_keys=True), flush=True)
    if args.provider in {"both", "bsc"}:
        try:
            run_bsc_backfill(
                args.start_utc, args.end_utc, args.chunk_blocks, args.max_chunks,
                args.max_total_requests,
            )
        except Exception as exc:
            failures.append({"provider": "bsc", "error": str(exc)})
            print(json.dumps(failures[-1], sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
