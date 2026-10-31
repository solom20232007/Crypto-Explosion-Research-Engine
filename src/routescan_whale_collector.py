"""Routescan global ERC-20 transfer collector.

Unlike wallet-history APIs, Routescan's token-level ERC-20 transfers endpoint
can return transfers for a chain without specifying a wallet address. This
collector is limited to Routescan-indexed EVM chains and ERC-20 events; it is
not a replacement for Solana-wide indexing or native transfers.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

BASE_URL = "https://api.routescan.io/v2/network/mainnet/evm"
PROVIDER = "routescan_v2"
PAGE_SIZE = 100
DEFAULT_MAX_TOTAL_REQUESTS_PER_CHAIN = 300
DEFAULT_CHAINS = {"ethereum": 1, "bsc": 56}


def _utc(value: Any) -> str:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def normalize_routescan_transfer(item: dict[str, Any], chain: str) -> dict[str, Any] | None:
    tx_hash = item.get("txHash")
    sender, receiver = item.get("from"), item.get("to")
    token_address = item.get("tokenAddress")
    if not tx_hash or not sender or not receiver or not token_address:
        return None
    decimals = item.get("tokenDecimals")
    raw_amount = str(item.get("amount")) if item.get("amount") is not None else None
    amount_token = None
    if raw_amount is not None and decimals is not None:
        try:
            amount_token = Decimal(raw_amount) / (Decimal(10) ** int(decimals))
        except (InvalidOperation, TypeError, ValueError):
            amount_token = None
    log_index = item.get("logIndex", 0)
    return {
        "chain": chain,
        "tx_hash": str(tx_hash),
        "event_index": int(log_index),
        "block_number": int(item["blockNumber"]) if item.get("blockNumber") is not None else None,
        "block_time_utc": _utc(item.get("timestamp") or item.get("createdAt")),
        "token_address": str(token_address),
        "token_symbol": item.get("tokenSymbol"),
        "token_name": item.get("tokenName"),
        "from_address": str(sender),
        "to_address": str(receiver),
        "amount_raw": raw_amount,
        "token_decimals": int(decimals) if decimals is not None else None,
        "amount_token": amount_token,
        "amount_usd": _decimal(item.get("amountInUsd")),
        "transfer_kind": "token",
        "provider": PROVIDER,
        "provider_event_id": str(log_index),
        "is_finalized": True,
        "source_payload": item,
    }


class RoutescanClient:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None,
                 sleep=time.sleep):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.sleep = sleep

    def _get_page(self, chain_id: int, params: dict[str, Any]) -> dict[str, Any]:
        query = dict(params)
        if self.api_key:
            query["apikey"] = self.api_key
        url = f"{BASE_URL}/{chain_id}/erc20-transfers"
        for attempt in range(6):
            try:
                response = self.session.get(url, params=query, headers=(
                    {"apikey": self.api_key} if self.api_key else {}
                ), timeout=(10, 45))
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == 5:
                    raise
                self.sleep(min(30, 2 ** attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 5:
                    response.raise_for_status()
                retry = response.headers.get("Retry-After")
                try:
                    delay = max(1, min(60, int(retry))) if retry else min(30, 2 ** attempt)
                except (TypeError, ValueError):
                    delay = min(30, 2 ** attempt)
                self.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(f"Routescan API error: {payload['error']}")
            return payload
        raise RuntimeError("Routescan request failed after retries")

    @staticmethod
    def _cursor_from_link(link: dict[str, Any]) -> str | None:
        direct = link.get("nextToken")
        if direct:
            return str(direct)
        next_link = link.get("next")
        if not next_link:
            return None
        parsed = urlparse(str(next_link))
        query = parse_qs(parsed.query)
        values = query.get("next")
        if values:
            return values[-1]
        # Some API versions return the opaque cursor directly.
        return str(next_link) if "://" not in str(next_link) and "?" not in str(next_link) else None

    def _fetch_interval(self, chain: str, chain_id: int, start: datetime, end: datetime,
                        max_pages: int, budget: list[int], max_total_requests: int,
                        depth: int = 0, on_interval=None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timestampFrom": start.isoformat().replace("+00:00", "Z"),
            "timestampTo": end.isoformat().replace("+00:00", "Z"),
            "sort": "desc",
            "limit": PAGE_SIZE,
        }
        rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        cursor = None
        capped = True
        for _page in range(max_pages):
            if budget[0] >= max_total_requests:
                raise RuntimeError(
                    f"Routescan {chain}: safety cap of {max_total_requests} requests per chain reached; "
                    "collection stopped; previously completed intervals were already uploaded."
                )
            budget[0] += 1
            if cursor:
                params["next"] = cursor
            payload = self._get_page(chain_id, params)
            items = payload.get("items") or []
            for item in items:
                row = normalize_routescan_transfer(item, chain)
                if row:
                    key = (row["chain"], row["tx_hash"], row["event_index"])
                    rows[key] = row
            link = payload.get("link") or {}
            next_cursor = self._cursor_from_link(link)
            if not next_cursor or not items:
                capped = False
                break
            if next_cursor == cursor:
                raise RuntimeError(f"Routescan {chain}: pagination cursor did not advance")
            cursor = next_cursor
            # Free keyless plan is 2 requests/second; registered free is 5.
            self.sleep(0.55 if not self.api_key else 0.22)

        if not capped:
            completed_rows = list(rows.values())
            # Persist each fully paginated leaf interval immediately. If a later
            # interval hits the request budget, completed intervals remain safe.
            if on_interval and completed_rows:
                on_interval(completed_rows)
            return completed_rows

        duration = (end - start).total_seconds()
        if duration <= 1 or depth >= 20:
            raise RuntimeError(
                f"Routescan {chain}: interval {start.isoformat()}..{end.isoformat()} still "
                f"exceeds {max_pages * PAGE_SIZE} rows at minimum split size; refusing truncation."
            )
        midpoint = start + (end - start) / 2
        overlap = min(1.0, duration / 100.0)
        left = self._fetch_interval(chain, chain_id, start, midpoint + timedelta(seconds=overlap),
                                    max_pages, budget, max_total_requests, depth + 1, on_interval)
        right = self._fetch_interval(chain, chain_id, midpoint - timedelta(seconds=overlap), end,
                                     max_pages, budget, max_total_requests, depth + 1, on_interval)
        for row in left + right:
            rows[(row["chain"], row["tx_hash"], row["event_index"])] = row
        return list(rows.values())

    def fetch_chain(self, chain: str, chain_id: int, hours: int = 2,
                    max_pages: int = 50,
                    max_total_requests: int = DEFAULT_MAX_TOTAL_REQUESTS_PER_CHAIN,
                    on_interval=None) -> list[dict[str, Any]]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        return self._fetch_interval(chain, chain_id, start, end, max_pages, [0], max_total_requests,
                                    on_interval=on_interval)


def upload_transfers(rows: list[dict[str, Any]], session: requests.Session | None = None) -> int:
    if not rows:
        return 0
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SECRET_KEY"]
    http = session or requests.Session()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates,return=minimal",
    }
    endpoint = f"{url}/rest/v1/onchain_transfers"
    for start in range(0, len(rows), 500):
        response = http.post(endpoint, params={"on_conflict": "chain,tx_hash,event_index"},
                             headers=headers, json=rows[start:start + 500], timeout=60)
        if not response.ok:
            raise RuntimeError(
                f"Supabase onchain_transfers upload failed: HTTP {response.status_code}: {response.text[:1000]}"
            )
    return len(rows)


def collect(hours: int = 1, max_pages: int = 50,
            max_total_requests: int = DEFAULT_MAX_TOTAL_REQUESTS_PER_CHAIN) -> dict[str, int]:
    client = RoutescanClient(os.getenv("ROUTESCAN_API_KEY"))
    counts: dict[str, int] = {}
    uploaded_attempted = 0
    for chain, chain_id in DEFAULT_CHAINS.items():

        def persist_completed_interval(rows: list[dict[str, Any]]) -> None:
            nonlocal uploaded_attempted
            observed = datetime.now(timezone.utc).isoformat()
            for row in rows:
                row["observed_at_utc"] = observed
            inserted_attempt = upload_transfers(rows)
            uploaded_attempted += inserted_attempt
            print(json.dumps({"provider": PROVIDER, "chain": chain,
                              "checkpoint_rows_attempted": inserted_attempt,
                              "uploaded_attempted_total": uploaded_attempted}, sort_keys=True), flush=True)

        rows = client.fetch_chain(chain, chain_id, hours=hours, max_pages=max_pages,
                                  max_total_requests=max_total_requests,
                                  on_interval=persist_completed_interval)
        # fetch_chain invokes the callback for every fully paginated interval.
        counts[chain] = len(rows)
    counts["total"] = sum(counts.values())
    counts["uploaded_attempted"] = uploaded_attempted
    result = {"provider": PROVIDER, "lookback_hours": hours, "scope": "global_erc20_transfers",
              "usd_filter": "none", "max_total_requests_per_chain": max_total_requests,
              "upload_mode": "completed_intervals", **counts}
    print(json.dumps(result, sort_keys=True))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect global EVM ERC-20 transfers from Routescan")
    parser.add_argument("--hours", type=int, default=int(os.getenv("ROUTESCAN_LOOKBACK_HOURS", "1")))
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("ROUTESCAN_MAX_PAGES", "50")))
    parser.add_argument("--max-total-requests", type=int, default=int(os.getenv("ROUTESCAN_MAX_TOTAL_REQUESTS", str(DEFAULT_MAX_TOTAL_REQUESTS_PER_CHAIN))))
    args = parser.parse_args()
    if not 1 <= args.hours <= 24 or args.max_pages < 1:
        parser.error("--hours must be 1-24 and --max-pages must be positive")
    if args.max_total_requests < 1:
        parser.error("--max-total-requests must be positive")
    collect(args.hours, args.max_pages, args.max_total_requests)


if __name__ == "__main__":
    main()
