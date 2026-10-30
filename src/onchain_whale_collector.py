"""All-token on-chain transfer collector for research.

Collects provider-visible fungible/native transfers without a USD-value filter.
USD estimates are retained as optional metadata only; they never determine
whether an event is collected. This feed does not infer buy/sell direction.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

BITQUERY_URL = "https://streaming.bitquery.io/graphql"
PROVIDER = "bitquery_v2"
LOOKBACK_HOURS = 1
PAGE_SIZE = 1000
MAX_PAGES = 100

EVM_QUERY = """
query AllTokenTransfers($hoursAgo: Int!, $limit: Int!, $offset: Int!) {
  EVM(dataset: realtime, network: NETWORK_PLACEHOLDER) {
    Transfers(
      where: {
        Block: {Time: {since_relative: {hours_ago: $hoursAgo}}}
        Transfer: {Success: true}
      }
      limit: {count: $limit, offset: $offset}
      orderBy: {descending: Block_Time}
    ) {
      Block { Time Number }
      Transaction { Hash Index }
      Transfer {
        Index Amount AmountInUSD Sender Receiver Type
        Currency { Name Symbol SmartContract Decimals Native }
      }
    }
  }
}
"""

SOLANA_QUERY = """
query AllTokenTransfers($hoursAgo: Int!, $limit: Int!, $offset: Int!) {
  Solana(dataset: realtime) {
    Transfers(
      where: {
        Block: {Time: {since_relative: {hours_ago: $hoursAgo}}}
      }
      limit: {count: $limit, offset: $offset}
      orderBy: {descending: Block_Time}
    ) {
      Block { Time Slot }
      Transaction { Signature }
      Transfer {
        Index Amount AmountInUSD Sender { Address } Receiver { Address }
        Currency { Name Symbol MintAddress Decimals Native Fungible }
      }
    }
  }
}
"""


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc(value: Any) -> str:
    if not value:
        raise ValueError("Provider transfer is missing block time")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def normalize_evm_transfer(item: dict[str, Any], chain: str) -> dict[str, Any] | None:
    transfer = item.get("Transfer") or {}
    currency = transfer.get("Currency") or {}
    tx = item.get("Transaction") or {}
    block = item.get("Block") or {}
    tx_hash = tx.get("Hash")
    sender, receiver = transfer.get("Sender"), transfer.get("Receiver")
    if not tx_hash or not sender or not receiver:
        return None
    index = transfer.get("Index")
    if index is None:
        index = tx.get("Index", 0)
    native = currency.get("Native") is True
    address = currency.get("SmartContract") or ("native:eth" if chain == "ethereum" else "native:bnb")
    return {
        "chain": chain,
        "tx_hash": str(tx_hash),
        "event_index": int(index),
        "block_number": int(block["Number"]) if block.get("Number") is not None else None,
        "block_time_utc": _utc(block.get("Time")),
        "token_address": str(address),
        "token_symbol": currency.get("Symbol"),
        "token_name": currency.get("Name"),
        "from_address": str(sender),
        "to_address": str(receiver),
        "amount_raw": None,
        "token_decimals": int(currency["Decimals"]) if currency.get("Decimals") is not None else None,
        "amount_token": _number(transfer.get("Amount")),
        "amount_usd": _number(transfer.get("AmountInUSD")),
        "transfer_kind": "native" if native else "token",
        "provider": PROVIDER,
        "provider_event_id": str(index),
        "is_finalized": True,
        "source_payload": item,
    }


def normalize_solana_transfer(item: dict[str, Any]) -> dict[str, Any] | None:
    transfer = item.get("Transfer") or {}
    currency = transfer.get("Currency") or {}
    tx = item.get("Transaction") or {}
    block = item.get("Block") or {}
    sender = (transfer.get("Sender") or {}).get("Address")
    receiver = (transfer.get("Receiver") or {}).get("Address")
    tx_hash = tx.get("Signature")
    if not tx_hash or not sender or not receiver:
        return None
    index = transfer.get("Index", 0)
    native = currency.get("Native") is True
    address = currency.get("MintAddress") or "native:sol"
    return {
        "chain": "solana",
        "tx_hash": str(tx_hash),
        "event_index": int(index),
        "block_number": int(block["Slot"]) if block.get("Slot") is not None else None,
        "block_time_utc": _utc(block.get("Time")),
        "token_address": str(address),
        "token_symbol": currency.get("Symbol"),
        "token_name": currency.get("Name"),
        "from_address": str(sender),
        "to_address": str(receiver),
        "amount_raw": None,
        "token_decimals": int(currency["Decimals"]) if currency.get("Decimals") is not None else None,
        "amount_token": _number(transfer.get("Amount")),
        "amount_usd": _number(transfer.get("AmountInUSD")),
        "transfer_kind": "native" if native else "token",
        "provider": PROVIDER,
        "provider_event_id": str(index),
        "is_finalized": True,
        "source_payload": item,
    }


class BitqueryClient:
    def __init__(self, token: str, session: requests.Session | None = None):
        if not token:
            raise ValueError("BITQUERY_ACCESS_TOKEN is required")
        self.token = token
        self.session = session or requests.Session()

    def query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        response = None
        for attempt in range(4):
            response = self.session.post(
                BITQUERY_URL,
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
                json={"query": query, "variables": variables},
                timeout=60,
            )
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            if attempt < 3:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = max(10, min(120, int(retry_after))) if retry_after else 30 * (attempt + 1)
                except ValueError:
                    delay = 30 * (attempt + 1)
                time.sleep(delay)
        if response is None or not response.ok:
            status = response.status_code if response is not None else "no-response"
            body = response.text[:500] if response is not None else ""
            raise RuntimeError(f"Bitquery HTTP {status}: {body}")
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"Bitquery GraphQL error: {payload['errors'][:2]}")
        return payload.get("data") or {}

    def fetch_chain(self, chain: str, hours: int, max_pages: int = MAX_PAGES) -> list[dict[str, Any]]:
        if chain not in {"ethereum", "bsc", "solana"}:
            raise ValueError(f"Unsupported chain: {chain}")
        query = SOLANA_QUERY if chain == "solana" else EVM_QUERY.replace(
            "NETWORK_PLACEHOLDER", "eth" if chain == "ethereum" else "bsc"
        )
        root = "Solana" if chain == "solana" else "EVM"
        rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for page in range(max_pages):
            data = self.query(query, {
                "hoursAgo": hours,
                "limit": PAGE_SIZE,
                "offset": page * PAGE_SIZE,
            })
            items = ((data.get(root) or {}).get("Transfers") or [])
            if not items:
                break
            for item in items:
                row = normalize_solana_transfer(item) if chain == "solana" else normalize_evm_transfer(item, chain)
                if row:
                    key = (row["chain"], row["tx_hash"], row["event_index"])
                    rows_by_key[key] = row
            if len(items) < PAGE_SIZE:
                break
        else:
            # Never silently present a capped/truncated page scan as complete.
            raise RuntimeError(
                f"{chain}: reached MAX_PAGES={max_pages} with full pages; "
                "collection is incomplete. Increase the page budget or use smaller time windows."
            )
        return list(rows_by_key.values())


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
        response = http.post(
            endpoint,
            params={"on_conflict": "chain,tx_hash,event_index"},
            headers=headers,
            json=rows[start:start + 500],
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(
                f"Supabase onchain_transfers upload failed: HTTP {response.status_code}: {response.text[:1000]}"
            )
    return len(rows)


def collect(hours: int = LOOKBACK_HOURS, max_pages: int = MAX_PAGES) -> dict[str, int]:
    token = os.environ.get("BITQUERY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Missing BITQUERY_ACCESS_TOKEN GitHub secret")
    client = BitqueryClient(token)
    all_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for chain in ("ethereum", "bsc", "solana"):
        chain_rows = client.fetch_chain(chain, hours=hours, max_pages=max_pages)
        counts[chain] = len(chain_rows)
        all_rows.extend(chain_rows)
    observed = datetime.now(timezone.utc).isoformat()
    for row in all_rows:
        row["observed_at_utc"] = observed
    counts["uploaded_attempted"] = upload_transfers(all_rows)
    counts["total"] = len(all_rows)
    print(json.dumps({
        "provider": PROVIDER,
        "lookback_hours": hours,
        "usd_filter": "none",
        "max_pages_per_chain": max_pages,
        **counts,
    }, sort_keys=True))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all provider-visible on-chain token transfers")
    parser.add_argument("--hours", type=int, default=int(os.getenv("ONCHAIN_LOOKBACK_HOURS", LOOKBACK_HOURS)))
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("ONCHAIN_MAX_PAGES", MAX_PAGES)))
    args = parser.parse_args()
    if args.hours < 1 or args.hours > 24 or args.max_pages < 1:
        parser.error("--hours must be 1-24 and --max-pages must be positive")
    collect(hours=args.hours, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
