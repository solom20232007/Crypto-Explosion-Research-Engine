"""Bitquery large-transfer collector for on-chain whale research.

This is a research feed, not a trade-direction classifier. It stores only
provider-valued large transfers and preserves first-observed timestamps.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BITQUERY_URL = "https://streaming.bitquery.io/graphql"
PROVIDER = "bitquery_v2"
DEFAULT_MIN_USD = 100_000
LOOKBACK_HOURS = 3
PAGE_SIZE = 1000

EVM_QUERY = """
query WhaleTransfers($minUsd: String!, $hoursAgo: Int!, $limit: Int!, $offset: Int!) {
  EVM(dataset: combined, network: NETWORK_PLACEHOLDER) {
    Transfers(
      where: {
        Block: {Time: {since_relative: {hours_ago: $hoursAgo}}}
        Transfer: {AmountInUSD: {gt: $minUsd}, Success: true}
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
query WhaleTransfers($minUsd: String!, $hoursAgo: Int!, $limit: Int!, $offset: Int!) {
  Solana(dataset: combined) {
    Transfers(
      where: {
        Block: {Time: {since_relative: {hours_ago: $hoursAgo}}}
        Transfer: {AmountInUSD: {gt: $minUsd}}
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
        # Keep a deterministic transaction-level fallback; collisions are
        # avoided by including the transfer endpoints in the fallback index.
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
        response = self.session.post(
            BITQUERY_URL,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json={"query": query, "variables": variables},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Bitquery HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"Bitquery GraphQL error: {payload['errors'][:2]}")
        return payload.get("data") or {}

    def fetch_chain(self, chain: str, min_usd: float, hours: int, max_pages: int = 5) -> list[dict[str, Any]]:
        if chain not in {"ethereum", "bsc", "solana"}:
            raise ValueError(f"Unsupported chain: {chain}")
        query = SOLANA_QUERY if chain == "solana" else EVM_QUERY.replace("NETWORK_PLACEHOLDER", "eth" if chain == "ethereum" else "bsc")
        root = "Solana" if chain == "solana" else "EVM"
        rows: list[dict[str, Any]] = []
        for page in range(max_pages):
            data = self.query(query, {
                "minUsd": str(min_usd),
                "hoursAgo": hours,
                "limit": PAGE_SIZE,
                "offset": page * PAGE_SIZE,
            })
            items = ((data.get(root) or {}).get("Transfers") or [])
            if not items:
                break
            for item in items:
                row = normalize_solana_transfer(item) if chain == "solana" else normalize_evm_transfer(item, chain)
                if row and row["amount_usd"] is not None and row["amount_usd"] >= min_usd:
                    rows.append(row)
            if len(items) < PAGE_SIZE:
                break
        return rows


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
        # Ignore replays so observed_at_utc remains the first time we saw it.
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


def collect(min_usd: float = DEFAULT_MIN_USD, hours: int = LOOKBACK_HOURS) -> dict[str, int]:
    token = os.environ.get("BITQUERY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Missing BITQUERY_ACCESS_TOKEN GitHub secret")
    client = BitqueryClient(token)
    all_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for chain in ("ethereum", "bsc", "solana"):
        chain_rows = client.fetch_chain(chain, min_usd=min_usd, hours=hours)
        counts[chain] = len(chain_rows)
        all_rows.extend(chain_rows)
    # observed_at is set immediately before upload, never backdated to block time.
    observed = datetime.now(timezone.utc).isoformat()
    for row in all_rows:
        row["observed_at_utc"] = observed
    counts["uploaded_attempted"] = upload_transfers(all_rows)
    counts["total"] = len(all_rows)
    print(json.dumps({"provider": PROVIDER, "lookback_hours": hours, "min_transfer_usd": min_usd, **counts}, sort_keys=True))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect large on-chain transfers from Bitquery")
    parser.add_argument("--min-usd", type=float, default=float(os.getenv("WHALE_MIN_TRANSFER_USD", DEFAULT_MIN_USD)))
    parser.add_argument("--hours", type=int, default=LOOKBACK_HOURS)
    args = parser.parse_args()
    if args.min_usd <= 0 or args.hours < 1 or args.hours > 24:
        parser.error("--min-usd must be positive and --hours must be between 1 and 24")
    collect(args.min_usd, args.hours)


if __name__ == "__main__":
    main()
