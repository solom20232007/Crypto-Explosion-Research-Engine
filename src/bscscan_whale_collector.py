"""Global BNB Smart Chain BEP-20 Transfer event collector via Etherscan V2.

Uses the chain-wide Logs API filtered by the ERC-20/BEP-20 Transfer topic.
Unlike account token-transfer endpoints, this can collect events without
knowing token contracts or wallet addresses. Token metadata is not inferred.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

API_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 56
CHAIN = "bsc"
PROVIDER = "etherscan_v2_bsc"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
PAGE_SIZE = 1000
DEFAULT_MAX_TOTAL_REQUESTS = 300


def _int_value(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text, 16) if text.lower().startswith("0x") else int(text)


def _utc_timestamp(value: Any) -> str:
    return datetime.fromtimestamp(_int_value(value), tz=timezone.utc).isoformat()


def normalize_bsc_log(item: dict[str, Any]) -> dict[str, Any] | None:
    topics = item.get("topics") or []
    if len(topics) < 3:
        return None
    tx_hash = item.get("transactionHash")
    token_address = item.get("address")
    if not tx_hash or not token_address:
        return None
    sender = "0x" + str(topics[1])[-40:]
    receiver = "0x" + str(topics[2])[-40:]
    raw_data = str(item.get("data") or "0x0")
    try:
        raw_amount = str(_int_value(raw_data))
        event_index = _int_value(item.get("logIndex", 0))
        block_number = _int_value(item["blockNumber"]) if item.get("blockNumber") is not None else None
        block_time = _utc_timestamp(item["timeStamp"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return {
        "chain": CHAIN,
        "tx_hash": str(tx_hash),
        "event_index": event_index,
        "block_number": block_number,
        "block_time_utc": block_time,
        "token_address": str(token_address),
        "token_symbol": None,
        "token_name": None,
        "from_address": sender,
        "to_address": receiver,
        "amount_raw": raw_amount,
        "token_decimals": None,
        "amount_token": None,
        "amount_usd": None,
        "transfer_kind": "token",
        "provider": PROVIDER,
        "provider_event_id": str(event_index),
        "is_finalized": True,
        "source_payload": item,
    }


class BscScanClient:
    def __init__(self, api_key: str, session: requests.Session | None = None,
                 sleep=time.sleep, max_total_requests: int = DEFAULT_MAX_TOTAL_REQUESTS):
        if not api_key:
            raise ValueError("BSC_SCAN_API_KEY is required")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.sleep = sleep
        self.max_total_requests = max_total_requests
        self.request_count = 0

    def _request(self, params: dict[str, Any]) -> Any:
        if self.request_count >= self.max_total_requests:
            raise RuntimeError(
                f"BscScan safety cap of {self.max_total_requests} API requests reached; "
                "completed block intervals have already been checkpointed."
            )
        query = {"chainid": CHAIN_ID, "apikey": self.api_key, **params}
        for attempt in range(6):
            self.request_count += 1
            try:
                response = self.session.get(API_URL, params=query, timeout=(10, 45))
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt == 5:
                    raise RuntimeError(f"BscScan network failure after 6 attempts: {exc}") from exc
                self.sleep(min(30, 2 ** attempt))
                if self.request_count >= self.max_total_requests:
                    raise RuntimeError("BscScan request safety cap reached during network retries") from exc
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 5:
                    raise RuntimeError(
                        f"BscScan HTTP {response.status_code} after 6 attempts: "
                        f"{(response.text or '')[:1000]}"
                    )
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = max(1, min(60, int(retry_after))) if retry_after else min(30, 2 ** attempt)
                except (TypeError, ValueError):
                    delay = min(30, 2 ** attempt)
                self.sleep(delay)
                if self.request_count >= self.max_total_requests:
                    raise RuntimeError("BscScan request safety cap reached during HTTP retries")
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"BscScan HTTP {response.status_code}: {(response.text or '')[:1000]}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError(f"BscScan returned invalid JSON: {(response.text or '')[:1000]}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("BscScan returned a non-object JSON response")
            status = str(payload.get("status", ""))
            result = payload.get("result")
            message = str(payload.get("message", ""))
            if status == "0":
                if isinstance(result, list) and not result:
                    return []
                if "no records found" in str(result).lower() or "no transactions found" in str(result).lower():
                    return []
                raise RuntimeError(f"BscScan API error: {message}: {str(result)[:1000]}")
            return result
        raise RuntimeError("BscScan request failed after retries")

    def _block_at(self, timestamp: int) -> int:
        result = self._request({
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": timestamp,
            "closest": "before",
        })
        try:
            return int(result)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"BscScan returned invalid block number for timestamp {timestamp}: {result}") from exc

    def _fetch_block_range(self, start_block: int, end_block: int, on_interval=None,
                           depth: int = 0) -> list[dict[str, Any]]:
        if end_block < start_block:
            return []
        result = self._request({
            "module": "logs",
            "action": "getLogs",
            "fromBlock": start_block,
            "toBlock": end_block,
            "topic0": TRANSFER_TOPIC,
            "page": 1,
            "offset": PAGE_SIZE,
        })
        if not isinstance(result, list):
            raise RuntimeError(f"BscScan getLogs returned unexpected result type: {type(result).__name__}")
        if len(result) < PAGE_SIZE:
            rows = [row for item in result if (row := normalize_bsc_log(item)) is not None]
            if on_interval and rows:
                on_interval(rows)
            return rows
        if start_block == end_block or depth >= 32:
            raise RuntimeError(
                f"BscScan block interval {start_block}..{end_block} still returned "
                f"{PAGE_SIZE} logs at minimum split size; refusing silent truncation."
            )
        midpoint = (start_block + end_block) // 2
        # Newest blocks first so a request-budget stop preserves the latest data.
        newer = self._fetch_block_range(midpoint + 1, end_block, on_interval, depth + 1)
        older = self._fetch_block_range(start_block, midpoint, on_interval, depth + 1)
        unique = {(row["chain"], row["tx_hash"], row["event_index"]): row for row in newer + older}
        return list(unique.values())

    def fetch_window(self, hours: int = 1, on_interval=None) -> list[dict[str, Any]]:
        now = int(datetime.now(timezone.utc).timestamp())
        start_block = self._block_at(now - hours * 3600)
        end_block = self._block_at(now)
        return self._fetch_block_range(start_block, end_block, on_interval=on_interval)


def collect(hours: int = 1, max_total_requests: int = DEFAULT_MAX_TOTAL_REQUESTS) -> dict[str, int]:
    from src.routescan_whale_collector import upload_transfers

    client = BscScanClient(
        os.getenv("BSC_SCAN_API_KEY", ""),
        max_total_requests=max_total_requests,
    )
    uploaded_attempted = 0

    def persist(rows: list[dict[str, Any]]) -> None:
        nonlocal uploaded_attempted
        observed = datetime.now(timezone.utc).isoformat()
        for row in rows:
            row["observed_at_utc"] = observed
        attempted = upload_transfers(rows)
        uploaded_attempted += attempted
        print(json.dumps({
            "provider": PROVIDER,
            "chain": CHAIN,
            "checkpoint_rows_attempted": attempted,
            "uploaded_attempted_total": uploaded_attempted,
            "api_requests_used": client.request_count,
        }, sort_keys=True), flush=True)

    rows = client.fetch_window(hours=hours, on_interval=persist)
    result = {
        "provider": PROVIDER,
        "chain": CHAIN,
        "lookback_hours": hours,
        "scope": "global_bep20_transfer_events",
        "total": len(rows),
        "uploaded_attempted": uploaded_attempted,
        "api_requests_used": client.request_count,
        "max_total_requests": max_total_requests,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect global BSC BEP-20 Transfer events via BscScan/Etherscan V2")
    parser.add_argument("--hours", type=int, default=int(os.getenv("ROUTESCAN_LOOKBACK_HOURS", "1")))
    parser.add_argument("--max-total-requests", type=int, default=int(
        os.getenv("ROUTESCAN_MAX_TOTAL_REQUESTS", str(DEFAULT_MAX_TOTAL_REQUESTS))
    ))
    args = parser.parse_args()
    if not 1 <= args.hours <= 24 or args.max_total_requests < 1:
        parser.error("--hours must be 1-24 and --max-total-requests must be positive")
    collect(args.hours, args.max_total_requests)


if __name__ == "__main__":
    main()
