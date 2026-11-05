"""Global BNB Smart Chain BEP-20 Transfer collector using public BSC JSON-RPC.

Queries the chain-wide Transfer(address,address,uint256) event topic without
requiring a token contract or wallet address. Uses only recent blocks, so no
archive node is required. ERC-721 Transfer logs are excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

DEFAULT_RPC_URL = "https://bsc-dataseed.bnbchain.org"
PROVIDER = "bsc_public_rpc"
CHAIN = "bsc"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
MAX_LOGS_PER_INTERVAL = 2000
DEFAULT_MAX_TOTAL_REQUESTS = 1000


class BscRpcClient:
    def __init__(self, rpc_url: str | None = None, session: requests.Session | None = None,
                 sleep=time.sleep, max_total_requests: int = DEFAULT_MAX_TOTAL_REQUESTS):
        self.rpc_url = rpc_url or DEFAULT_RPC_URL
        self.session = session or requests.Session()
        self.sleep = sleep
        self.max_total_requests = max_total_requests
        self.request_count = 0
        self._request_id = 0

    def _post(self, payload: Any) -> Any:
        if self.request_count >= self.max_total_requests:
            raise RuntimeError(
                f"BSC RPC safety cap of {self.max_total_requests} requests reached; "
                "previously completed block intervals have already been uploaded."
            )
        self.request_count += 1
        try:
            response = self.session.post(self.rpc_url, json=payload, timeout=(10, 60))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            raise RuntimeError(f"BSC RPC network failure: {exc}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise RuntimeError(
                f"BSC RPC HTTP {response.status_code}: {(getattr(response, 'text', '') or '')[:1000]}"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"BSC RPC HTTP {response.status_code}: {(getattr(response, 'text', '') or '')[:1000]}"
            )
        try:
            payload_out = response.json()
        except ValueError as exc:
            raise RuntimeError("BSC RPC returned invalid JSON") from exc
        self.sleep(0.20)  # Conservative pacing for the shared public endpoint.
        return payload_out

    def _rpc(self, method: str, params: list[Any]) -> Any:
        self._request_id += 1
        payload = self._post({
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        })
        if not isinstance(payload, dict):
            raise RuntimeError(f"BSC RPC {method} returned a non-object response")
        if payload.get("error"):
            error = payload["error"]
            raise RuntimeError(f"BSC RPC {method} error: {error}")
        if "result" not in payload:
            raise RuntimeError(f"BSC RPC {method} response is missing result")
        return payload["result"]

    def _rpc_batch(self, calls: list[tuple[str, list[Any]]]) -> list[Any]:
        payload = []
        ids = []
        for method, params in calls:
            self._request_id += 1
            ids.append(self._request_id)
            payload.append({
                "jsonrpc": "2.0", "id": self._request_id,
                "method": method, "params": params,
            })
        response = self._post(payload)
        if not isinstance(response, list):
            raise RuntimeError("BSC RPC does not support JSON-RPC batch responses")
        by_id = {item.get("id"): item for item in response if isinstance(item, dict)}
        results = []
        for request_id in ids:
            item = by_id.get(request_id)
            if item is None:
                raise RuntimeError(f"BSC RPC batch response missing id {request_id}")
            if item.get("error"):
                raise RuntimeError(f"BSC RPC batch error: {item['error']}")
            results.append(item.get("result"))
        return results

    @staticmethod
    def _hex_int(value: Any) -> int:
        return int(str(value), 16) if isinstance(value, str) and value.startswith("0x") else int(value)

    def _block(self, number: int) -> dict[str, Any]:
        result = self._rpc("eth_getBlockByNumber", [hex(number), False])
        if not isinstance(result, dict):
            raise RuntimeError(f"BSC RPC returned no block for height {number}")
        return result

    def _find_start_block(self, target_timestamp: int, latest_number: int) -> int:
        low, high = 0, latest_number
        while low < high:
            mid = (low + high) // 2
            block = self._block(mid)
            timestamp = self._hex_int(block["timestamp"])
            if timestamp < target_timestamp:
                low = mid + 1
            else:
                high = mid
        return max(0, low - 1)  # Include the block immediately before the boundary.

    def _block_timestamps(self, logs: list[dict[str, Any]]) -> dict[int, int]:
        numbers = sorted({self._hex_int(item["blockNumber"]) for item in logs})
        timestamps: dict[int, int] = {}
        for start in range(0, len(numbers), 100):
            chunk = numbers[start:start + 100]
            calls = [("eth_getBlockByNumber", [hex(number), False]) for number in chunk]
            results = self._rpc_batch(calls)
            for number, block in zip(chunk, results):
                if not isinstance(block, dict) or "timestamp" not in block:
                    raise RuntimeError(f"BSC RPC batch omitted block timestamp for {number}")
                timestamps[number] = self._hex_int(block["timestamp"])
        return timestamps

    @staticmethod
    def _normalize_log(item: dict[str, Any], timestamp: int) -> dict[str, Any] | None:
        topics = item.get("topics") or []
        data = str(item.get("data") or "")
        # Standard BEP-20 has exactly 3 topics and a 32-byte uint256 data word.
        # ERC-721 uses the same topic0 but has a fourth indexed topic and empty data.
        if len(topics) != 3 or len(data) != 66:
            return None
        tx_hash = item.get("transactionHash")
        token_address = item.get("address")
        if not tx_hash or not token_address:
            return None
        try:
            event_index = BscRpcClient._hex_int(item["logIndex"])
            block_number = BscRpcClient._hex_int(item["blockNumber"])
            amount_raw = str(BscRpcClient._hex_int(data))
        except (KeyError, TypeError, ValueError):
            return None
        return {
            "chain": CHAIN,
            "tx_hash": str(tx_hash),
            "event_index": event_index,
            "block_number": block_number,
            "block_time_utc": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
            "token_address": str(token_address),
            "token_symbol": None,
            "token_name": None,
            "from_address": "0x" + str(topics[1])[-40:],
            "to_address": "0x" + str(topics[2])[-40:],
            "amount_raw": amount_raw,
            "token_decimals": None,
            "amount_token": None,
            "amount_usd": None,
            "transfer_kind": "token",
            "provider": PROVIDER,
            "provider_event_id": str(event_index),
            "is_finalized": True,
            "source_payload": item,
        }

    def _fetch_block_range(self, start_block: int, end_block: int, on_interval=None,
                           depth: int = 0) -> list[dict[str, Any]]:
        if end_block < start_block:
            return []
        try:
            logs = self._rpc("eth_getLogs", [{
                "fromBlock": hex(start_block),
                "toBlock": hex(end_block),
                "topics": [TRANSFER_TOPIC],
            }])
        except RuntimeError as exc:
            message = str(exc).lower()
            splittable = any(term in message for term in (
                "too many", "more than", "limit exceeded", "response size",
                "query returned", "block range", "timeout", "timed out",
            ))
            if not splittable:
                raise
            logs = None

        if logs is not None and not isinstance(logs, list):
            raise RuntimeError("BSC eth_getLogs returned a non-list result")

        # Some RPC gateways cap eth_getLogs results even for a single block.
        # A block cannot be split further, so retrieve every transaction receipt
        # and verify receipt coverage against the block transaction count before
        # accepting the complete set of Transfer logs.
        receipts_complete = False
        receipt_block_timestamp = None
        if start_block == end_block and (
            logs is None or len(logs) >= MAX_LOGS_PER_INTERVAL
        ):
            try:
                block = self._block(start_block)
                transactions = block.get("transactions")
                if not isinstance(transactions, list):
                    raise RuntimeError("block response omitted its transaction list")

                # Prefer the bulk receipt method. If the public RPC does not
                # implement it, fall back to standard per-transaction receipts
                # in JSON-RPC batches. Never accept partial receipt coverage.
                try:
                    receipts = self._rpc("eth_getBlockReceipts", [hex(start_block)])
                    if not isinstance(receipts, list):
                        raise RuntimeError("eth_getBlockReceipts returned a non-list result")
                except RuntimeError:
                    receipts = []
                    for offset in range(0, len(transactions), 50):
                        hashes = transactions[offset:offset + 50]
                        calls = [("eth_getTransactionReceipt", [tx_hash]) for tx_hash in hashes]
                        receipts.extend(self._rpc_batch(calls))

                if len(receipts) != len(transactions) or any(
                    not isinstance(receipt, dict) for receipt in receipts
                ):
                    raise RuntimeError(
                        f"receipt coverage mismatch: block has {len(transactions)} transactions "
                        f"but RPC returned {len(receipts)} complete receipts"
                    )
                receipt_logs = []
                for receipt in receipts:
                    if not isinstance(receipt.get("logs"), list):
                        raise RuntimeError("a transaction receipt omitted its logs array")
                    for item in receipt["logs"]:
                        topics = item.get("topics") or []
                        if topics and str(topics[0]).lower() == TRANSFER_TOPIC:
                            receipt_logs.append(item)
                logs = receipt_logs
                receipt_block_timestamp = self._hex_int(block["timestamp"])
                receipts_complete = True
            except RuntimeError as exc:
                reason = (
                    "eth_getLogs exceeded the provider result cap"
                    if logs is None else f"eth_getLogs returned at least {MAX_LOGS_PER_INTERVAL} logs"
                )
                raise RuntimeError(
                    f"BSC block {start_block}: {reason}; complete receipt fallback failed: {exc}. "
                    "No potentially truncated block data was uploaded."
                ) from exc

        if logs is not None and (len(logs) < MAX_LOGS_PER_INTERVAL or receipts_complete):
            timestamps = (
                {start_block: receipt_block_timestamp}
                if receipts_complete
                else (self._block_timestamps(logs) if logs else {})
            )
            rows = []
            for item in logs:
                number = self._hex_int(item["blockNumber"])
                row = self._normalize_log(item, timestamps[number])
                if row:
                    rows.append(row)
            unique = {(row["chain"], row["tx_hash"], row["event_index"]): row for row in rows}
            completed = list(unique.values())
            if on_interval and completed:
                on_interval(completed)
            return completed

        if start_block == end_block or depth >= 32:
            raise RuntimeError(
                f"BSC block interval {start_block}..{end_block} still exceeds "
                f"{MAX_LOGS_PER_INTERVAL} logs at minimum split size; refusing truncation."
            )
        midpoint = (start_block + end_block) // 2
        # Newest half first: if the request budget is reached, recent data is retained.
        newer = self._fetch_block_range(midpoint + 1, end_block, on_interval, depth + 1)
        older = self._fetch_block_range(start_block, midpoint, on_interval, depth + 1)
        unique = {(row["chain"], row["tx_hash"], row["event_index"]): row for row in newer + older}
        return list(unique.values())

    def fetch_window(self, hours: int = 1, on_interval=None) -> list[dict[str, Any]]:
        latest_hex = self._rpc("eth_blockNumber", [])
        latest_number = self._hex_int(latest_hex)
        latest_block = self._block(latest_number)
        latest_timestamp = self._hex_int(latest_block["timestamp"])
        start_block = self._find_start_block(latest_timestamp - hours * 3600, latest_number)
        return self._fetch_block_range(start_block, latest_number, on_interval=on_interval)


def collect(hours: int = 1, max_total_requests: int = DEFAULT_MAX_TOTAL_REQUESTS) -> dict[str, int]:
    from src.routescan_whale_collector import upload_transfers

    client = BscRpcClient(
        rpc_url=os.getenv("BSC_RPC_URL") or DEFAULT_RPC_URL,
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
            "rpc_requests_used": client.request_count,
        }, sort_keys=True), flush=True)

    rows = client.fetch_window(hours=hours, on_interval=persist)
    result = {
        "provider": PROVIDER,
        "chain": CHAIN,
        "lookback_hours": hours,
        "scope": "global_bep20_transfer_events",
        "total": len(rows),
        "uploaded_attempted": uploaded_attempted,
        "rpc_requests_used": client.request_count,
        "max_total_requests": max_total_requests,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect global BSC BEP-20 Transfer events using public JSON-RPC")
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
