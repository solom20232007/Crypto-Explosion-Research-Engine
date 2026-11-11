"""Durable, monotonic checkpoints for resumable on-chain historical collection."""
from __future__ import annotations

import os
from typing import Any

import requests


class SupabaseCheckpointStore:
    def __init__(self, url: str | None = None, key: str | None = None,
                 session: requests.Session | None = None):
        self.url = (url or os.environ["SUPABASE_URL"]).rstrip("/")
        self.key = key or os.environ["SUPABASE_SECRET_KEY"]
        self.session = session or requests.Session()
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def get(self, provider: str, chain: str) -> dict[str, Any] | None:
        response = self.session.get(
            f"{self.url}/rest/v1/onchain_collection_checkpoints",
            params={"provider": f"eq.{provider}", "chain": f"eq.{chain}", "select": "*", "limit": "1"},
            headers=self.headers, timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"Supabase checkpoint read failed: HTTP {response.status_code}: {response.text[:1000]}"
            )
        rows = response.json()
        if not isinstance(rows, list):
            raise RuntimeError("Supabase checkpoint read returned a non-list response")
        return rows[0] if rows else None

    def advance(self, *, provider: str, chain: str, cursor_kind: str,
                cursor_value: int, initial_cursor_value: int | None = None,
                completed_from: int | None = None, completed_to: int | None = None,
                status: str = "ready", last_error: str | None = None) -> dict[str, Any]:
        if cursor_kind not in {"unix_seconds", "block_number"}:
            raise ValueError("cursor_kind must be unix_seconds or block_number")
        if cursor_value < 0 or (initial_cursor_value is not None and initial_cursor_value < 0):
            raise ValueError("checkpoint cursor values must be non-negative")
        if status not in {"ready", "running", "paused", "complete", "error"}:
            raise ValueError("unsupported checkpoint status")
        payload = {
            "p_provider": provider,
            "p_chain": chain,
            "p_cursor_kind": cursor_kind,
            "p_cursor_value": cursor_value,
            "p_initial_cursor_value": initial_cursor_value,
            "p_last_completed_from": completed_from,
            "p_last_completed_to": completed_to,
            "p_status": status,
            "p_last_error": last_error,
        }
        response = self.session.post(
            f"{self.url}/rest/v1/rpc/advance_onchain_collection_checkpoint",
            headers={**self.headers, "Prefer": "return=representation"},
            json=payload, timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"Supabase checkpoint advance failed: HTTP {response.status_code}: {response.text[:1000]}"
            )
        result = response.json()
        if isinstance(result, list):
            if not result:
                raise RuntimeError("Supabase checkpoint advance returned no row")
            return result[0]
        if isinstance(result, dict):
            return result
        raise RuntimeError("Supabase checkpoint advance returned an unexpected response")
