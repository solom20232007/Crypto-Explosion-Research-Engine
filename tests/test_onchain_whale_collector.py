from datetime import datetime, timezone

from src.onchain_whale_collector import (
    EVM_QUERY,
    SOLANA_QUERY,
    BitqueryClient,
    normalize_evm_transfer,
    normalize_solana_transfer,
)


def test_normalize_evm_transfer_keeps_small_transfer_without_usd_value():
    item = {
        "Block": {"Time": "2026-10-31T10:00:00Z", "Number": 123},
        "Transaction": {"Hash": "0xabc", "Index": 4},
        "Transfer": {
            "Index": 7, "Amount": "0.00001", "AmountInUSD": None,
            "Sender": "0xfrom", "Receiver": "0xto", "Type": "call",
            "Currency": {"Name": "Micro Token", "Symbol": "MICRO", "SmartContract": "0xtoken", "Decimals": 18, "Native": False},
        },
    }
    row = normalize_evm_transfer(item, "ethereum")
    assert row["event_index"] == 7
    assert row["amount_token"] == 0.00001
    assert row["amount_usd"] is None
    assert row["token_address"] == "0xtoken"


def test_normalize_solana_transfer_uses_mint_and_signature_even_without_usd():
    item = {
        "Block": {"Time": "2026-10-31T10:00:00Z", "Slot": 99},
        "Transaction": {"Signature": "sig"},
        "Transfer": {
            "Index": 2, "Amount": "0.01", "AmountInUSD": None,
            "Sender": {"Address": "from"}, "Receiver": {"Address": "to"},
            "Currency": {"Name": "New Token", "Symbol": "NEW", "MintAddress": "mint", "Decimals": 6, "Native": False},
        },
    }
    row = normalize_solana_transfer(item)
    assert row["chain"] == "solana"
    assert row["tx_hash"] == "sig"
    assert row["event_index"] == 2
    assert row["token_address"] == "mint"
    assert row["amount_usd"] is None


def test_queries_do_not_filter_on_usd_or_minimum_transfer():
    assert "AmountInUSD: {gt:" not in EVM_QUERY
    assert "AmountInUSD: {gt:" not in SOLANA_QUERY
    assert "$minUsd" not in EVM_QUERY + SOLANA_QUERY


def test_fetch_chain_keeps_rows_with_missing_or_small_usd_and_deduplicates():
    item = {
        "Block": {"Time": "2026-10-31T10:00:00Z", "Number": 123},
        "Transaction": {"Hash": "0xabc"},
        "Transfer": {
            "Index": 7, "Amount": "1", "AmountInUSD": "0.02",
            "Sender": "0xfrom", "Receiver": "0xto",
            "Currency": {"Symbol": "TINY", "SmartContract": "0xtiny", "Decimals": 18},
        },
    }

    class FakeClient(BitqueryClient):
        def __init__(self):
            pass

        def query(self, query, variables):
            assert "since" in variables and "till" in variables
            return {"EVM": {"Transfers": [item]}}

    rows = FakeClient().fetch_chain("ethereum", hours=1, max_pages=2)
    assert len(rows) == 1
    assert rows[0]["amount_usd"] == 0.02


def test_fetch_chain_splits_dense_intervals_instead_of_truncating():
    item = {
        "Block": {"Time": "2026-10-31T10:00:00Z", "Number": 123},
        "Transaction": {"Hash": "0xsplit"},
        "Transfer": {
            "Index": 1, "Amount": "2", "AmountInUSD": None,
            "Sender": "0xfrom", "Receiver": "0xto",
            "Currency": {"Symbol": "TINY", "SmartContract": "0xtiny", "Decimals": 18},
        },
    }

    class DenseThenSparseClient(BitqueryClient):
        def __init__(self):
            self.calls = 0

        def query(self, query, variables):
            self.calls += 1
            assert variables["since"] < variables["till"]
            if self.calls == 1:
                return {"EVM": {"Transfers": [{}] * 1000}}
            return {"EVM": {"Transfers": [item]}}

    client = DenseThenSparseClient()
    rows = client.fetch_chain("ethereum", hours=1, max_pages=1)
    assert client.calls > 1
    assert len(rows) == 1
    assert rows[0]["tx_hash"] == "0xsplit"


def test_fetch_chain_raises_when_one_second_interval_is_still_capped():
    class PermanentlyDenseClient(BitqueryClient):
        def __init__(self):
            pass

        def query(self, query, variables):
            return {"EVM": {"Transfers": [{}] * 1000}}

    try:
        PermanentlyDenseClient()._fetch_interval(
            "ethereum", EVM_QUERY.replace("NETWORK_PLACEHOLDER", "eth"), "EVM",
            datetime(2026, 10, 31, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 10, 31, 10, 0, 1, tzinfo=timezone.utc),
            max_pages=1,
        )
    except RuntimeError as exc:
        assert "cannot guarantee complete collection" in str(exc)
    else:
        raise AssertionError("Expected explicit minimum-window truncation error")


def test_missing_transfer_endpoints_are_skipped():
    assert normalize_evm_transfer({"Transfer": {}, "Transaction": {}, "Block": {}}, "bsc") is None


def test_interval_queries_use_supported_bitquery_datetime_scalar():
    assert "$since: DateTime!" in EVM_QUERY
    assert "$till: DateTime!" in EVM_QUERY
    assert "$since: DateTime!" in SOLANA_QUERY
    assert "$till: DateTime!" in SOLANA_QUERY
    assert "ISO8601DateTime" not in EVM_QUERY + SOLANA_QUERY
