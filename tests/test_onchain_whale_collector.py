from src.onchain_whale_collector import normalize_evm_transfer, normalize_solana_transfer


def test_normalize_evm_transfer_preserves_usd_and_event_index():
    item = {
        "Block": {"Time": "2026-10-31T10:00:00Z", "Number": 123},
        "Transaction": {"Hash": "0xabc", "Index": 4},
        "Transfer": {
            "Index": 7, "Amount": "2500", "AmountInUSD": "125000",
            "Sender": "0xfrom", "Receiver": "0xto", "Type": "call",
            "Currency": {"Name": "Example", "Symbol": "EX", "SmartContract": "0xtoken", "Decimals": 18, "Native": False},
        },
    }
    row = normalize_evm_transfer(item, "ethereum")
    assert row["event_index"] == 7
    assert row["amount_usd"] == 125000
    assert row["token_address"] == "0xtoken"
    assert row["block_time_utc"] == "2026-10-31T10:00:00+00:00"


def test_normalize_solana_transfer_uses_mint_and_signature():
    item = {
        "Block": {"Time": "2026-10-31T10:00:00Z", "Slot": 99},
        "Transaction": {"Signature": "sig"},
        "Transfer": {
            "Index": 2, "Amount": "100", "AmountInUSD": "110000",
            "Sender": {"Address": "from"}, "Receiver": {"Address": "to"},
            "Currency": {"Name": "Token", "Symbol": "TOK", "MintAddress": "mint", "Decimals": 6, "Native": False},
        },
    }
    row = normalize_solana_transfer(item)
    assert row["chain"] == "solana"
    assert row["tx_hash"] == "sig"
    assert row["event_index"] == 2
    assert row["token_address"] == "mint"
    assert row["amount_usd"] == 110000


def test_missing_transfer_endpoints_are_skipped():
    assert normalize_evm_transfer({"Transfer": {}, "Transaction": {}, "Block": {}}, "bsc") is None
