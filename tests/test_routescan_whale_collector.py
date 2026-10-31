from decimal import Decimal

from src.routescan_whale_collector import RoutescanClient, normalize_routescan_transfer


def sample_transfer():
    return {
        "blockNumber": 12345,
        "txHash": "0xabc",
        "logIndex": 8,
        "from": "0xfrom",
        "to": "0xto",
        "timestamp": "2026-10-31T10:00:00Z",
        "amount": "1234567",
        "amountInUsd": None,
        "tokenAddress": "0xtoken",
        "tokenName": "Tiny Token",
        "tokenSymbol": "TINY",
        "tokenDecimals": 6,
    }


def test_routescan_normalizer_keeps_low_or_unpriced_transfers():
    row = normalize_routescan_transfer(sample_transfer(), "ethereum")
    assert row is not None
    assert row["chain"] == "ethereum"
    assert row["event_index"] == 8
    assert row["amount_raw"] == "1234567"
    assert row["amount_token"] == Decimal("1.234567")
    assert row["amount_usd"] is None
    assert row["provider"] == "routescan_v2"


def test_routescan_normalizer_rejects_incomplete_events():
    assert normalize_routescan_transfer({"txHash": "0x1"}, "bsc") is None


def test_routescan_paginates_and_deduplicates():
    first = sample_transfer()
    second = dict(sample_transfer(), txHash="0xdef", logIndex=9)

    class FakeResponse:
        status_code = 200
        headers = {}

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, params, headers, timeout):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse({"items": [first, first], "link": {"next": "cursor-2"}})
            return FakeResponse({"items": [second], "link": {}})

    session = FakeSession()
    client = RoutescanClient(session=session, sleep=lambda _: None)
    rows = client.fetch_chain("ethereum", 1, hours=1, max_pages=3)
    assert session.calls == 2
    assert {row["tx_hash"] for row in rows} == {"0xabc", "0xdef"}


def test_routescan_fails_instead_of_silently_truncating():
    item = sample_transfer()

    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"items": [item], "link": {"next": "still-more"}}

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    client = RoutescanClient(session=FakeSession(), sleep=lambda _: None)
    try:
        client.fetch_chain("ethereum", 1, hours=1, max_pages=1)
    except RuntimeError as exc:
        assert "refusing truncation" in str(exc)
    else:
        raise AssertionError("Expected explicit pagination-cap error")


def test_routescan_cursor_extracts_token_from_next_url():
    link = {"next": "/v2/network/mainnet/evm/1/erc20-transfers?limit=100&next=opaque-cursor"}
    assert RoutescanClient._cursor_from_link(link) == "opaque-cursor"


def test_routescan_splits_dense_windows_and_deduplicates_overlap():
    first = sample_transfer()

    class FakeResponse:
        status_code = 200
        headers = {}

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse({"items": [first], "link": {"next": "cursor"}})
            return FakeResponse({"items": [first], "link": {}})

    session = FakeSession()
    client = RoutescanClient(session=session, sleep=lambda _: None)
    rows = client.fetch_chain("ethereum", 1, hours=1, max_pages=1)
    assert session.calls == 3
    assert len(rows) == 1
    assert rows[0]["tx_hash"] == "0xabc"


def test_routescan_enforces_total_request_budget():
    item = sample_transfer()

    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"items": [item], "link": {"next": "cursor"}}

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    client = RoutescanClient(session=FakeSession(), sleep=lambda _: None)
    try:
        client.fetch_chain("ethereum", 1, hours=1, max_pages=1, max_total_requests=1)
    except RuntimeError as exc:
        assert "safety cap of 1 requests per chain" in str(exc)
    else:
        raise AssertionError("Expected request budget protection")



def test_routescan_persists_each_completed_interval_before_later_failure():
    item = sample_transfer()
    completed = []

    class FakeResponse:
        status_code = 200
        headers = {}

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # Root interval reaches its page cap and must split.
                return FakeResponse({"items": [item], "link": {"next": "root-more"}})
            # First child interval completes; the next child cannot start
            # because the shared request budget has been consumed.
            return FakeResponse({"items": [dict(item, txHash="0xdef")], "link": {}})

    client = RoutescanClient(session=FakeSession(), sleep=lambda _: None)
    from datetime import datetime, timedelta, timezone
    start = datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(seconds=10)
    try:
        client._fetch_interval("ethereum", 1, start, end, 1, [0], 2,
                               on_interval=lambda rows: completed.extend(rows))
    except RuntimeError as exc:
        assert "safety cap of 2 requests per chain" in str(exc)
    else:
        raise AssertionError("Expected request-budget failure")
    assert len(completed) == 1
    assert completed[0]["tx_hash"] == "0xdef"
