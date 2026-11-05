import json
from decimal import Decimal

from src.routescan_whale_collector import RoutescanClient, normalize_routescan_transfer, upload_transfers


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


def test_routescan_paginates_and_deduplicates(monkeypatch):
    import src.routescan_whale_collector as collector
    monkeypatch.setattr(collector, "BASE_INTERVAL_SECONDS", 3600)
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


def test_routescan_fails_instead_of_silently_truncating(monkeypatch):
    import src.routescan_whale_collector as collector
    monkeypatch.setattr(collector, "BASE_INTERVAL_SECONDS", 3600)
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


def test_routescan_splits_dense_windows_and_deduplicates_overlap(monkeypatch):
    import src.routescan_whale_collector as collector
    monkeypatch.setattr(collector, "BASE_INTERVAL_SECONDS", 3600)
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


def test_routescan_enforces_total_request_budget(monkeypatch):
    import src.routescan_whale_collector as collector
    monkeypatch.setattr(collector, "BASE_INTERVAL_SECONDS", 3600)
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


def test_upload_transfers_serializes_decimal_values_without_float_loss(monkeypatch):
    class FakeResponse:
        ok = True

    class FakeSession:
        def __init__(self):
            self.payload = None

        def post(self, url, params, headers, data, timeout):
            self.payload = json.loads(data)
            return FakeResponse()

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "test-secret")
    session = FakeSession()
    rows = [{
        "chain": "ethereum",
        "tx_hash": "0xabc",
        "event_index": 1,
        "amount_token": Decimal("0.000000000000000123456789"),
        "amount_usd": Decimal("12345678901234567890.123456789"),
        "source_payload": {"exact": Decimal("1.0000000000000000001")},
    }]

    assert upload_transfers(rows, session=session) == 1
    assert session.payload[0]["amount_token"] == "0.000000000000000123456789"
    assert session.payload[0]["amount_usd"] == "12345678901234567890.123456789"
    assert session.payload[0]["source_payload"]["exact"] == "1.0000000000000000001"



def test_dense_interval_fetches_newest_half_before_older_half():
    from datetime import datetime, timedelta, timezone

    item = sample_transfer()

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
            self.params = []
            self.calls = 0

        def get(self, url, params, headers, timeout):
            self.calls += 1
            self.params.append(dict(params))
            if self.calls == 1:
                return FakeResponse({"items": [item], "link": {"next": "root-more"}})
            return FakeResponse({"items": [dict(item, txHash=f"0x{self.calls}")], "link": {}})

    session = FakeSession()
    client = RoutescanClient(session=session, sleep=lambda _: None)
    start = datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(seconds=10)
    client._fetch_interval("ethereum", 1, start, end, 1, [0], 3)

    # First child request must cover the newer/right half, not the oldest half.
    first_child_start = datetime.fromisoformat(session.params[1]["timestampFrom"].replace("Z", "+00:00"))
    assert first_child_start > start + timedelta(seconds=4)


def test_routescan_collects_only_ethereum_after_bsc_split(monkeypatch):
    import src.routescan_whale_collector as collector

    attempted_chains = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_chain(self, chain, chain_id, **kwargs):
            attempted_chains.append(chain)
            if chain == "ethereum":
                raise RuntimeError("request budget reached")
            return []

    monkeypatch.setattr(collector, "RoutescanClient", FakeClient)
    monkeypatch.setattr(collector, "upload_transfers", lambda rows: len(rows))

    try:
        collector.collect(hours=1, max_pages=1, max_total_requests=1)
    except RuntimeError as exc:
        assert "ethereum" in str(exc)
    else:
        raise AssertionError("Expected incomplete collection error")

    assert attempted_chains == ["ethereum"]


def test_routescan_api_key_is_sent_only_in_header():
    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            return {"items": [], "link": {}}

    class FakeSession:
        def __init__(self):
            self.params = None
            self.headers = None

        def get(self, url, params, headers, timeout):
            self.params = dict(params)
            self.headers = dict(headers)
            return FakeResponse()

    session = FakeSession()
    client = RoutescanClient(api_key="test-routescan-key", session=session, sleep=lambda _: None)
    client._get_page(56, {"limit": 100})

    assert "apikey" not in session.params
    assert session.headers == {"apikey": "test-routescan-key"}


def test_routescan_http_400_reports_provider_response_body():
    class FakeResponse:
        status_code = 400
        headers = {}
        text = '{"message":"invalid request parameter"}'

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    client = RoutescanClient(session=FakeSession(), sleep=lambda _: None)
    try:
        client._get_page(56, {"limit": 100})
    except RuntimeError as exc:
        assert "HTTP 400" in str(exc)
        assert "chain 56" in str(exc)
        assert "invalid request parameter" in str(exc)
    else:
        raise AssertionError("Expected Routescan HTTP 400 to become a diagnostic RuntimeError")



def test_routescan_fetch_chain_uses_contiguous_newest_first_one_minute_slices(monkeypatch):
    import src.routescan_whale_collector as collector

    intervals = []

    class RecordingClient(RoutescanClient):
        def _fetch_interval(self, chain, chain_id, start, end, max_pages,
                            budget, max_total_requests, depth=0, on_interval=None):
            intervals.append((start, end))
            return []

    monkeypatch.setattr(collector, "BASE_INTERVAL_SECONDS", 60)
    client = RecordingClient(sleep=lambda _: None)
    client.fetch_chain("ethereum", 1, hours=1, max_pages=50, max_total_requests=100)

    assert len(intervals) == 60
    assert all(intervals[i][0] == intervals[i + 1][1] for i in range(len(intervals) - 1))
    assert all(intervals[i][1] > intervals[i][0] for i in range(len(intervals)))
    assert intervals[0][1] > intervals[-1][1]



def test_routescan_fetch_chain_supports_short_minute_lookback(monkeypatch):
    import src.routescan_whale_collector as collector

    intervals = []

    class RecordingClient(RoutescanClient):
        def _fetch_interval(self, chain, chain_id, start, end, max_pages,
                            budget, max_total_requests, depth=0, on_interval=None):
            intervals.append((start, end))
            return []

    monkeypatch.setattr(collector, "BASE_INTERVAL_SECONDS", 60)
    client = RecordingClient(sleep=lambda _: None)
    client.fetch_chain("ethereum", 1, minutes=15, max_pages=50, max_total_requests=100)

    assert len(intervals) == 15
    assert all(intervals[i][0] == intervals[i + 1][1] for i in range(len(intervals) - 1))
    assert all(intervals[i][1] > intervals[i][0] for i in range(len(intervals)))
