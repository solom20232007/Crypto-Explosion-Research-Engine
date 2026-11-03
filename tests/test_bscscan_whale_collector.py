from datetime import datetime, timezone

from src.bscscan_whale_collector import (
    BscScanClient,
    PROVIDER,
    TRANSFER_TOPIC,
    normalize_bsc_log,
)


def sample_log(log_index="0x2", block_number="0x123"):
    return {
        "address": "0x" + "a" * 40,
        "topics": [
            TRANSFER_TOPIC,
            "0x" + "0" * 24 + "1" * 40,
            "0x" + "0" * 24 + "2" * 40,
        ],
        "data": "0x" + format(123456789, "064x"),
        "blockNumber": block_number,
        "timeStamp": hex(1793527200),
        "gasPrice": "0x1",
        "gasUsed": "0x2",
        "logIndex": log_index,
        "transactionHash": "0x" + "b" * 64,
        "transactionIndex": "0x1",
    }


def test_normalize_global_bep20_transfer_log():
    row = normalize_bsc_log(sample_log())
    assert row is not None
    assert row["chain"] == "bsc"
    assert row["provider"] == PROVIDER
    assert row["event_index"] == 2
    assert row["block_number"] == 0x123
    assert row["from_address"] == "0x" + "1" * 40
    assert row["to_address"] == "0x" + "2" * 40
    assert row["amount_raw"] == "123456789"
    assert row["amount_token"] is None
    assert row["token_decimals"] is None


def test_normalizer_rejects_non_transfer_or_malformed_log():
    assert normalize_bsc_log({"topics": [TRANSFER_TOPIC]}) is None
    item = sample_log()
    item.pop("timeStamp")
    assert normalize_bsc_log(item) is None


def test_bscscan_request_uses_v2_chain_56_and_transfer_topic():
    class Response:
        status_code = 200
        headers = {}

        def json(self):
            return {"status": "1", "message": "OK", "result": []}

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, params, timeout):
            self.calls.append((url, dict(params)))
            return Response()

    session = Session()
    client = BscScanClient("test-key", session=session, sleep=lambda _: None)
    assert client._request({
        "module": "logs", "action": "getLogs",
        "fromBlock": 1, "toBlock": 2, "topic0": TRANSFER_TOPIC,
        "page": 1, "offset": 1000,
    }) == []
    url, params = session.calls[0]
    assert url == "https://api.etherscan.io/v2/api"
    assert params["chainid"] == 56
    assert params["apikey"] == "test-key"
    assert params["topic0"] == TRANSFER_TOPIC


def test_bscscan_splits_full_1000_result_interval_and_checkpoints_leaves():
    class Response:
        status_code = 200
        headers = {}

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, url, params, timeout):
            self.calls += 1
            if params["fromBlock"] == params["toBlock"]:
                return Response({"status": "1", "message": "OK", "result": [sample_log()]})
            return Response({"status": "1", "message": "OK", "result": [sample_log()] * 1000})

    checkpoints = []
    client = BscScanClient("test-key", session=Session(), sleep=lambda _: None)
    rows = client._fetch_block_range(100, 103, on_interval=lambda batch: checkpoints.extend(batch))
    assert client.request_count == 3
    assert len(rows) == 1
    assert len(checkpoints) == 2


def test_bscscan_request_budget_is_enforced():
    class Response:
        status_code = 200
        headers = {}

        def json(self):
            return {"status": "1", "message": "OK", "result": []}

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    client = BscScanClient("test-key", session=Session(), sleep=lambda _: None, max_total_requests=1)
    client._request({"module": "block", "action": "getblocknobytime", "timestamp": 1, "closest": "before"})
    try:
        client._request({"module": "block", "action": "getblocknobytime", "timestamp": 2, "closest": "before"})
    except RuntimeError as exc:
        assert "safety cap of 1" in str(exc)
    else:
        raise AssertionError("Expected BscScan request cap")


def test_bscscan_no_records_is_a_valid_empty_result():
    class Response:
        status_code = 200
        headers = {}

        def json(self):
            return {"status": "0", "message": "No records found", "result": "No records found"}

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    client = BscScanClient("test-key", session=Session(), sleep=lambda _: None)
    assert client._request({"module": "logs", "action": "getLogs"}) == []


def test_bscscan_http_error_contains_response_body():
    class Response:
        status_code = 400
        headers = {}
        text = '{"message":"bad parameters"}'

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    client = BscScanClient("test-key", session=Session(), sleep=lambda _: None)
    try:
        client._request({"module": "logs", "action": "getLogs"})
    except RuntimeError as exc:
        assert "HTTP 400" in str(exc)
        assert "bad parameters" in str(exc)
    else:
        raise AssertionError("Expected diagnostic HTTP error")
