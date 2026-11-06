from src.bsc_rpc_whale_collector import BscRpcClient, TRANSFER_TOPIC, PROVIDER


def sample_log(block="0x10", index="0x2"):
    return {
        "address": "0x" + "a" * 40,
        "topics": [
            TRANSFER_TOPIC,
            "0x" + "0" * 24 + "1" * 40,
            "0x" + "0" * 24 + "2" * 40,
        ],
        "data": "0x" + format(123456789, "064x"),
        "blockNumber": block,
        "logIndex": index,
        "transactionHash": "0x" + "b" * 64,
    }


def test_normalize_bep20_log_and_reject_erc721():
    row = BscRpcClient._normalize_log(sample_log(), 1793527200)
    assert row is not None
    assert row["chain"] == "bsc"
    assert row["provider"] == PROVIDER
    assert row["event_index"] == 2
    assert row["amount_raw"] == "123456789"
    assert row["from_address"] == "0x" + "1" * 40
    assert row["to_address"] == "0x" + "2" * 40
    erc721 = sample_log()
    erc721["topics"].append("0x" + "0" * 64)
    erc721["data"] = "0x"
    assert BscRpcClient._normalize_log(erc721, 1793527200) is None


def test_rpc_uses_chain_wide_transfer_topic():
    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"jsonrpc": "2.0", "id": 1, "result": []}

    class Session:
        def __init__(self):
            self.payload = None

        def post(self, url, json, timeout):
            self.payload = json
            return Response()

    session = Session()
    client = BscRpcClient(session=session, sleep=lambda _: None)
    assert client._rpc("eth_getLogs", [{
        "fromBlock": "0x1", "toBlock": "0x2", "topics": [TRANSFER_TOPIC],
    }]) == []
    assert session.payload["method"] == "eth_getLogs"
    assert session.payload["params"][0]["topics"] == [TRANSFER_TOPIC]


def test_rpc_batch_resolves_block_timestamps():
    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def post(self, url, json, timeout):
            return Response([
                {"jsonrpc": "2.0", "id": item["id"], "result": {"timestamp": "0x10"}}
                for item in json
            ])

    client = BscRpcClient(session=Session(), sleep=lambda _: None)
    assert client._block_timestamps([sample_log("0x10"), sample_log("0x11")]) == {
        16: 16, 17: 16,
    }


def test_splits_dense_block_ranges_and_checkpoints_completed_leaves():
    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def post(self, url, json, timeout):
            if isinstance(json, list):
                return Response([
                    {"jsonrpc": "2.0", "id": item["id"], "result": {"timestamp": "0x10"}}
                    for item in json
                ])
            params = json["params"][0]
            start, end = int(params["fromBlock"], 16), int(params["toBlock"], 16)
            if start == end:
                return Response({"jsonrpc": "2.0", "id": json["id"], "result": [sample_log(hex(start))]})
            return Response({"jsonrpc": "2.0", "id": json["id"], "result": [sample_log()] * 2000})

    client = BscRpcClient(session=Session(), sleep=lambda _: None)
    checkpoints = []
    rows = client._fetch_block_range(100, 101, on_interval=lambda batch: checkpoints.extend(batch))
    assert len(rows) == 1
    assert len(checkpoints) == 2


def test_rpc_request_budget_is_enforced():
    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"jsonrpc": "2.0", "id": 1, "result": "0x1"}

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    client = BscRpcClient(session=Session(), sleep=lambda _: None, max_total_requests=1)
    assert client._rpc("eth_blockNumber", []) == "0x1"
    try:
        client._rpc("eth_blockNumber", [])
    except RuntimeError as exc:
        assert "safety cap of 1" in str(exc)
    else:
        raise AssertionError("Expected RPC request safety cap")



def test_single_dense_block_uses_complete_receipts_instead_of_truncating(monkeypatch):
    import src.bsc_rpc_whale_collector as collector

    monkeypatch.setattr(collector, "MAX_LOGS_PER_INTERVAL", 1)

    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def post(self, url, json, timeout):
            method = json["method"]
            if method == "eth_getLogs":
                result = [sample_log("0x10", "0x1")]
            elif method == "eth_getBlockReceipts":
                result = [{
                    "transactionHash": "0x" + "b" * 64,
                    "blockNumber": "0x10",
                    "logs": [
                        sample_log("0x10", "0x1"),
                        sample_log("0x10", "0x2"),
                    ]
                }]
            elif method == "eth_getBlockByNumber":
                result = {"timestamp": "0x10", "transactions": ["0x" + "b" * 64]}
            else:
                raise AssertionError(f"Unexpected RPC method: {method}")
            return Response({"jsonrpc": "2.0", "id": json["id"], "result": result})

    client = BscRpcClient(session=Session(), sleep=lambda _: None)
    rows = client._fetch_block_range(16, 16)
    assert len(rows) == 2
    assert {row["event_index"] for row in rows} == {1, 2}


def test_single_dense_block_rejects_incomplete_receipt_fallback(monkeypatch):
    import src.bsc_rpc_whale_collector as collector

    monkeypatch.setattr(collector, "MAX_LOGS_PER_INTERVAL", 1)

    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def post(self, url, json, timeout):
            method = json["method"]
            if method == "eth_getLogs":
                result = [sample_log("0x10", "0x1")]
            elif method == "eth_getBlockReceipts":
                result = []
            elif method == "eth_getBlockByNumber":
                result = {"timestamp": "0x10", "transactions": ["0xtx"]}
            else:
                raise AssertionError(f"Unexpected RPC method: {method}")
            return Response({"jsonrpc": "2.0", "id": json["id"], "result": result})

    client = BscRpcClient(session=Session(), sleep=lambda _: None)
    try:
        client._fetch_block_range(16, 16)
    except RuntimeError as exc:
        assert "receipt coverage mismatch" in str(exc)
        assert "No potentially truncated block data was uploaded" in str(exc)
    else:
        raise AssertionError("Expected incomplete receipt fallback to be rejected")



def test_single_block_eth_getlogs_error_uses_complete_transaction_receipts(monkeypatch):
    import src.bsc_rpc_whale_collector as collector

    monkeypatch.setattr(collector, "MAX_LOGS_PER_INTERVAL", 1)

    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def post(self, url, json, timeout):
            if isinstance(json, list):
                return Response([
                    {
                        "jsonrpc": "2.0",
                        "id": item["id"],
                        "result": {
                            "transactionHash": "0x" + "b" * 64,
                            "blockNumber": "0x10",
                            "logs": [
                                sample_log("0x10", "0x1"),
                                sample_log("0x10", "0x2"),
                            ]
                        },
                    }
                    for item in json
                ])
            method = json["method"]
            if method == "eth_getLogs":
                return Response({
                    "jsonrpc": "2.0", "id": json["id"],
                    "error": {"code": -32005, "message": "query returned more than 2000 results"},
                })
            if method == "eth_getBlockReceipts":
                return Response({
                    "jsonrpc": "2.0", "id": json["id"],
                    "error": {"code": -32601, "message": "method not found"},
                })
            if method == "eth_getBlockByNumber":
                return Response({
                    "jsonrpc": "2.0", "id": json["id"],
                    "result": {"timestamp": "0x10", "transactions": ["0x" + "b" * 64]},
                })
            raise AssertionError(f"Unexpected RPC method: {method}")

    client = BscRpcClient(session=Session(), sleep=lambda _: None)
    rows = client._fetch_block_range(16, 16)
    assert len(rows) == 2
    assert {row["event_index"] for row in rows} == {1, 2}
