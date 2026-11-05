from src import onchain_backfill


class FakeStore:
    def __init__(self):
        self.row = None
        self.history = []

    def get(self, provider, chain):
        return self.row

    def advance(self, **kwargs):
        self.history.append(kwargs.copy())
        self.row = {
            "provider": kwargs["provider"],
            "chain": kwargs["chain"],
            "cursor_kind": kwargs["cursor_kind"],
            "cursor_value": kwargs["cursor_value"],
            "initial_cursor_value": kwargs.get("initial_cursor_value", kwargs["cursor_value"]),
            "last_completed_from": kwargs.get("completed_from"),
            "last_completed_to": kwargs.get("completed_to"),
            "status": kwargs.get("status", "ready"),
        }
        return self.row


class FakeRoutescan:
    def __init__(self, fail=False, fail_on_call=None, budget_fail=False):
        self.calls = []
        self.fail = fail
        self.fail_on_call = fail_on_call
        self.budget_fail = budget_fail

    def _fetch_interval(self, chain, chain_id, start, end, max_pages, budget, max_requests,
                        on_interval_complete=None, newest_first=True):
        self.calls.append((int(start.timestamp()), int(end.timestamp())))
        budget[0] += 1
        if self.budget_fail:
            raise RuntimeError("Routescan safety cap of 1 requests per chain reached")
        if self.fail or self.fail_on_call == len(self.calls):
            raise RuntimeError("simulated request cap")
        if on_interval_complete:
            on_interval_complete(start, end, [])
        return []


class FakeBsc:
    def __init__(self):
        self.request_count = 0
        self.ranges = []

    @staticmethod
    def _hex_int(value):
        return int(str(value), 16) if isinstance(value, str) and value.startswith("0x") else int(value)

    def _rpc(self, method, params):
        self.request_count += 1
        assert method == "eth_blockNumber"
        return hex(1000)

    def _block(self, number):
        self.request_count += 1
        return {"timestamp": hex(1000 + number * 3)}

    def _fetch_block_range(self, start, end, on_interval_complete=None, newest_first=True):
        self.ranges.append((start, end))
        self.request_count += 1
        if on_interval_complete:
            on_interval_complete(start, end, [])
        return []


def test_routescan_backfill_advances_checkpoint_even_for_empty_completed_window(monkeypatch):
    store = FakeStore()
    client = FakeRoutescan()
    monkeypatch.setattr(onchain_backfill, "upload_transfers", lambda rows: len(rows))
    result = onchain_backfill.run_routescan_backfill(
        "1970-01-12T13:46:40Z", "1970-01-12T13:56:40Z",
        chunk_minutes=5, max_chunks=1, store=store, client=client,
    )
    assert result["chunks_completed_this_run"] == 1
    assert store.row["cursor_value"] == 1000300
    assert store.row["last_completed_from"] == 1000000
    assert client.calls == [(1000000, 1000300)]


def test_routescan_backfill_does_not_advance_failed_interval(monkeypatch):
    store = FakeStore()
    client = FakeRoutescan(fail=True)
    monkeypatch.setattr(onchain_backfill, "upload_transfers", lambda rows: len(rows))
    try:
        onchain_backfill.run_routescan_backfill(
            "1970-01-12T13:46:40Z", "1970-01-12T13:56:40Z",
            chunk_minutes=5, max_chunks=1, store=store, client=client,
        )
    except RuntimeError as exc:
        assert "simulated request cap" in str(exc)
    else:
        raise AssertionError("Expected simulated failure")
    assert store.row["cursor_value"] == 1000000
    assert store.row["status"] == "error"


def test_bsc_backfill_advances_in_block_ranges_even_when_empty(monkeypatch):
    store = FakeStore()
    client = FakeBsc()
    monkeypatch.setattr(onchain_backfill, "upload_transfers", lambda rows: len(rows))
    result = onchain_backfill.run_bsc_backfill(
        "1970-01-01T00:16:40Z", "1970-01-01T00:26:40Z",
        chunk_blocks=100, max_chunks=1, store=store, client=client,
    )
    assert result["chunks_completed_this_run"] == 1
    assert client.ranges == [(0, 99)]
    assert store.row["cursor_value"] == 100
    assert store.row["last_completed_from"] == 0
    assert store.row["last_completed_to"] == 99


def test_routescan_resume_preserves_last_completed_chunk_after_later_failure(monkeypatch):
    store = FakeStore()
    client = FakeRoutescan(fail_on_call=2)
    monkeypatch.setattr(onchain_backfill, "upload_transfers", lambda rows: len(rows))
    try:
        onchain_backfill.run_routescan_backfill(
            "1970-01-12T13:46:40Z", "1970-01-12T13:56:40Z",
            chunk_minutes=5, max_chunks=2, store=store, client=client,
        )
    except RuntimeError as exc:
        assert "simulated request cap" in str(exc)
    else:
        raise AssertionError("Expected second chunk to fail")
    assert store.row["cursor_value"] == 1000300
    assert store.row["last_completed_from"] == 1000000
    assert store.row["last_completed_to"] == 1000300
    assert store.row["status"] == "error"


def test_routescan_budget_cap_is_a_resumable_pause_not_a_failed_run(monkeypatch):
    store = FakeStore()
    client = FakeRoutescan(budget_fail=True)
    monkeypatch.setattr(onchain_backfill, "upload_transfers", lambda rows: len(rows))
    result = onchain_backfill.run_routescan_backfill(
        "1970-01-12T13:46:40Z", "1970-01-12T13:56:40Z",
        chunk_minutes=5, max_chunks=1, store=store, client=client,
    )
    assert result["status"] == "paused"
    assert "safety cap" in result["pause_reason"]
    assert store.row["cursor_value"] == 1000000
    assert store.row["status"] == "paused"
