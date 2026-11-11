from src.onchain_checkpoint import SupabaseCheckpointStore


class Response:
    def __init__(self, payload, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self.payload


class Session:
    def __init__(self):
        self.get_response = Response([])
        self.post_response = Response([{"provider": "routescan_v2", "chain": "ethereum", "cursor_value": 100}])
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.get_response

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self.post_response


def test_checkpoint_get_returns_none_when_not_initialized():
    session = Session()
    store = SupabaseCheckpointStore("https://example.supabase.co", "secret", session)
    assert store.get("routescan_v2", "ethereum") is None
    assert "onchain_collection_checkpoints" in session.calls[0][1]


def test_checkpoint_advance_uses_atomic_database_rpc_and_integer_cursor():
    session = Session()
    store = SupabaseCheckpointStore("https://example.supabase.co", "secret", session)
    result = store.advance(
        provider="routescan_v2", chain="ethereum", cursor_kind="unix_seconds",
        cursor_value=100, initial_cursor_value=0, completed_from=50, completed_to=100,
    )
    assert result["cursor_value"] == 100
    method, url, kwargs = session.calls[0]
    assert method == "post"
    assert url.endswith("/rpc/advance_onchain_collection_checkpoint")
    assert kwargs["json"]["p_cursor_value"] == 100


def test_checkpoint_rejects_invalid_cursor_kind_before_network_call():
    session = Session()
    store = SupabaseCheckpointStore("https://example.supabase.co", "secret", session)
    try:
        store.advance(provider="p", chain="c", cursor_kind="timestamp", cursor_value=1)
    except ValueError as exc:
        assert "cursor_kind" in str(exc)
    else:
        raise AssertionError("Expected invalid cursor kind to fail")
    assert session.calls == []
