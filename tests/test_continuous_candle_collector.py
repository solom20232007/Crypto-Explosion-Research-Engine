from datetime import datetime, timezone
import pandas as pd
import pytest

from src import continuous_candle_collector as collector


def _frame(now):
    closed_open = int((pd.Timestamp(now).floor("h") - pd.Timedelta(hours=1)).timestamp() * 1000)
    return pd.DataFrame([{
        "open_time": closed_open, "open": 10, "high": 12, "low": 9, "close": 11,
        "volume": 100, "close_time": closed_open + 3_599_999, "quote_volume": 1100,
        "trade_count": 42, "taker_buy_base_volume": 55, "taker_buy_quote_volume": 600,
        "ignore": 0,
    }])


def test_hourly_collector_upserts_closed_candles_and_writes_summary(tmp_path, monkeypatch):
    now = datetime(2026, 10, 31, 12, 17, tzinfo=timezone.utc)
    monkeypatch.setattr(collector, "get_json", lambda path: {
        "symbols": [
            {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ABCUSDT", "baseAsset": "ABC", "quoteAsset": "USDT", "status": "TRADING"},
        ]
    })
    monkeypatch.setattr(collector, "discover_target_symbols", lambda info: ["ABCUSDT"])
    monkeypatch.setattr(collector, "fetch_klines", lambda *args, **kwargs: _frame(now))
    writes = []
    monkeypatch.setattr(collector, "_supabase_request", lambda *args, **kwargs: writes.append(kwargs["rows"]) or [])
    result = collector.collect_hourly_candles(
        supabase_url="https://example.supabase.co", secret_key="secret",
        now=now, upload=True, pause_seconds=0, output_dir=tmp_path,
    )
    assert result["candle_rows_fetched"] == 2
    assert result["supabase_upserted"] is True
    assert sum(len(batch) for batch in writes) == 2
    assert {row["symbol"] for batch in writes for row in batch} == {"BTCUSDT", "ABCUSDT"}
    assert all(row["timeframe"] == "1h" for batch in writes for row in batch)
    assert (tmp_path / "collection_summary.json").exists()


def test_hourly_collector_refuses_empty_results(monkeypatch):
    monkeypatch.setattr(collector, "get_json", lambda path: {"symbols": []})
    monkeypatch.setattr(collector, "discover_target_symbols", lambda info: ["ABCUSDT"])
    monkeypatch.setattr(collector, "fetch_klines", lambda *args, **kwargs: pd.DataFrame(columns=collector.COLS))
    with pytest.raises(RuntimeError, match="zero rows"):
        collector.collect_hourly_candles(upload=False, pause_seconds=0)


def test_hourly_collector_requires_supabase_credentials(monkeypatch):
    now = datetime(2026, 10, 31, 12, 17, tzinfo=timezone.utc)
    monkeypatch.setattr(collector, "get_json", lambda path: {"symbols": []})
    monkeypatch.setattr(collector, "discover_target_symbols", lambda info: ["ABCUSDT"])
    monkeypatch.setattr(collector, "fetch_klines", lambda *args, **kwargs: _frame(now))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        collector.collect_hourly_candles(now=now, upload=True, pause_seconds=0)
