from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest
from src import news_candle_collector as collector


def _candle_frame():
    closed_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)
    open_time = int(closed_day.timestamp() * 1000)
    return pd.DataFrame([{
        "open_time": open_time, "open": 1.0, "high": 1.2, "low": 0.9,
        "close": 1.1, "volume": 100.0, "close_time": open_time + 86_399_999,
        "quote_volume": 110.0, "trade_count": 10,
        "taker_buy_base_volume": 55.0, "taker_buy_quote_volume": 60.5,
        "ignore": 0,
    }])


def test_recent_candle_collector_writes_symbol_daily_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "get_json", lambda path: {"symbols": []})
    monkeypatch.setattr(collector, "discover_target_symbols", lambda info: ["ABCUSDT", "XYZUSDT"])
    monkeypatch.setattr(collector, "fetch_klines", lambda *args, **kwargs: _candle_frame())
    output = tmp_path / "daily_candles.csv"
    summary = collector.collect_recent_daily_candles(output, days=14, pause_seconds=0)
    result = pd.read_csv(output)
    assert len(result) == 2
    assert set(result.symbol) == {"ABCUSDT", "XYZUSDT"}
    assert summary["candle_rows"] == 2
    assert (tmp_path / "news_candle_collection_summary.json").exists()


def test_recent_candle_collector_refuses_empty_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "get_json", lambda path: {"symbols": []})
    monkeypatch.setattr(collector, "discover_target_symbols", lambda info: ["ABCUSDT"])
    monkeypatch.setattr(collector, "fetch_klines", lambda *args, **kwargs: pd.DataFrame(columns=collector.COLS))
    with pytest.raises(RuntimeError, match="zero rows"):
        collector.collect_recent_daily_candles(tmp_path / "daily.csv", days=7, pause_seconds=0)


def test_recent_candle_collector_validates_days(tmp_path):
    with pytest.raises(ValueError, match="days"):
        collector.collect_recent_daily_candles(tmp_path / "daily.csv", days=1)
