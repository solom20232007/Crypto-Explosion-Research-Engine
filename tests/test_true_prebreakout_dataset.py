import pandas as pd

from src.true_prebreakout_dataset import label_future


def make_klines(anchor, highs, lows, closes):
    rows = []
    for i, (high, low, close) in enumerate(zip(highs, lows, closes)):
        rows.append({
            "open_time": anchor + i * 3_600_000,
            "high": high,
            "low": low,
            "close": close,
        })
    # One prior candle establishes the anchor close.
    rows.append({
        "open_time": anchor - 3_600_000,
        "high": 100.5,
        "low": 99.5,
        "close": 100.0,
    })
    return pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)


def test_labels_pump_like_reversal():
    anchor = 1_700_000_000_000
    highs = [100, 101, 103, 108, 112] + [104] * 19 + [103] * 24
    lows = [99, 100, 101, 104, 102] + [101] * 19 + [100] * 24
    closes = [100, 100, 102, 106, 103] + [102] * 19 + [101] * 24
    result = label_future(make_klines(anchor, highs, lows, closes), anchor)
    assert result["label"] == "pump_like_reversal"
    assert result["target_10pct_24h"] is True


def test_labels_no_breakout():
    anchor = 1_700_000_000_000
    highs = [101] * 72
    lows = [99] * 72
    closes = [100] * 72
    result = label_future(make_klines(anchor, highs, lows, closes), anchor)
    assert result["label"] == "no_10pct_breakout"


def test_labels_sustained_breakout():
    anchor = 1_700_000_000_000
    highs = [100 + i * 0.3 for i in range(72)]
    lows = [99 + i * 0.3 for i in range(72)]
    closes = [100 + i * 0.3 for i in range(72)]
    result = label_future(make_klines(anchor, highs, lows, closes), anchor)
    assert result["label"] == "sustained_breakout"
