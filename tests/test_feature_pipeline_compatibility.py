import numpy as np
import pandas as pd

from src.analysis import build_features
from src.prebreakout_advisor import FEATURE_COLUMNS


def test_feature_builder_emits_every_prebreakout_model_feature():
    count = 120
    close = 10 + np.arange(count, dtype=float) * 0.02 + np.sin(np.arange(count) / 4)
    frame = pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC"),
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(count, 1000.0) + np.arange(count),
        "quote_volume": (np.full(count, 1000.0) + np.arange(count)) * close,
    })
    features = build_features(frame, "TESTUSDT")
    missing = sorted(set(FEATURE_COLUMNS) - set(features.columns))
    assert not missing, f"Missing advisor feature columns: {missing}"
    assert {"timestamp", "symbol", "close", "high"}.issubset(features.columns)
    assert features["timestamp"].is_monotonic_increasing


def test_feature_builder_parses_binance_epoch_milliseconds():
    count = 100
    close = 10 + np.arange(count, dtype=float) * 0.01
    times = pd.date_range("2025-03-01", periods=count, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "open_time": times.astype("int64") // 10**6,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(count, 1000.0),
        "quote_volume": np.full(count, 10000.0),
    })
    features = build_features(frame, "TESTUSDT")
    assert features["timestamp"].min() == times[0]
    assert features["timestamp"].max() == times[-1]
