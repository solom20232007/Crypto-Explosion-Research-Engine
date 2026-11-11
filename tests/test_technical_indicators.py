import numpy as np
import pandas as pd

from src.technical_indicators import MODEL_TECHNICAL_COLUMNS, add_technical_indicators
from src.analysis import build_features


def _candles(count=260):
    idx = np.arange(count, dtype=float)
    close = 10 + idx * 0.005 + np.sin(idx / 5) * 0.12
    return pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC"),
        "open": close * (1 - 0.001),
        "high": close * 1.008,
        "low": close * 0.992,
        "close": close,
        "volume": 1000 + idx * 2 + (idx % 7) * 20,
        "quote_volume": (1000 + idx * 2 + (idx % 7) * 20) * close,
    })


def test_indicator_engine_emits_model_columns_and_phase():
    result = add_technical_indicators(_candles())
    assert set(MODEL_TECHNICAL_COLUMNS).issubset(result.columns)
    assert {"prebreakout_score", "early_breakout_score", "breakout_phase"}.issubset(result.columns)
    assert set(result["breakout_phase"].dropna().unique()).issubset({
        "normal", "prebreakout_setup", "early_breakout", "active_explosion"
    })
    assert result["prebreakout_score"].dropna().between(0, 1).all()
    assert result["early_breakout_score"].dropna().between(0, 1).all()


def test_technical_indicators_do_not_use_future_candles():
    frame = _candles(260)
    baseline = add_technical_indicators(frame)
    changed = frame.copy()
    changed.loc[200:, ["open", "high", "low", "close", "volume", "quote_volume"]] *= 4
    modified = add_technical_indicators(changed)
    cols = [c for c in MODEL_TECHNICAL_COLUMNS if c in baseline.columns]
    pd.testing.assert_frame_equal(
        baseline.loc[:199, cols].reset_index(drop=True),
        modified.loc[:199, cols].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_analysis_pipeline_includes_indicator_columns():
    result = build_features(_candles(260), "TESTUSDT")
    assert set(MODEL_TECHNICAL_COLUMNS).issubset(result.columns)
    assert "breakout_phase" in result.columns
