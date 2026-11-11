import pandas as pd

from src.backtest import _metrics


def test_metrics_count_true_false_positives_and_lift():
    frame = pd.DataFrame({"target_future_15pct_24h": [1.0, 0.0, 1.0, 0.0]})
    pred = pd.Series([True, True, False, False])
    result = _metrics(frame, pred, "test")
    assert result["signals"] == 2
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["lift_over_baseline"] == 1.0


def test_metrics_handle_no_signals():
    frame = pd.DataFrame({"target_future_15pct_24h": [0.0, 1.0]})
    result = _metrics(frame, pd.Series([False, False]), "none")
    assert result["signals"] == 0
    assert result["precision"] is None
    assert result["recall"] == 0.0
