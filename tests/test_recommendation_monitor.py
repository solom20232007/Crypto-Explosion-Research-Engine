import unittest

import numpy as np
import pandas as pd

from src.recommendation_monitor import build_prediction_ledger


class RecommendationMonitorTests(unittest.TestCase):
    def test_forward_outcomes_and_cooldown_are_recorded(self):
        timestamps = pd.date_range("2026-01-01", periods=32, freq="h", tz="UTC")
        close = np.full(32, 100.0)
        high = np.full(32, 101.0)
        low = np.full(32, 99.0)
        high[2] = 116.0
        close[2] = 110.0

        features = pd.DataFrame({
            "timestamp": timestamps,
            "symbol": "AAAUSDT",
            "close": close,
            "high": high,
            "low": low,
            "ret_1h": np.zeros(32),
        })
        predictions = features.iloc[:3].copy()
        predictions["actual"] = [1, 1, 0]
        predictions["probability"] = [0.9, 0.8, 0.1]
        predictions["selected_threshold"] = 0.5

        ledger = build_prediction_ledger(
            features, predictions, cooldown_bars=1, target_return=0.15
        )

        self.assertEqual(len(ledger), 3)
        self.assertTrue(bool(ledger.loc[0, "is_model_signal"]))
        self.assertTrue(bool(ledger.loc[0, "signal_emitted"]))
        self.assertTrue(bool(ledger.loc[1, "cooldown_suppressed"]))
        self.assertFalse(bool(ledger.loc[2, "is_model_signal"]))
        self.assertAlmostEqual(
            float(ledger.loc[0, "max_favorable_return_24h"]), 0.16
        )
        self.assertTrue(bool(ledger.loc[0, "target_hit_15pct_24h"]))
        self.assertEqual(int(ledger.loc[0, "bars_to_target_15pct_24h"]), 2)

    def test_missing_future_window_is_not_marked_negative(self):
        timestamps = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        features = pd.DataFrame({
            "timestamp": timestamps,
            "symbol": "AAAUSDT",
            "close": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
        })
        predictions = features.iloc[[3]].copy()
        predictions["actual"] = 0
        predictions["probability"] = 0.9
        predictions["selected_threshold"] = 0.5

        ledger = build_prediction_ledger(features, predictions, cooldown_bars=0)
        self.assertTrue(pd.isna(ledger.loc[0, "target_hit_15pct_24h"]))
        self.assertEqual(ledger.loc[0, "outcome_status"], "incomplete_forward_window")


if __name__ == "__main__":
    unittest.main()
