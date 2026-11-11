import unittest
import pandas as pd

from src.surge_lifecycle_analysis import build_episodes, summarize_lifecycle


class SurgeLifecycleTests(unittest.TestCase):
    def test_clusters_nearby_triggers_and_measures_runup_and_drawdown(self):
        dates = pd.date_range("2025-01-01", periods=50, freq="D", tz="UTC")
        candles = pd.DataFrame({
            "symbol": ["ABCUSDT"] * len(dates),
            "open_time": (dates.astype("int64") // 10**6),
            "high": [100 + i for i in range(50)],
            "low": [95 + i for i in range(50)],
            "close": [98 + i for i in range(50)],
        })
        candles.loc[3, "high"] = 180
        candles.loc[4, "low"] = 120
        events = pd.DataFrame({
            "symbol": ["ABCUSDT"] * 3,
            "event_date_utc": ["2025-01-02", "2025-01-05", "2025-01-20"],
            "trigger_types": ["daily_high", "three_day_close", "daily_close"],
        })
        episodes = build_episodes(events, candles)
        self.assertEqual(len(episodes), 2)
        first = episodes.iloc[0]
        self.assertEqual(first.trigger_days, 2)
        self.assertEqual(first.episode_start.strftime("%Y-%m-%d"), "2025-01-02")
        self.assertGreater(first["during_7d_max_high_pct"], 0)
        self.assertIn("after_peak_7d_max_drawdown_pct", episodes.columns)
        report = summarize_lifecycle(episodes)
        self.assertEqual(report["episode_count"], 2)

    def test_rejects_missing_required_columns(self):
        with self.assertRaises(ValueError):
            build_episodes(pd.DataFrame({"symbol": ["X"]}), pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
