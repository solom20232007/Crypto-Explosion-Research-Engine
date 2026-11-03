import unittest
import pandas as pd

from src.hourly_episode_refinement import refine_episode, HOUR_MS


class HourlyEpisodeRefinementTests(unittest.TestCase):
    def test_finds_forward_threshold_without_using_signal_candle_high(self):
        # Signal close is 100; the next 24 highs rise to 111.
        rows = []
        start = pd.Timestamp("2026-01-10T00:00:00Z")
        for i in range(145):
            close = 100.0 if i == 0 else 100.0
            high = 100.0
            if 1 <= i <= 24:
                high = 111.0
            rows.append({
                "open_time": int(start.timestamp() * 1000) + i * HOUR_MS,
                "close": close, "high": high, "low": 99.0
            })
        frame = pd.DataFrame(rows)
        episode = {"symbol": "TESTUSDT", "episode_start": "2026-01-10T00:00:00Z",
                   "episode_id": "TESTUSDT_20260110", "episode_tier": "early_surge_only",
                   "trigger_days": 1}
        result = refine_episode(episode, frame)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["early_10pct_24h_future_max_high_pct"], 11.0)
        self.assertIsNotNone(result["early_10pct_24h_signal_time_utc"])

    def test_requires_complete_contiguous_forward_window(self):
        rows = []
        start = pd.Timestamp("2026-01-10T00:00:00Z")
        for i in range(145):
            rows.append({"open_time": int(start.timestamp() * 1000) + i * 2 * HOUR_MS,
                         "close": 100.0, "high": 120.0, "low": 99.0})
        result = refine_episode(
            {"symbol": "TESTUSDT", "episode_start": "2026-01-10T00:00:00Z"},
            pd.DataFrame(rows))
        self.assertIsNone(result["confirmed_15pct_24h_signal_time_utc"])


if __name__ == "__main__":
    unittest.main()
