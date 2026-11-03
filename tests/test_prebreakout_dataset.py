import unittest
import pandas as pd
from src.prebreakout_dataset import build_dataset, summarize


class PrebreakoutDatasetTests(unittest.TestCase):
    def candles(self, n=45):
        return pd.DataFrame({
            "symbol": ["XYZUSDT"] * n,
            "open_time": [i * 86400000 for i in range(n)],
            "open": [10.0] * n,
            "high": [10.5] * n,
            "low": [9.5] * n,
            "close": [10.0] * n,
            "volume": [100.0] * n,
            "quote_volume": [10000.0] * n,
        })

    def test_labels_only_look_forward_after_signal_close(self):
        frame = self.candles(12)
        frame.loc[5, "high"] = 20.0  # same-day spike must not label row 5
        frame.loc[6, "high"] = 13.0  # +30% after row 5 close
        data = build_dataset(frame)
        row5 = data.loc[data.open_time == 5 * 86400000].iloc[0]
        self.assertEqual(row5["explosion_1d"], 1)
        row6 = data.loc[data.open_time == 6 * 86400000].iloc[0]
        self.assertEqual(row6["explosion_1d"], 0)
        self.assertTrue(pd.isna(data.iloc[-1]["explosion_1d"]))

    def test_chronological_splits_and_summary(self):
        data = build_dataset(self.candles(1000))
        self.assertEqual(data.loc[data.date_utc < "2025-10-28", "split"].unique().tolist(), ["train"])
        report = summarize(data)
        self.assertEqual(report["symbols"], 1)
        self.assertIn("train", report["chronological_splits"])

    def test_technical_and_liquidity_features_are_added(self):
        data = build_dataset(self.candles(80))
        expected = {
            "rsi_14", "ema9_vs_ema21_pct", "macd_pct", "atr_14_pct",
            "bollinger_width_20_pct", "close_vs_prior_20d_high_pct",
            "quote_volume_zscore_30d", "trade_count_ratio_1d_7d",
            "taker_buy_quote_ratio", "macd_histogram_pct", "macd_bullish_cross", "ema9_21_bullish_cross", "bars_since_macd_bullish_cross", "rsi_cross_up_50", "day_of_week_sin", "month_cos", "days_since_first_seen", "ema_bullish_alignment",
        }
        self.assertTrue(expected.issubset(set(data.columns)))
        finite_rsi = data["rsi_14"].dropna()
        self.assertTrue(((finite_rsi >= 0) & (finite_rsi <= 100)).all())

    def test_crossover_flags_and_calendar_features_are_available(self):
        frame = self.candles(90)
        frame.loc[45:, "close"] = [10 + (i - 45) * 0.5 for i in range(45, 90)]
        frame.loc[45:, "open"] = frame.loc[45:, "close"] - 0.1
        frame.loc[45:, "high"] = frame.loc[45:, "close"] + 0.2
        frame.loc[45:, "low"] = frame.loc[45:, "close"] - 0.2
        data = build_dataset(frame)
        for col in ("macd_bullish_cross", "macd_bearish_cross", "ema9_21_bullish_cross",
                    "ema9_21_bearish_cross", "rsi_cross_up_30", "rsi_cross_up_50"):
            self.assertTrue(set(data[col].dropna().unique()).issubset({0, 1}))
        self.assertTrue((data["days_since_first_seen"] >= 0).all())
        self.assertTrue(data["day_of_week_sin"].between(-1, 1).all())

    def test_requires_required_columns(self):
        with self.assertRaises(ValueError):
            build_dataset(pd.DataFrame({"symbol": ["XYZUSDT"]}))


if __name__ == "__main__":
    unittest.main()
