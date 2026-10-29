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

    def test_requires_required_columns(self):
        with self.assertRaises(ValueError):
            build_dataset(pd.DataFrame({"symbol": ["XYZUSDT"]}))


if __name__ == "__main__":
    unittest.main()
