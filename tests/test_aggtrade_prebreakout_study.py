import unittest
import pandas as pd

from src.aggtrade_prebreakout_study import extract_features, _event_anchor


class AggtradePrebreakoutStudyTests(unittest.TestCase):
    def test_aggressor_side_and_windows(self):
        anchor = 1_800_000_000_000
        trades = pd.DataFrame([
            {"timestamp": anchor - 60_000, "price": 2.0, "quantity": 100.0,
             "is_buyer_maker": False, "agg_trade_id": 1},
            {"timestamp": anchor - 2 * 60_000, "price": 2.0, "quantity": 50.0,
             "is_buyer_maker": True, "agg_trade_id": 2},
            {"timestamp": anchor - 2 * HOUR_MS, "price": 2.0, "quantity": 10.0,
             "is_buyer_maker": False, "agg_trade_id": 3},
        ])
        trades["quote"] = trades.price * trades.quantity
        result = extract_features(trades, anchor)
        self.assertEqual(result["h1_aggressive_buy_usdt"], 200.0)
        self.assertEqual(result["h1_aggressive_sell_usdt"], 100.0)
        self.assertAlmostEqual(result["h1_buy_ratio"], 2 / 3, places=4)
        self.assertEqual(result["h6_trade_count"], 3)

    def test_anchor_prefers_earliest_confirmed_timestamp(self):
        row = {
            "confirmed_15pct_24h_signal_time_utc": "2026-01-02T10:00:00Z",
            "confirmed_25pct_48h_signal_time_utc": "2026-01-02T08:00:00Z",
            "confirmed_40pct_72h_signal_time_utc": None,
            "early_10pct_24h_signal_time_utc": "2026-01-02T01:00:00Z",
        }
        anchor, tier = _event_anchor(row)
        self.assertEqual(anchor, "2026-01-02T08:00:00Z")
        self.assertEqual(tier, "confirmed")


HOUR_MS = 3_600_000


if __name__ == "__main__":
    unittest.main()
