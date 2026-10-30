import unittest

import pandas as pd

from src.audit_prebreakout_data import audit


class DataQualityAuditTests(unittest.TestCase):
    def test_flags_duplicates_gaps_invalid_ohlc_and_forbidden_symbols(self):
        candles = pd.DataFrame([
            {"symbol":"ABCUSDT","open_time":0,"open":10,"high":11,"low":9,"close":10,"volume":100,"quote_volume":1000},
            {"symbol":"ABCUSDT","open_time":172800000,"open":10,"high":9,"low":8,"close":9,"volume":100,"quote_volume":900},
            {"symbol":"ABCUSDT","open_time":172800000,"open":10,"high":12,"low":8,"close":11,"volume":100,"quote_volume":1100},
            {"symbol":"BTCUSDT","open_time":0,"open":10,"high":11,"low":9,"close":10,"volume":100,"quote_volume":1000},
        ])
        dataset = pd.DataFrame([
            {"symbol":"ABCUSDT","split":"train","date_utc":"2024-01-01","explosion_1d":0,"explosion_3d":1,"explosion_7d":0},
            {"symbol":"BTCUSDT","split":"test","date_utc":"2026-06-01","explosion_1d":1,"explosion_3d":0,"explosion_7d":1},
        ])
        result = audit(candles, dataset)
        self.assertEqual(result["invalid_ohlc_rows"], 1)
        self.assertEqual(result["duplicate_symbol_open_time_rows"], 2)
        self.assertEqual(result["forbidden_candidate_symbols_present"], ["BTCUSDT"])
        self.assertGreaterEqual(result["symbols_with_gaps_over_1d"], 1)
        self.assertTrue(all(result["label_values_binary_or_missing"].values()))

    def test_rejects_missing_required_candle_columns(self):
        with self.assertRaises(ValueError):
            audit(pd.DataFrame([{"symbol":"ABCUSDT"}]), pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
