import unittest

from src.orderbook_liquidity import calculate_metrics, eligible_symbols


class OrderbookLiquidityTests(unittest.TestCase):
    def test_excludes_major_coins_stables_and_leveraged(self):
        info = {"symbols": [
            {"symbol": "ABCUSDT", "baseAsset": "ABC", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "SOLUSDT", "baseAsset": "SOL", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ABCUPUSDT", "baseAsset": "ABCUP", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ABCUSDT", "baseAsset": "ABC", "quoteAsset": "USDT", "status": "BREAK"},
            {"symbol": "ABCUSDC", "baseAsset": "ABC", "quoteAsset": "USDC", "status": "TRADING"},
        ]}
        self.assertEqual(eligible_symbols(info), ["ABCUSDT"])

    def test_spread_depth_imbalance_and_slippage(self):
        book = {
            "bids": [["99", "20"], ["98.5", "20"]],
            "asks": [["101", "20"], ["101.5", "20"]],
        }
        row = calculate_metrics("TESTUSDT", book, "2026-10-29T00:00:00+00:00", notionals=(100, 10000))
        self.assertAlmostEqual(row["spread_bps"], 200, places=3)
        self.assertGreater(row["depth_bid_usdt_1pct"], 0)
        self.assertGreater(row["depth_ask_usdt_1pct"], 0)
        self.assertAlmostEqual(row["bid_ask_qty_imbalance"], 0, places=6)
        self.assertIsNotNone(row["buy_slippage_bps_100_usdt"])
        self.assertIsNone(row["buy_slippage_bps_10000_usdt"])

    def test_empty_or_crossed_book_rejected(self):
        with self.assertRaises(ValueError):
            calculate_metrics("TESTUSDT", {"bids": [], "asks": []}, "now")
        with self.assertRaises(ValueError):
            calculate_metrics("TESTUSDT", {"bids": [["101", "1"]], "asks": [["100", "1"]]}, "now")


if __name__ == "__main__":
    unittest.main()
