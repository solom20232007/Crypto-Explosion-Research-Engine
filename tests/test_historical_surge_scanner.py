import unittest
import pandas as pd
from src.historical_surge_scanner import detect_surge_events, discover_target_symbols


class HistoricalSurgeScannerTests(unittest.TestCase):
    def test_excludes_major_assets_stables_and_leveraged_tokens(self):
        payload={"symbols":[
          {"symbol":"BTCUSDT","baseAsset":"BTC","quoteAsset":"USDT","status":"TRADING"},
          {"symbol":"ETHUSDT","baseAsset":"ETH","quoteAsset":"USDT","status":"TRADING"},
          {"symbol":"SOLUSDT","baseAsset":"SOL","quoteAsset":"USDT","status":"TRADING"},
          {"symbol":"USDCUSDT","baseAsset":"USDC","quoteAsset":"USDT","status":"TRADING"},
          {"symbol":"XYZUPUSDT","baseAsset":"XYZUP","quoteAsset":"USDT","status":"TRADING"},
          {"symbol":"PEPEUSDT","baseAsset":"PEPE","quoteAsset":"USDT","status":"TRADING"},
          {"symbol":"ABCUSDT","baseAsset":"ABC","quoteAsset":"USDT","status":"BREAK"}]}
        self.assertEqual(discover_target_symbols(payload), ["PEPEUSDT"])

    def test_detects_surge_events(self):
        frame=pd.DataFrame({"open_time":[i*86400000 for i in range(10)],"open":[10]*10,
          "high":[10,10,10,10,10,10,10,13,14,15],"low":[9]*10,
          "close":[10,10,10,10,10,10,10,12,13,14],"volume":[100]*10,"quote_volume":[200000]*10})
        events=detect_surge_events(frame,"PEPEUSDT")
        self.assertFalse(events.empty)
        self.assertEqual(set(events.symbol), {"PEPEUSDT"})
        self.assertTrue(events.event_date_utc.is_monotonic_increasing)
        self.assertTrue((events.daily_high_change_pct>=25).any())


if __name__ == "__main__":
    unittest.main()
