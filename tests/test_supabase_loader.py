import os
import unittest
from unittest.mock import patch

from src import supabase_loader


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, rows):
        self._rows = rows

    def json(self):
        return self._rows


class FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get(self, url, headers, params, timeout):
        self.calls.append((url, dict(headers), dict(params), timeout))
        return FakeResponse(self.rows)


class SupabaseLoaderTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "open_time": "2026-01-01T00:00:00+00:00",
            "close_time": "2026-01-01T00:59:59+00:00",
            "open": "100",
            "high": "105",
            "low": "99",
            "close": "103",
            "volume": "10",
            "quote_volume": "1030",
            "trades": 100,
        }

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SECRET_KEY": "test-key",
    }, clear=True)
    def test_fetch_candles_reads_and_normalizes_rows(self):
        session = FakeSession([self.row])
        result = supabase_loader.fetch_candles("BTCUSDT", "1h", session=session)
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), supabase_loader.OUTPUT_COLUMNS)
        self.assertEqual(float(result.iloc[0]["close"]), 103.0)
        self.assertEqual(session.calls[0][2]["symbol"], "eq.BTCUSDT")
        self.assertEqual(session.calls[0][2]["timeframe"], "eq.1h")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_fails_without_network_call(self):
        with self.assertRaisesRegex(RuntimeError, "credentials missing"):
            supabase_loader.fetch_candles("BTCUSDT")

    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SECRET_KEY": "test-key",
    }, clear=True)
    def test_invalid_ohlc_is_rejected(self):
        row = dict(self.row, high="98")
        with self.assertRaisesRegex(RuntimeError, "Invalid high"):
            supabase_loader.fetch_candles(
                "BTCUSDT", "1h", session=FakeSession([row])
            )


if __name__ == "__main__":
    unittest.main()
