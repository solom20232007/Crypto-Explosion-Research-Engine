from pathlib import Path

from src.binance_history_collector import read_symbols_file


def test_read_symbols_file_deduplicates_and_ignores_comments(tmp_path: Path):
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("# historical symbols\nBTCUSDT\nethusdt\nBTCUSDT\n\n", encoding="utf-8")
    assert read_symbols_file(symbols_file) == ["BTCUSDT", "ETHUSDT"]
