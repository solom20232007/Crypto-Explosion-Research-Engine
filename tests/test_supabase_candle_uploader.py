from pathlib import Path

import pytest

from src.supabase_candle_uploader import discover_files


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("open_time\n")


def test_discovers_all_csv_files_without_symbols_file(tmp_path):
    raw = tmp_path / "raw"
    _touch(raw, "ZECUSDT_1h.csv", "PEPEUSDT_1h.csv")

    files = discover_files(raw, tmp_path / "missing.txt")

    assert [p.name for p in files] == ["PEPEUSDT_1h.csv", "ZECUSDT_1h.csv"]


def test_symbols_file_selects_only_listed_symbols(tmp_path):
    raw = tmp_path / "raw"
    _touch(raw, "ZECUSDT_1h.csv", "PEPEUSDT_1h.csv", "ADAUSDT_1h.csv")
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("zecusdt\nPEPEUSDT\n\n")

    files = discover_files(raw, symbols)

    assert [p.name for p in files] == ["ZECUSDT_1h.csv", "PEPEUSDT_1h.csv"]


def test_missing_symbol_is_a_warning_not_a_failure(tmp_path, capsys):
    raw = tmp_path / "raw"
    _touch(raw, "ZECUSDT_1h.csv")
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("ZECUSDT\nNEWUSDT\n")

    files = discover_files(raw, symbols)

    assert [p.name for p in files] == ["ZECUSDT_1h.csv"]
    assert "NEWUSDT_1h.csv" in capsys.readouterr().out


def test_raises_when_no_output_exists(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("NEWUSDT\n")

    with pytest.raises(FileNotFoundError):
        discover_files(raw, symbols)
