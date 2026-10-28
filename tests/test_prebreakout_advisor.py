import pandas as pd
from src.prebreakout_advisor import add_cross_sectional_targets


def test_top_gainer_label_is_cross_sectional_and_future_only():
    rows = []
    for symbol_index in range(10):
        symbol = f"S{symbol_index}USDT"
        for hour in range(4):
            close = 100.0
            high = 100.0
            if symbol_index == 0 and hour == 1:
                high = 150.0
            rows.append({
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=hour),
                "symbol": symbol,
                "close": close,
                "high": high,
            })
    result = add_cross_sectional_targets(pd.DataFrame(rows), horizon=2, top_fraction=0.1, min_symbols=10)
    target = result[result["timestamp"] == pd.Timestamp("2026-01-01T00:00:00Z")]
    assert target["target_top_gainer_2h"].notna().sum() == 10
    assert int(target.loc[target["symbol"] == "S0USDT", "target_top_gainer_2h"].iloc[0]) == 1
    assert result.groupby("symbol")["target_top_gainer_2h"].apply(lambda s: s.tail(2).isna().all()).all()


def test_insufficient_cross_section_stays_unlabelled():
    rows = []
    for symbol in ["AAAUSDT", "BBBUSDT"]:
        for hour in range(3):
            rows.append({
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=hour),
                "symbol": symbol,
                "close": 1.0,
                "high": 1.0,
            })
    result = add_cross_sectional_targets(pd.DataFrame(rows), horizon=1, min_symbols=10)
    assert result["target_top_gainer_1h"].isna().all()


def test_gap_in_hourly_candles_leaves_horizon_unlabelled():
    rows = []
    timestamps = [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T03:00:00Z"),  # missing 02:00 candle
        pd.Timestamp("2026-01-01T04:00:00Z"),
    ]
    for symbol_index in range(10):
        for pos, timestamp in enumerate(timestamps):
            rows.append({
                "timestamp": timestamp,
                "symbol": f"S{symbol_index}USDT",
                "close": 100.0,
                "high": 150.0 if symbol_index == 0 and pos == 2 else 100.0,
            })

    result = add_cross_sectional_targets(
        pd.DataFrame(rows), horizon=2, top_fraction=0.1, min_symbols=10
    )
    at_midnight = result[result["timestamp"] == timestamps[0]]
    assert at_midnight["future_max_return_2h"].isna().all()
    assert at_midnight["target_top_gainer_2h"].isna().all()


def test_horizon_uses_elapsed_hourly_timestamps_not_row_count():
    rows = []
    timestamps = [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T02:00:00Z"),
        pd.Timestamp("2026-01-01T03:00:00Z"),
    ]
    for symbol_index in range(10):
        for pos, timestamp in enumerate(timestamps):
            rows.append({
                "timestamp": timestamp,
                "symbol": f"S{symbol_index}USDT",
                "close": 100.0,
                "high": 150.0 if symbol_index == 0 and pos == 2 else 100.0,
            })

    result = add_cross_sectional_targets(
        pd.DataFrame(rows), horizon=2, top_fraction=0.1, min_symbols=10
    )
    at_midnight = result[result["timestamp"] == timestamps[0]]
    assert at_midnight["future_max_return_2h"].notna().all()
    assert int(
        at_midnight.loc[
            at_midnight["symbol"] == "S0USDT", "target_top_gainer_2h"
        ].iloc[0]
    ) == 1

