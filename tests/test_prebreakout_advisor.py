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
