"""Historical recommendation ledger and post-signal outcome evaluation.

This module only evaluates saved research predictions. It does not trade,
send alerts, or make live predictions.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd


FEATURES_PATH = Path("data/features.csv")
PREDICTIONS_PATH = Path("reports/test_predictions.csv")
REPORT_DIR = Path("reports")
HORIZONS = (1, 4, 12, 24)
TARGET_RETURN = 0.15
DEFAULT_COOLDOWN_BARS = 24


def _as_utc(series):
    return pd.to_datetime(series, utc=True, errors="coerce")


def build_prediction_ledger(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS,
    target_return: float = TARGET_RETURN,
) -> pd.DataFrame:
    """Attach forward outcomes to every prediction row and mark emitted signals."""
    required_features = {"timestamp", "symbol", "close", "high", "low"}
    required_predictions = {
        "timestamp", "symbol", "close", "high", "low", "probability", "actual"
    }
    missing_features = sorted(required_features - set(features.columns))
    missing_predictions = sorted(required_predictions - set(predictions.columns))
    if missing_features:
        raise ValueError(f"features missing columns: {', '.join(missing_features)}")
    if missing_predictions:
        raise ValueError(
            f"predictions missing columns: {', '.join(missing_predictions)}"
        )
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars cannot be negative.")
    if not 0 < target_return < 5:
        raise ValueError("target_return must be greater than 0 and less than 5.")

    history = features.copy()
    history["timestamp"] = _as_utc(history["timestamp"])
    preds = predictions.copy()
    preds["timestamp"] = _as_utc(preds["timestamp"])

    for frame in (history, preds):
        frame["symbol"] = frame["symbol"].astype(str)
        for column in ("close", "high", "low"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    history = history.dropna(
        subset=["timestamp", "symbol", "close", "high", "low"]
    ).sort_values(["symbol", "timestamp"])
    preds = preds.dropna(
        subset=["timestamp", "symbol", "close", "high", "low", "probability", "actual"]
    ).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if preds.empty:
        raise ValueError("No usable rows found in reports/test_predictions.csv.")

    # Build a timestamp -> row-index lookup for each symbol. Outcomes use only
    # candles strictly after the prediction candle.
    symbol_data = {}
    for symbol, group in history.groupby("symbol", sort=False):
        group = group.reset_index(drop=True)
        symbol_data[symbol] = {
            "frame": group,
            "index": {timestamp: i for i, timestamp in enumerate(group["timestamp"])},
            "high": group["high"].to_numpy(dtype=float),
            "low": group["low"].to_numpy(dtype=float),
            "close": group["close"].to_numpy(dtype=float),
        }

    rows = []
    for _, prediction in preds.iterrows():
        row = prediction.to_dict()
        symbol = str(prediction["symbol"])
        data = symbol_data.get(symbol)
        index = data["index"].get(prediction["timestamp"]) if data else None

        if data is None or index is None:
            row["outcome_status"] = "missing_entry_candle"
            for horizon in HORIZONS:
                row[f"close_return_{horizon}h"] = np.nan
                row[f"max_high_return_{horizon}h"] = np.nan
                row[f"max_low_return_{horizon}h"] = np.nan
            row["max_favorable_return_24h"] = np.nan
            row["max_adverse_return_24h"] = np.nan
            row["target_hit_15pct_24h"] = np.nan
            row["bars_to_target_15pct_24h"] = np.nan
            rows.append(row)
            continue

        closes = data["close"]
        highs = data["high"]
        lows = data["low"]
        entry = float(closes[index])
        max_forward = min(24, len(closes) - index - 1)
        future_highs = highs[index + 1:index + 1 + max_forward]
        future_lows = lows[index + 1:index + 1 + max_forward]

        row["outcome_status"] = (
            "complete_24h" if max_forward >= 24 else "incomplete_forward_window"
        )
        for horizon in HORIZONS:
            end_index = index + horizon
            if end_index < len(closes):
                row[f"close_return_{horizon}h"] = (
                    float(closes[end_index] / entry - 1) if entry > 0 else np.nan
                )
            else:
                row[f"close_return_{horizon}h"] = np.nan

            window_high = highs[index + 1:min(index + horizon + 1, len(highs))]
            window_low = lows[index + 1:min(index + horizon + 1, len(lows))]
            row[f"max_high_return_{horizon}h"] = (
                float(np.max(window_high) / entry - 1)
                if entry > 0 and len(window_high) == horizon else np.nan
            )
            row[f"max_low_return_{horizon}h"] = (
                float(np.min(window_low) / entry - 1)
                if entry > 0 and len(window_low) == horizon else np.nan
            )

        row["max_favorable_return_24h"] = (
            float(np.max(future_highs) / entry - 1)
            if entry > 0 and len(future_highs) == 24 else np.nan
        )
        row["max_adverse_return_24h"] = (
            float(np.min(future_lows) / entry - 1)
            if entry > 0 and len(future_lows) == 24 else np.nan
        )
        target_indices = np.flatnonzero(future_highs >= entry * (1 + target_return))
        row["target_hit_15pct_24h"] = (
            bool(len(target_indices)) if len(future_highs) == 24 else np.nan
        )
        row["bars_to_target_15pct_24h"] = (
            int(target_indices[0] + 1) if len(target_indices) else np.nan
        ) if len(future_highs) == 24 else np.nan
        rows.append(row)

    ledger = pd.DataFrame(rows)
    if "selected_threshold" in ledger.columns:
        thresholds = pd.to_numeric(ledger["selected_threshold"], errors="coerce")
    else:
        thresholds = pd.Series(0.5, index=ledger.index, dtype=float)
    probabilities = pd.to_numeric(ledger["probability"], errors="coerce")
    ledger["is_model_signal"] = probabilities >= thresholds
    ledger["signal_emitted"] = False
    ledger["cooldown_suppressed"] = False

    last_signal_time = {}
    cooldown = pd.Timedelta(hours=cooldown_bars)
    for idx in ledger.sort_values(["timestamp", "symbol"]).index:
        symbol = str(ledger.at[idx, "symbol"])
        if not bool(ledger.at[idx, "is_model_signal"]):
            continue
        current_time = ledger.at[idx, "timestamp"]
        previous_time = last_signal_time.get(symbol)
        if previous_time is not None and current_time - previous_time <= cooldown:
            ledger.at[idx, "cooldown_suppressed"] = True
            continue
        ledger.at[idx, "signal_emitted"] = True
        last_signal_time[symbol] = current_time

    return ledger.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _feature_associations(ledger: pd.DataFrame) -> pd.DataFrame:
    """Descriptive test-period feature differences; not causal attribution."""
    excluded = {
        "timestamp", "symbol", "actual", "probability", "selected_threshold",
        "predicted_at_selected_threshold", "predicted_at_0_5", "split",
        "open", "high", "low", "close", "volume", "quote_volume",
    }
    numeric = [
        col for col in ledger.select_dtypes(include=[np.number]).columns
        if col not in excluded
    ]
    if "actual" not in ledger.columns or ledger["actual"].nunique() < 2:
        return pd.DataFrame(columns=[
            "feature", "mean_when_target_positive", "mean_when_target_negative",
            "standardized_mean_difference",
        ])

    positives = ledger.loc[ledger["actual"] == 1]
    negatives = ledger.loc[ledger["actual"] == 0]
    rows = []
    for feature in numeric:
        positive_values = pd.to_numeric(positives[feature], errors="coerce").dropna()
        negative_values = pd.to_numeric(negatives[feature], errors="coerce").dropna()
        if positive_values.empty or negative_values.empty:
            continue
        pos_mean = float(positive_values.mean())
        neg_mean = float(negative_values.mean())
        pooled = np.sqrt(
            (positive_values.var(ddof=1) + negative_values.var(ddof=1)) / 2
        )
        standardized = (
            float((pos_mean - neg_mean) / pooled)
            if pd.notna(pooled) and pooled > 0 else np.nan
        )
        rows.append({
            "feature": feature,
            "mean_when_target_positive": pos_mean,
            "mean_when_target_negative": neg_mean,
            "standardized_mean_difference": standardized,
        })
    return pd.DataFrame(rows).sort_values(
        "standardized_mean_difference",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last",
    ) if rows else pd.DataFrame(columns=[
        "feature", "mean_when_target_positive", "mean_when_target_negative",
        "standardized_mean_difference",
    ])


def run_monitor(
    features_path: Path = FEATURES_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    report_dir: Path = REPORT_DIR,
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS,
):
    """Write the all-predictions ledger, emitted-signal outcomes, and summary."""
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing {features_path}. Run `python main.py analyze` first."
        )
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing {predictions_path}. Run `python main.py train` first."
        )

    features = pd.read_csv(features_path, parse_dates=["timestamp"])
    predictions = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    ledger = build_prediction_ledger(
        features, predictions, cooldown_bars=cooldown_bars
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = report_dir / "prediction_ledger.csv"
    signals_path = report_dir / "recommendation_monitor.csv"
    associations_path = report_dir / "feature_outcome_associations.csv"
    summary_path = report_dir / "recommendation_monitor_summary.json"

    ledger.to_csv(ledger_path, index=False)
    signals = ledger.loc[ledger["signal_emitted"]].copy()
    signals.to_csv(signals_path, index=False)
    associations = _feature_associations(ledger)
    associations.to_csv(associations_path, index=False)

    actual = pd.to_numeric(ledger["actual"], errors="coerce")
    predicted = ledger["is_model_signal"].astype(bool)
    complete = ledger["outcome_status"] == "complete_24h"
    target_outcome = ledger["target_hit_15pct_24h"].fillna(False).astype(bool)

    summary = {
        "mode": "historical_test_evaluation_only",
        "trading_enabled": False,
        "live_alerts_enabled": False,
        "rows_in_prediction_ledger": int(len(ledger)),
        "symbols_evaluated": int(ledger["symbol"].nunique()),
        "test_start": str(ledger["timestamp"].min()),
        "test_end": str(ledger["timestamp"].max()),
        "selected_threshold": (
            float(pd.to_numeric(ledger["selected_threshold"], errors="coerce").dropna().iloc[0])
            if "selected_threshold" in ledger.columns
            and pd.to_numeric(ledger["selected_threshold"], errors="coerce").notna().any()
            else 0.5
        ),
        "model_positive_rows_at_threshold": int(predicted.sum()),
        "signals_emitted_after_cooldown": int(len(signals)),
        "signals_suppressed_by_cooldown": int(ledger["cooldown_suppressed"].sum()),
        "actual_future_15pct_events": int((actual == 1).sum()),
        "false_negative_rows": int(((actual == 1) & (~predicted)).sum()),
        "false_positive_rows": int(((actual == 0) & predicted).sum()),
        "target_hits_among_emitted_signals": int(
            (signals["target_hit_15pct_24h"] == True).sum()  # noqa: E712
        ),
        "complete_24h_outcome_rows": int(complete.sum()),
        "incomplete_or_missing_outcome_rows": int((~complete).sum()),
        "mean_max_favorable_return_24h_emitted_signals": (
            float(pd.to_numeric(
                signals.loc[signals["outcome_status"] == "complete_24h",
                            "max_favorable_return_24h"], errors="coerce"
            ).mean()) if not signals.empty else None
        ),
        "mean_max_adverse_return_24h_emitted_signals": (
            float(pd.to_numeric(
                signals.loc[signals["outcome_status"] == "complete_24h",
                            "max_adverse_return_24h"], errors="coerce"
            ).mean()) if not signals.empty else None
        ),
        "outputs": {
            "prediction_ledger": str(ledger_path),
            "recommendation_monitor": str(signals_path),
            "feature_outcome_associations": str(associations_path),
        },
        "interpretation_warning": (
            "Historical test-set research only. Feature associations are "
            "descriptive, not causal. No trades or live alerts are performed."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_monitor()
