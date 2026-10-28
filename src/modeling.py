"""Leakage-aware temporal baseline training for the research engine.

This model is for historical research only. It does not execute trades or
produce live alerts.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "ret_1h", "ret_3h", "ret_6h", "ret_12h", "ret_24h", "ret_48h", "ret_72h",
    "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14_pct",
    "bb_width_20", "bb_position_20", "sma_ratio_20", "sma_ratio_50",
    "volume_ratio_20", "quote_volume_log", "quote_volume_ratio_20", "volatility_24h",
]
TARGET = "target_future_15pct_24h"
HORIZON_HOURS = 24
RANDOM_SEED = 42


def _make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.07,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def _choose_threshold(y_true, probabilities):
    """Choose a threshold on validation data only; ties prefer fewer alerts."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.arange(0.01, 0.501, 0.01)
    scored = []

    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predicted, zero_division=0)
        scored.append((float(score), float(threshold)))

    # A fixed grid avoids choosing arbitrary thresholds from test outcomes.
    # On an F1 tie, prefer the higher threshold to reduce alert volume.
    best_score = max(item[0] for item in scored)
    best_threshold = max(
        threshold for score, threshold in scored if score == best_score
    )
    return best_threshold, best_score


def _metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)

    result = {
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rows": int(predicted.sum()),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_score": float(f1_score(y_true, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities))
        if np.unique(y_true).size == 2 else None
    )
    result["average_precision"] = float(
        average_precision_score(y_true, probabilities)
    )
    return result


def train_model():
    path = Path("data/features.csv")
    if not path.exists():
        raise SystemExit(
            "Missing data/features.csv. Run `python main.py analyze` first."
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "close", "high", "low", TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=[TARGET, "timestamp"]).copy()
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if len(df) < 500:
        raise SystemExit(
            f"Not enough labeled rows ({len(df)}). Collect more history/symbols."
        )

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise SystemExit(f"Missing feature columns: {', '.join(missing_features)}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise SystemExit("Target must contain only 0/1 labels.")

    unique_times = np.array(sorted(df["timestamp"].dropna().unique()))
    if len(unique_times) < 200:
        raise SystemExit("Not enough unique timestamps for purged temporal splits.")

    # Train / validation / test are chronological and share boundaries across
    # symbols. Exclude 24 hours before each boundary because labels look forward
    # 24 hourly bars. This is a timestamp embargo, not a random split.
    first_boundary = pd.Timestamp(unique_times[int(len(unique_times) * 0.70)])
    second_boundary = pd.Timestamp(unique_times[int(len(unique_times) * 0.85)])
    # NumPy may strip the UTC timezone from datetime64 values.
    if first_boundary.tzinfo is None:
        first_boundary = first_boundary.tz_localize("UTC")
    else:
        first_boundary = first_boundary.tz_convert("UTC")
    if second_boundary.tzinfo is None:
        second_boundary = second_boundary.tz_localize("UTC")
    else:
        second_boundary = second_boundary.tz_convert("UTC")
    embargo = pd.Timedelta(hours=HORIZON_HOURS)

    train_mask = df["timestamp"] < (first_boundary - embargo)
    validation_mask = (
        (df["timestamp"] >= first_boundary)
        & (df["timestamp"] < (second_boundary - embargo))
    )
    test_mask = df["timestamp"] >= second_boundary

    split_masks = {
        "train": train_mask,
        "validation": validation_mask,
        "test": test_mask,
    }
    for name, mask in split_masks.items():
        subset = df.loc[mask]
        if subset.empty or subset[TARGET].nunique() < 2:
            raise SystemExit(
                f"{name.capitalize()} period must contain both target classes. "
                "Collect more history/symbols or revisit the time boundaries."
            )

    X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET].astype(int)

    model = _make_model()
    model.fit(X.loc[train_mask], y.loc[train_mask])

    validation_probabilities = model.predict_proba(X.loc[validation_mask])[:, 1]
    selected_threshold, validation_best_f1 = _choose_threshold(
        y.loc[validation_mask], validation_probabilities
    )
    test_probabilities = model.predict_proba(X.loc[test_mask])[:, 1]

    validation_metrics = _metrics(
        y.loc[validation_mask], validation_probabilities, selected_threshold
    )
    test_metrics_selected = _metrics(
        y.loc[test_mask], test_probabilities, selected_threshold
    )
    test_metrics_05 = _metrics(y.loc[test_mask], test_probabilities, 0.5)

    trained_at = pd.Timestamp.now(tz="UTC")
    metrics = {
        "model": "HistGradientBoostingClassifier baseline",
        "model_name": "explosion_baseline",
        "target": "future high >= +15% versus current close in next 24 hourly bars",
        "threshold": float(selected_threshold),
        "selected_threshold": float(selected_threshold),
        "threshold_selection": "Maximum F1 on validation only; fixed 0.01..0.50 grid; ties prefer higher threshold",
        "validation_best_f1": float(validation_best_f1),
        "validation": validation_metrics,
        "test_at_selected_threshold": test_metrics_selected,
        "test_at_0_5": test_metrics_05,
        "split_method": "Chronological 70/15/15 timestamps with 24-hour embargo before validation/test boundaries",
        "train_start": df.loc[train_mask, "timestamp"].min().isoformat(),
        "train_end": df.loc[train_mask, "timestamp"].max().isoformat(),
        "validation_start": df.loc[validation_mask, "timestamp"].min().isoformat(),
        "validation_end": df.loc[validation_mask, "timestamp"].max().isoformat(),
        "test_start": df.loc[test_mask, "timestamp"].min().isoformat(),
        "test_end": df.loc[test_mask, "timestamp"].max().isoformat(),
        "trained_at_utc": trained_at.isoformat(),
        "rows_total": int(len(df)),
        "rows_train": int(train_mask.sum()),
        "rows_validation": int(validation_mask.sum()),
        "rows_test": int(test_mask.sum()),
        "symbols": sorted(df["symbol"].astype(str).unique().tolist()),
        # Backward-compatible metric keys used by the existing Supabase uploader.
        "roc_auc": test_metrics_05["roc_auc"],
        "average_precision": test_metrics_05["average_precision"],
        "precision_at_0_5": test_metrics_05["precision"],
        "recall_at_0_5": test_metrics_05["recall"],
        "f1_score_at_0_5": test_metrics_05["f1_score"],
        "brier_score": test_metrics_05["brier_score"],
        "precision_at_selected_threshold": test_metrics_selected["precision"],
        "recall_at_selected_threshold": test_metrics_selected["recall"],
        "f1_score_at_selected_threshold": test_metrics_selected["f1_score"],
        "warning": (
            "Research prototype only, not live trading signals. Threshold is "
            "selected on validation only; final test metrics are kept separate. "
            "Results may remain unstable because positive events are rare and "
            "the target is based on overlapping 24-hour future windows."
        ),
    }

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "target": TARGET,
            "threshold": selected_threshold,
            "version": "0.3.0",
            "horizon_hours": HORIZON_HOURS,
        },
        "models/explosion_baseline.joblib",
    )
    Path("reports/model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Keep raw prices and all features with every test prediction so the
    # evaluation module can reconstruct post-signal outcomes without leakage.
    test_rows = df.loc[test_mask].copy()
    test_rows["actual"] = y.loc[test_mask].to_numpy()
    test_rows["probability"] = test_probabilities
    test_rows["selected_threshold"] = selected_threshold
    test_rows["predicted_at_selected_threshold"] = (
        test_probabilities >= selected_threshold
    ).astype(int)
    test_rows["predicted_at_0_5"] = (test_probabilities >= 0.5).astype(int)
    test_rows["split"] = "test"
    test_rows.to_csv("reports/test_predictions.csv", index=False)

    validation_rows = df.loc[validation_mask].copy()
    validation_rows["actual"] = y.loc[validation_mask].to_numpy()
    validation_rows["probability"] = validation_probabilities
    validation_rows["selected_threshold"] = selected_threshold
    validation_rows["predicted_at_selected_threshold"] = (
        validation_probabilities >= selected_threshold
    ).astype(int)
    validation_rows["split"] = "validation"
    validation_rows.to_csv("reports/validation_predictions.csv", index=False)

    metadata = {
        "model_name": "explosion_baseline",
        "version": "0.3.0",
        "trained_at_utc": trained_at.isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "selected_threshold": float(selected_threshold),
        "threshold_selected_on": "validation",
        "horizon_hours": HORIZON_HOURS,
        "not_for_live_signals": True,
    }
    Path("reports/model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_model()
