
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "ret_1h", "ret_3h", "ret_6h", "ret_12h", "ret_24h", "ret_48h", "ret_72h",
    "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14_pct",
    "bb_width_20", "bb_position_20", "sma_ratio_20", "sma_ratio_50",
    "volume_ratio_20", "quote_volume_log", "quote_volume_ratio_20", "volatility_24h",
]

TARGET = "target_future_15pct_24h"
THRESHOLD = 0.5


def train_model():
    path = Path("data/features.csv")

    if not path.exists():
        raise SystemExit(
            "Missing data/features.csv. Run `python main.py analyze` first."
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.dropna(subset=[TARGET, "timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 500:
        raise SystemExit(
            f"Not enough labeled rows ({len(df)}). "
            "Collect more history/symbols before training."
        )

    missing_features = [
        col for col in FEATURE_COLUMNS if col not in df.columns
    ]
    if missing_features:
        raise SystemExit(
            f"Missing feature columns: {', '.join(missing_features)}"
        )

    X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET].astype(int)

    # Keep all symbols from the same timestamp in the same split.
    unique_times = np.array(sorted(df["timestamp"].unique()))

    if len(unique_times) < 2:
        raise SystemExit("Not enough unique timestamps for a time-based split.")

    split_index = int(len(unique_times) * 0.80)
    split_index = min(max(split_index, 1), len(unique_times) - 1)
    split_time = unique_times[split_index]

    train_mask = df["timestamp"] < split_time
    test_mask = df["timestamp"] >= split_time

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise SystemExit(
            "Training/test period has only one class. "
            "Collect more symbols/history or adjust the time split."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.07,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    trained_at = pd.Timestamp.now(tz="UTC")

    metrics = {
        "model": "HistGradientBoostingClassifier baseline",
        "model_name": "explosion_baseline",
        "target": (
            "future high >= +15% versus current close "
            "in next 24 hourly bars"
        ),
        "threshold": THRESHOLD,
        "split": (
            f"chronological timestamp split at "
            f"{pd.Timestamp(split_time).isoformat()} "
            "(about 80% train / 20% test)"
        ),
        "train_start": pd.Timestamp(
            df.loc[train_mask, "timestamp"].min()
        ).isoformat(),
        "train_end": pd.Timestamp(
            df.loc[train_mask, "timestamp"].max()
        ).isoformat(),
        "test_start": pd.Timestamp(
            df.loc[test_mask, "timestamp"].min()
        ).isoformat(),
        "test_end": pd.Timestamp(
            df.loc[test_mask, "timestamp"].max()
        ).isoformat(),
        "trained_at_utc": trained_at.isoformat(),
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "precision_at_0_5": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_at_0_5": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score_at_0_5": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "brier_score": float(
            brier_score_loss(y_test, probabilities)
        ),
        "warning": (
            "Prototype baseline only. Overlapping future labels and "
            "cross-symbol timestamp split can complicate evaluation. "
            "Next iteration should add purged walk-forward validation, "
            "threshold selection on validation data only, and per-symbol "
            "liquidity filters. Not for live trading signals."
        ),
    }

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "target": TARGET,
        },
        "models/explosion_baseline.joblib",
    )

    Path("reports/model_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    test_rows = df.loc[test_mask, ["timestamp", "symbol"]].copy()
    test_rows["actual"] = y_test.to_numpy()
    test_rows["probability"] = probabilities
    test_rows["predicted_at_0_5"] = predictions
    test_rows.to_csv("reports/test_predictions.csv", index=False)

    metadata = {
        "model_name": "explosion_baseline",
        "version": "0.2.0",
        "trained_at_utc": trained_at.isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "not_for_live_signals": True,
    }

    Path("reports/model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_model()
