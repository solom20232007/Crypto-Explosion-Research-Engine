"""Cross-sectional Binance pre-breakout research baseline.

This module ranks assets by their likelihood of becoming future top gainers.
It is research-only: it does not change V7 settings or emit live trade calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = [
    "ret_1h", "ret_3h", "ret_6h", "ret_12h", "ret_24h",
    "ret_48h", "ret_72h", "rsi_14", "macd_hist", "atr_14_pct",
    "bb_width_20", "bb_position_20", "sma_ratio_20", "sma_ratio_50",
    "volume_ratio_20", "quote_volume_log", "quote_volume_ratio_20",
    "volatility_24h",
]
HORIZONS = (4, 12, 24)
TOP_FRACTION = 0.10
MIN_SYMBOLS_PER_TIMESTAMP = 10
EMBARGO_HOURS = 24
FEATURE_PATH = Path("data/features.csv")
REPORT_PATH = Path("reports/prebreakout_advisor.json")


def add_cross_sectional_targets(frame: pd.DataFrame, horizon: int,
                                top_fraction: float = TOP_FRACTION,
                                min_symbols: int = MIN_SYMBOLS_PER_TIMESTAMP) -> pd.DataFrame:
    """Label the future top fraction at each timestamp using future highs."""
    required = {"timestamp", "symbol", "close", "high"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["timestamp", "symbol", "close", "high"])
    data = data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    future_returns = pd.Series(np.nan, index=data.index, dtype=float)

    for _, indexes in data.groupby("symbol", sort=False).groups.items():
        indexes = list(indexes)
        closes = pd.to_numeric(data.loc[indexes, "close"], errors="coerce").to_numpy(float)
        highs = pd.to_numeric(data.loc[indexes, "high"], errors="coerce").to_numpy(float)
        timestamps = data.loc[indexes, "timestamp"].to_numpy()
        values = np.full(len(indexes), np.nan, dtype=float)
        expected_step = np.timedelta64(1, "h")
        for pos in range(len(indexes) - horizon):
            # A row-count horizon is only a true hour horizon when every candle
            # is present. Missing/duplicate-time gaps must not silently stretch
            # or shrink the forecast window.
            window_times = timestamps[pos:pos + horizon + 1]
            if not np.all(np.diff(window_times) == expected_step):
                continue
            future_window = highs[pos + 1:pos + horizon + 1]
            if (np.isfinite(future_window).any()
                    and np.isfinite(closes[pos]) and closes[pos] > 0):
                future_high = np.nanmax(future_window)
                values[pos] = future_high / closes[pos] - 1.0
        future_returns.loc[indexes] = values

    return_col = f"future_max_return_{horizon}h"
    target_name = f"target_top_gainer_{horizon}h"
    data[return_col] = future_returns
    data[target_name] = np.nan

    valid = data.dropna(subset=[return_col])
    counts = valid.groupby("timestamp")["symbol"].transform("nunique")
    eligible = valid.loc[counts >= min_symbols].copy()
    if not eligible.empty:
        ranks = eligible.groupby("timestamp")[return_col].rank(
            method="first", ascending=False, pct=True
        )
        data.loc[eligible.index, target_name] = (ranks <= top_fraction).astype(int)
    return data


def _make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )),
    ])


def _top_k_metrics(frame: pd.DataFrame, probabilities: np.ndarray,
                   target: str, fraction: float = TOP_FRACTION) -> dict:
    scored = frame[["timestamp", "symbol", target]].copy()
    scored["probability"] = probabilities
    hits = []
    total_hits = 0
    total_selected = 0
    for _, group in scored.groupby("timestamp"):
        group = group.sort_values("probability", ascending=False)
        k = max(1, int(np.ceil(len(group) * fraction)))
        selected = group.head(k)
        hits.extend(selected[target].astype(int).tolist())
        total_hits += int(selected[target].sum())
        total_selected += len(selected)
    y = scored[target].astype(int).to_numpy()
    result = {
        "rows": int(len(scored)),
        "positive_rate": float(np.mean(y)),
        "top_fraction": float(fraction),
        "precision_at_top_fraction": float(np.mean(hits)) if hits else None,
        "selected_rows": total_selected,
        "selected_hits": total_hits,
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if np.unique(y).size == 2 else None,
    }
    baseline = result["positive_rate"]
    result["lift_at_top_fraction"] = (
        result["precision_at_top_fraction"] / baseline if baseline > 0 else None
    )
    return result


def run_advisor(feature_path: Path = FEATURE_PATH, report_path: Path = REPORT_PATH,
                horizons: tuple[int, ...] = HORIZONS) -> dict:
    if not feature_path.exists():
        raise SystemExit(
            f"Missing {feature_path}. Run collection and analysis first; "
            "the current analysis data may not cover enough symbols."
        )
    frame = pd.read_csv(feature_path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    missing_features = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing_features:
        raise SystemExit("Missing model features: " + ", ".join(missing_features))
    frame = frame.dropna(subset=["timestamp", "symbol", "close", "high"]).copy()
    symbols = int(frame["symbol"].nunique())
    if symbols < MIN_SYMBOLS_PER_TIMESTAMP:
        raise SystemExit(
            f"Pre-breakout advisor needs at least {MIN_SYMBOLS_PER_TIMESTAMP} "
            f"symbols in the historical universe; found only {symbols}. "
            "Do not interpret BTC/ETH/SOL-only results as a top-gainer model."
        )

    results = {
        "status": "research_baseline",
        "symbols": symbols,
        "rows": int(len(frame)),
        "date_min": str(frame["timestamp"].min()),
        "date_max": str(frame["timestamp"].max()),
        "target_definition": (
            "At each timestamp, label the top 10% of symbols by maximum future "
            "high return over the next horizon; incomplete horizons are excluded."
        ),
        "horizons_hours": {},
        "v7_recommendations": [],
        "limitations": [
            "Historical coverage is only as complete as data/features.csv.",
            "News and announcement features are not included in this baseline.",
            "This script does not edit or automatically deploy V7 settings.",
            "Historical top-gainer labels can be noisy and do not guarantee future gains.",
        ],
    }

    for horizon in horizons:
        labelled = add_cross_sectional_targets(frame, horizon)
        target = f"target_top_gainer_{horizon}h"
        labelled = labelled.dropna(subset=[target]).copy()
        if labelled.empty or labelled[target].nunique() < 2:
            results["horizons_hours"][str(horizon)] = {
                "status": "insufficient_labels",
                "rows": int(len(labelled)),
                "message": "No usable cross-sectional top-gainer labels; collect more symbols/history.",
            }
            continue

        labelled[target] = labelled[target].astype(int)
        times = np.array(sorted(labelled["timestamp"].unique()))
        if len(times) < 100:
            results["horizons_hours"][str(horizon)] = {
                "status": "insufficient_history",
                "unique_timestamps": int(len(times)),
            }
            continue

        first = pd.Timestamp(times[int(len(times) * 0.70)])
        second = pd.Timestamp(times[int(len(times) * 0.85)])
        if first.tzinfo is None:
            first = first.tz_localize("UTC")
        if second.tzinfo is None:
            second = second.tz_localize("UTC")
        embargo = pd.Timedelta(hours=max(EMBARGO_HOURS, horizon))
        train = labelled[labelled["timestamp"] < first - embargo]
        validation = labelled[
            (labelled["timestamp"] >= first) & (labelled["timestamp"] < second - embargo)
        ]
        test = labelled[labelled["timestamp"] >= second]
        if (train.empty or validation.empty or test.empty
                or train[target].nunique() < 2
                or validation[target].nunique() < 2
                or test[target].nunique() < 2):
            results["horizons_hours"][str(horizon)] = {
                "status": "insufficient_class_coverage",
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "message": "A chronological split lacks both target classes; no model metrics reported.",
            }
            continue

        model = _make_model()
        model.fit(train[FEATURE_COLUMNS], train[target])
        validation_prob = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        test_prob = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        validation_metrics = _top_k_metrics(validation, validation_prob, target)
        test_metrics = _top_k_metrics(test, test_prob, target)
        importance = permutation_importance(
            model, validation[FEATURE_COLUMNS], validation[target],
            scoring="average_precision", n_repeats=5, random_state=42,
        )
        ranked = sorted(
            [{"feature": name, "importance_mean": float(mean),
              "importance_std": float(std)}
             for name, mean, std in zip(FEATURE_COLUMNS, importance.importances_mean,
                                        importance.importances_std)],
            key=lambda item: item["importance_mean"], reverse=True,
        )
        results["horizons_hours"][str(horizon)] = {
            "status": "ok",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "validation": validation_metrics,
            "test": test_metrics,
            "validation_feature_importance": ranked[:10],
        }

    for horizon, detail in results["horizons_hours"].items():
        if detail.get("status") != "ok":
            continue
        top = detail["validation_feature_importance"][:5]
        results["v7_recommendations"].append({
            "horizon_hours": int(horizon),
            "type": "review_indicator_priority",
            "priority_features": [item["feature"] for item in top],
            "evidence": (
                "Ranked by permutation importance on chronological validation data; "
                "confirm on out-of-time test and compare against V7's actual rules."
            ),
            "action": "Research only: inspect V7 settings and test one change at a time; do not auto-apply.",
        })

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved advisor report: {report_path}")
    return results


if __name__ == "__main__":
    run_advisor()
