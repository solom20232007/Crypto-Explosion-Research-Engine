"""Evaluate simple pre-breakout classifiers with chronological holdout splits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline

FEATURES = [
    "return_1d_pct", "return_3d_pct", "return_7d_pct", "return_14d_pct",
    "range_pct", "body_pct", "close_location", "upper_wick_pct", "lower_wick_pct",
    "range_mean_7d_pct", "range_mean_30d_pct", "volatility_ratio_7d_30d",
    "quote_volume_ratio_1d_7d", "quote_volume_ratio_1d_30d", "quote_volume_trend_3d_vs_7d",
]
TARGETS = ["explosion_1d", "explosion_3d", "explosion_7d"]


def safe_metrics(y, pred, score):
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "precision_at_threshold_0_5": float(precision_score(y, pred, zero_division=0)) if len(y) else None,
        "recall_at_threshold_0_5": float(recall_score(y, pred, zero_division=0)) if len(y) else None,
        "average_precision": float(average_precision_score(y, score)) if len(y) and y.nunique() == 2 else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(y) and y.nunique() == 2 else None,
    }


def evaluate(data: pd.DataFrame) -> dict:
    results = {"features": FEATURES, "targets": {}, "notes": [
        "Baseline and random forest are research benchmarks, not trading recommendations.",
        "Rows are daily and labels overlap; confidence intervals and episode-level deduplication are still needed.",
        "Current listed symbols only: survivorship bias remains.",
        "No fees, slippage, funding, order-book depth, or executable-entry assumptions.",
    ]}
    for target in TARGETS:
        labelled = data.dropna(subset=[target]).copy()
        labelled[target] = labelled[target].astype(int)
        train = labelled[labelled.split == "train"]
        val = labelled[labelled.split == "validation"]
        test = labelled[labelled.split == "test"]
        item = {"labelled_rows": int(len(labelled)), "splits": {}}
        if train.empty or val.empty or test.empty or train[target].nunique() < 2:
            item["error"] = "Insufficient rows or both classes are not present in training."
            results["targets"][target] = item
            continue
        x_train, y_train = train[FEATURES], train[target]
        models = {
            "majority_baseline": make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="prior")),
            "random_forest": make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                RandomForestClassifier(
                    n_estimators=250, min_samples_leaf=20, class_weight="balanced_subsample",
                    max_features="sqrt", random_state=42, n_jobs=-1
                ),
            ),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            item["splits"][name] = {}
            for split_name, part in (("validation", val), ("test", test)):
                if part.empty:
                    item["splits"][name][split_name] = {"rows": 0}
                    continue
                scores = model.predict_proba(part[FEATURES])[:, 1]
                pred = (scores >= 0.5).astype(int)
                item["splits"][name][split_name] = safe_metrics(part[target], pd.Series(pred, index=part.index), pd.Series(scores, index=part.index))
            if name == "random_forest":
                forest = model.named_steps["randomforestclassifier"]
                item["feature_importance"] = sorted(
                    [{"feature": feature, "importance": float(value)} for feature, value in zip(FEATURES, forest.feature_importances_)],
                    key=lambda x: x["importance"], reverse=True
                )
        results["targets"][target] = item
    return results


def main():
    src = Path("reports/prebreakout_dataset/prebreakout_dataset.csv")
    out_dir = Path("reports/prebreakout_dataset")
    data = pd.read_csv(src)
    result = evaluate(data)
    (out_dir / "model_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
