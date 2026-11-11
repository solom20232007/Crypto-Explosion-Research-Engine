"""Chronological holdout evaluation for causal pre-breakout heuristics.

This evaluates fixed, predeclared thresholds on the last 20% of each symbol's
timeline. It does not fit thresholds on the holdout and is research-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/features.csv")
REPORT_DIR = Path("reports")
TARGET = "target_future_15pct_24h"
SCORE_COLUMNS = {
    "prebreakout_score_ge_0_6": ("prebreakout_score", 0.6),
    "early_breakout_score_ge_0_6": ("early_breakout_score", 0.6),
}
PHASE_SIGNALS = {
    "phase_prebreakout_setup": ("breakout_phase", "prebreakout_setup"),
    "phase_early_breakout": ("breakout_phase", "early_breakout"),
    "phase_active_explosion": ("breakout_phase", "active_explosion"),
}


def _metrics(frame: pd.DataFrame, prediction: pd.Series, label: str) -> dict:
    y = frame[TARGET].astype(int)
    pred = prediction.reindex(frame.index).fillna(False).astype(bool)
    tp = int((pred & y.eq(1)).sum())
    fp = int((pred & y.eq(0)).sum())
    fn = int((~pred & y.eq(1)).sum())
    tn = int((~pred & y.eq(0)).sum())
    n = len(frame)
    prevalence = float(y.mean()) if n else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "signal": label,
        "test_rows": n,
        "positive_rows": int(y.sum()),
        "baseline_positive_rate": prevalence,
        "signals": int(pred.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "lift_over_baseline": (precision / prevalence if precision is not None and prevalence else None),
        "signal_rate": float(pred.mean()) if n else None,
    }


def run_backtest(features_path: Path = FEATURES_PATH, report_dir: Path = REPORT_DIR) -> dict:
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}; run analysis first.")
    df = pd.read_csv(features_path)
    required = {"timestamp", "symbol", TARGET, "prebreakout_score",
                "early_breakout_score", "breakout_phase"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required backtest columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", TARGET]).copy()
    if not set(df[TARGET].unique()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError("Target must contain only 0/1 values after removing incomplete windows.")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    test_parts = []
    split_rows = []
    for symbol, group in df.groupby("symbol", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        split = int(np.floor(len(group) * 0.8))
        if split <= 0 or split >= len(group):
            raise ValueError(f"Not enough rows for chronological split: {symbol} ({len(group)})")
        test = group.iloc[split:].copy()
        test_parts.append(test)
        split_rows.append({
            "symbol": symbol,
            "rows_total": int(len(group)),
            "train_rows": int(split),
            "test_rows_with_valid_target": int(len(test)),
            "train_end": str(group.iloc[split - 1]["timestamp"]),
            "test_start": str(test["timestamp"].min()),
            "test_end": str(test["timestamp"].max()),
        })
    test_df = pd.concat(test_parts, ignore_index=True)
    if test_df.empty:
        raise ValueError("Chronological holdout contains no rows.")

    records = []
    signals = {}
    for name, (column, threshold) in SCORE_COLUMNS.items():
        signals[name] = test_df[column] >= threshold
    for name, (column, value) in PHASE_SIGNALS.items():
        signals[name] = test_df[column].eq(value)

    for symbol in sorted(test_df["symbol"].unique()):
        part = test_df[test_df["symbol"] == symbol]
        for name, pred in signals.items():
            records.append({"symbol": symbol, **_metrics(part, pred.loc[part.index], name)})
    for name, pred in signals.items():
        records.append({"symbol": "ALL", **_metrics(test_df, pred, name)})

    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(report_dir / "backtest_metrics.csv", index=False)
    summary = {
        "method": "Per-symbol chronological 80/20 split; fixed heuristic thresholds; no threshold tuning on holdout.",
        "target": "Any high in the next 24 hourly candles reaches at least 15% above the current close.",
        "holdout_rows": int(len(test_df)),
        "holdout_positive_rows": int(test_df[TARGET].sum()),
        "holdout_positive_rate": float(test_df[TARGET].mean()),
        "splits": split_rows,
        "metrics_csv": str(report_dir / "backtest_metrics.csv"),
        "interpretation": (
            "Precision is the share of signal candles followed by the target event within 24 bars. "
            "Because horizons overlap, rows are not independent and metrics are not trade-level returns. "
            "This is a first diagnostic, not proof of predictive edge or trading advice."
        ),
        "results": records,
    }
    (report_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_backtest()
