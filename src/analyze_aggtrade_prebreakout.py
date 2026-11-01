"""Statistical analysis for the Binance aggTrades pre-breakout pilot.

This is a research report only: it does not create trading signals or train a model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


def bh_adjust(items):
    ordered = sorted(items, key=lambda x: x["p_value"])
    running = 1.0
    m = len(ordered)
    for rank in range(m, 0, -1):
        item = ordered[rank - 1]
        running = min(running, item["p_value"] * m / rank)
        item["q_value_bh"] = float(running)
    return items


def safe_float(value):
    return None if pd.isna(value) else float(value)


def paired_test(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if len(values) < 5 or np.allclose(values, 0):
        return None
    result = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
    return {
        "n_pairs": int(len(values)),
        "median_event_minus_control": float(values.median()),
        "mean_event_minus_control": float(values.mean()),
        "event_higher_share": float((values > 0).mean()),
        "p_value": float(result.pvalue),
    }


def independent_test(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 5:
        return None
    result = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return {
        "n_confirmed": int(len(a)),
        "n_early_only": int(len(b)),
        "median_confirmed": float(a.median()),
        "median_early_only": float(b.median()),
        "median_confirmed_minus_early_only": float(a.median() - b.median()),
        "p_value": float(result.pvalue),
    }


def find_file(root, filename):
    matches = list(Path(root).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[0]


def main():
    parser_root = Path(__file__).resolve().parents[1]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("input-artifact"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/aggtrade_analysis"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(find_file(args.input_dir, "aggtrade_prebreakout_features.csv"))
    pairs = pd.read_csv(find_file(args.input_dir, "aggtrade_matched_pair_deltas.csv"))
    features["data_complete"] = features["data_complete"].astype(str).str.lower().eq("true")
    complete = features[features["data_complete"]].copy()
    events = complete[complete["role"] == "event"].copy()
    controls = complete[complete["role"] == "control"].copy()

    summary = {
        "sample": {
            "manifest_rows": int(len(features)),
            "complete_rows": int(len(complete)),
            "complete_event_cases": int(len(events)),
            "complete_control_cases": int(len(controls)),
            "complete_matched_pairs": int(len(pairs)),
            "event_cases_by_tier": {str(k): int(v) for k, v in events.groupby("case_tier").size().items()},
            "matched_pairs_by_tier": {str(k): int(v) for k, v in pairs.groupby("event_tier").size().items()},
        },
        "paired_event_vs_control": {},
        "confirmed_vs_early_only": [],
        "chronological_stability": {},
        "interpretation_limits": [
            "This is an observational pilot, not evidence of causality or a deployable trading edge.",
            "Confirmed and early_only anchors are trigger timestamps; preceding windows can include part of an ongoing price move.",
            "early_only means a +10%/24h trigger without a confirmed higher-tier trigger in the source catalog; it is not a clean no-move control.",
            "The source catalog is limited to the existing historical scan and currently listed Binance Spot USDT pairs, retaining discovery and survivorship bias.",
            "Large trades are defined by top 1% quote notional per case; this does not identify wallets or beneficial owners.",
            "All statistical feature families are exploratory; Benjamini-Hochberg correction is applied separately within each tier/comparison family."
        ],
    }

    # Paired event/control comparisons.
    feature_cols = [c for c in pairs.columns if c not in ("pair_id", "event_tier")]
    for tier in ("confirmed", "early_only"):
        group = pairs[pairs["event_tier"] == tier]
        rows = []
        for col in feature_cols:
            test = paired_test(group[col])
            if test:
                rows.append({"feature": col.removesuffix("_event_minus_control"), **test})
        bh_adjust(rows)
        rows.sort(key=lambda x: x["q_value_bh"])
        summary["paired_event_vs_control"][tier] = {
            "n_pairs": int(len(group)),
            "features_tested": len(rows),
            "fdr_significant_features": [
                x for x in rows if x["q_value_bh"] < 0.05
            ],
            "top_features": rows[:15],
        }

    # Direct comparison of the two event tiers, using only complete event rows.
    event_features = [
        c for c in events.columns
        if c.startswith(("h72_", "h24_", "h6_", "h1_")) or c == "volume_acceleration_6h_vs_prior6h"
    ]
    confirmed = events[events["case_tier"] == "confirmed"]
    early = events[events["case_tier"] == "early_only"]
    between = []
    for col in event_features:
        test = independent_test(confirmed[col], early[col])
        if test:
            between.append({"feature": col, **test})
    bh_adjust(between)
    between.sort(key=lambda x: x["q_value_bh"])
    summary["confirmed_vs_early_only"] = between[:20]
    summary["confirmed_vs_early_only_fdr_significant"] = [
        x for x in between if x["q_value_bh"] < 0.05
    ]

    # Chronological stability: rank features on the earliest 70% of each tier,
    # then test only those selected features on the latest 30%.
    event_time = events.set_index(["pair_id"])
    for tier in ("confirmed", "early_only"):
        g = pairs[pairs["event_tier"] == tier].copy()
        anchors = event_time[event_time["case_tier"] == tier]["anchor_ms"].to_dict()
        g["anchor_ms"] = g["pair_id"].map(anchors)
        g = g.dropna(subset=["anchor_ms"]).sort_values("anchor_ms")
        cut = max(5, min(len(g) - 5, int(np.floor(len(g) * 0.70))))
        train, test = g.iloc[:cut], g.iloc[cut:]
        ranked = []
        for col in feature_cols:
            tr = paired_test(train[col])
            if tr:
                ranked.append({"feature": col.removesuffix("_event_minus_control"), **tr})
        ranked.sort(key=lambda x: x["p_value"])
        stability = []
        for item in ranked[:10]:
            col = item["feature"] + "_event_minus_control"
            te = paired_test(test[col])
            if not te:
                continue
            train_direction = np.sign(item["median_event_minus_control"])
            test_direction = np.sign(te["median_event_minus_control"])
            stability.append({
                "feature": item["feature"],
                "train_n": item["n_pairs"],
                "train_median_delta": item["median_event_minus_control"],
                "train_p_value": item["p_value"],
                "test_n": te["n_pairs"],
                "test_median_delta": te["median_event_minus_control"],
                "test_p_value": te["p_value"],
                "same_direction_in_later_period": bool(train_direction == test_direction),
            })
        summary["chronological_stability"][tier] = {
            "method": "Earliest 70% of event timestamps used to rank features; latest 30% used once as a temporal stability check. No model was trained.",
            "train_pairs": int(len(train)),
            "later_test_pairs": int(len(test)),
            "top_train_features_tested_later": stability,
        }

    out = args.out_dir / "aggtrade_prebreakout_analysis.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"ANALYSIS_JSON={out}")


if __name__ == "__main__":
    main()
