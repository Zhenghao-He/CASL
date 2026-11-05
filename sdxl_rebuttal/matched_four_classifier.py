from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "four_classifier_mean_target_change"
PRESERVATION = [
    "four_classifier_mean_offtarget_change",
    "four_classifier_max_offtarget_change",
    "lpips",
    "arcface",
    "four_classifier_mean_epr",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match CASL and Asyrp-style settings by four-classifier target strength."
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output")
    parser.add_argument("--min-positive-fraction", type=float, default=0.75)
    args = parser.parse_args()

    source = Path(args.summary)
    frame = pd.read_csv(source)
    frame = frame[
        (frame[TARGET] > 0)
        & (
            frame["four_classifier_positive_target_fraction"]
            >= args.min_positive_fraction
        )
    ].copy()
    rows: list[dict] = []
    for concept, concept_frame in frame.groupby("concept"):
        casl = concept_frame[concept_frame["method"].isin(["casl", "casl_lowrank"])]
        baseline = concept_frame[concept_frame["method"] == "asyrp_cnn"]
        for _, base in baseline.iterrows():
            if casl.empty:
                continue
            chosen = casl.loc[(casl[TARGET] - base[TARGET]).abs().idxmin()]
            denominator = max(abs(float(base[TARGET])), 1e-8)
            row = {
                "concept": concept,
                "baseline_strength": base["strength"],
                "baseline_target": base[TARGET],
                "casl_method": chosen["method"],
                "casl_topk": chosen["topk"],
                "casl_strength": chosen["strength"],
                "casl_target": chosen[TARGET],
                "absolute_target_mismatch": abs(chosen[TARGET] - base[TARGET]),
                "relative_target_mismatch": abs(chosen[TARGET] - base[TARGET])
                / denominator,
                "baseline_positive_fraction": base[
                    "four_classifier_positive_target_fraction"
                ],
                "casl_positive_fraction": chosen[
                    "four_classifier_positive_target_fraction"
                ],
            }
            for metric in PRESERVATION:
                row[f"baseline_{metric}"] = base[metric]
                row[f"casl_{metric}"] = chosen[metric]
                row[f"casl_minus_baseline_{metric}"] = chosen[metric] - base[metric]
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values(
            ["concept", "relative_target_mismatch", "baseline_target"]
        )
    output = (
        Path(args.output)
        if args.output
        else source.with_name("matched_target_four_classifier.csv")
    )
    result.to_csv(output, index=False)
    close = (
        result[result["relative_target_mismatch"] <= 0.10]
        if len(result)
        else result
    )
    print(f"[saved] {output} ({len(result)} pairs; {len(close)} within 10% target)")
    if len(close):
        wins = pd.DataFrame(
            {
                "lower_offtarget": close[
                    "casl_minus_baseline_four_classifier_mean_offtarget_change"
                ]
                < 0,
                "lower_lpips": close["casl_minus_baseline_lpips"] < 0,
                "higher_arcface": close["casl_minus_baseline_arcface"] > 0,
                "higher_epr": close[
                    "casl_minus_baseline_four_classifier_mean_epr"
                ]
                > 0,
            }
        )
        print(wins.mean().rename("CASL win fraction").to_string())


if __name__ == "__main__":
    main()
