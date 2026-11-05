#!/usr/bin/env python3
"""Evaluate native intensity sweeps with a shared four-classifier EPR."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novelty_ablation.evaluate_attribute_metrics import (
    CLASSIFIERS,
    CONCEPTS,
    casl_directory,
    id_path_map,
)
from sdxl_rebuttal.evaluate import attribute_names, classify_paths


CASL_STRENGTHS = (0, 32, 64, 96, 128, 160, 192)
BOUNDARY_STRENGTHS = (0, 3, 5, 7, 9, 12, 15, 19)
ASYRP_STRENGTHS = (0, 0.5, 1, 1.5, 2)
SLIDER_STRENGTHS = (0, 32, 64, 128, 192)
BOUNDARY_NAMES = {
    "smiling": "Smiling",
    "young": "Young",
    "big_nose": "Big_Nose",
    "beards": "Mustache",
    "blond_hair": "Blond_Hair",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "alpha_rebuttal/results/epr_sweeps",
    )
    return parser.parse_args()


def add_records(records, method, concept, strength, reference, edited, ids):
    for image_id in ids:
        if image_id in reference and image_id in edited:
            records.append(
                {
                    "method": method,
                    "concept": concept,
                    "strength": float(strength),
                    "image_id": image_id,
                    "reference": str(reference[image_id]),
                    "edited": str(edited[image_id]),
                }
            )


def directory_map(directory, suffix):
    return {
        path.stem: path
        for path in directory.glob(f"*{suffix}")
        if path.is_file()
    }


def build_records(n_images):
    ids = [f"{value:05d}" for value in range(29501, 29501 + n_images)]
    records = []
    for concept in CONCEPTS:
        casl_reference = id_path_map(casl_directory(concept, 0), "casl")
        for strength in CASL_STRENGTHS:
            add_records(
                records,
                "CASL",
                concept,
                strength,
                casl_reference,
                id_path_map(casl_directory(concept, strength), "casl"),
                ids,
            )

        boundary_root = (
            ROOT
            / "results_500range"
            / f"results_{BOUNDARY_NAMES[concept]}_500Range"
        )
        boundary_reference = {
            path.parent.name: path for path in boundary_root.glob("*/recons.png")
        }
        for strength in BOUNDARY_STRENGTHS:
            edited = {
                path.parent.name: path
                for path in boundary_root.glob(f"*/edited_{strength}.png")
            }
            add_records(
                records,
                "Boundary",
                concept,
                strength,
                boundary_reference,
                edited,
                ids,
            )

        asyrp_root = ROOT / "alpha_rebuttal/results/asyrp" / concept
        asyrp_reference = directory_map(asyrp_root / "coeff_0", ".png")
        for strength in ASYRP_STRENGTHS:
            add_records(
                records,
                "ASYRP",
                concept,
                strength,
                asyrp_reference,
                directory_map(asyrp_root / f"coeff_{strength:g}", ".png"),
                ids,
            )

        slider_root = ROOT / "alpha_rebuttal/results/slider" / concept
        slider_reference = directory_map(slider_root / "scale_0", ".png")
        for strength in SLIDER_STRENGTHS:
            add_records(
                records,
                "Concept Slider",
                concept,
                strength,
                slider_reference,
                directory_map(slider_root / f"scale_{strength:g}", ".png"),
                ids,
            )
    return records


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(args.n_images)
    if not records:
        raise RuntimeError("No intensity outputs found")
    paths = sorted(
        {row["reference"] for row in records} | {row["edited"] for row in records}
    )
    predictions = classify_paths(paths, torch.device(args.device), args.batch_size)
    names = attribute_names()
    rows = []
    for record in records:
        attribute, sign, _ = CONCEPTS[record["concept"]]
        target = names.index(attribute)
        non_target = [index for index in range(len(names)) if index != target]
        row = dict(record)
        for classifier in CLASSIFIERS:
            before = predictions[record["reference"]]
            after = predictions[record["edited"]]
            probability_delta = (
                after[f"{classifier}_probs"] - before[f"{classifier}_probs"]
            )
            logit_delta = (
                after[f"{classifier}_logits"] - before[f"{classifier}_logits"]
            )
            row[f"{classifier}_target_delta"] = float(
                sign * probability_delta[target]
            )
            row[f"{classifier}_target_logit_abs"] = float(
                abs(logit_delta[target])
            )
            row[f"{classifier}_nontarget_delta"] = float(
                np.abs(probability_delta[non_target]).mean()
            )
            row[f"{classifier}_nontarget_logit_abs"] = float(
                np.abs(logit_delta[non_target]).mean()
            )
        row["target_delta"] = float(
            np.mean([row[f"{name}_target_delta"] for name in CLASSIFIERS])
        )
        row["nontarget_delta"] = float(
            np.mean([row[f"{name}_nontarget_delta"] for name in CLASSIFIERS])
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    summaries = []
    for (method, concept, strength), group in frame.groupby(
        ["method", "concept", "strength"]
    ):
        result = {
            "method": method,
            "concept": concept,
            "strength": strength,
            "n": len(group),
            "target_delta": group["target_delta"].mean(),
            "nontarget_delta": group["nontarget_delta"].mean(),
        }
        eprs = []
        for classifier in CLASSIFIERS:
            numerator = group[f"{classifier}_target_logit_abs"].mean()
            denominator = group[f"{classifier}_nontarget_logit_abs"].mean()
            eprs.append(float(numerator / max(float(denominator), 1e-8)))
        result["epr"] = float(np.mean(eprs))
        summaries.append(result)
    summary = pd.DataFrame(summaries).sort_values(
        ["method", "concept", "strength"]
    )
    frame.to_csv(args.output_dir / "per_image.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
