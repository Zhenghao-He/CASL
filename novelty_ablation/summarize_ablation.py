#!/usr/bin/env python3
"""Create a compact rebuttal table from held-out novelty-ablation results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_NAMES = {
    "asyrp": "Official ASYRP",
    "dense_clip": "Dense-CLIP",
    "sae_only": "SAE-only",
    "casl": "CASL",
}
CONCEPT_NAMES = {
    "smiling": "Smiling",
    "young": "Young",
    "big_nose": "Big Nose",
    "beards": "Beards",
    "blond_hair": "Blond Hair",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-image",
        type=Path,
        default=Path("novelty_ablation/results/test_per_image.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("novelty_ablation/results/ablation_table.md"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("asyrp", "sae_only", "casl"),
        choices=tuple(METHOD_NAMES),
        help="Methods to include and their display order.",
    )
    return parser.parse_args()


def epr(group):
    values = []
    for classifier in ("resnet18", "mobilenetv2", "vgg16", "vit_b_16"):
        numerator = group[f"{classifier}_target_logit_abs"].mean()
        denominator = group[f"{classifier}_nontarget_logit_abs"].mean()
        values.append(numerator / max(denominator, 1e-8))
    return float(np.mean(values))


def metrics(group):
    return np.array(
        [
            group["target_delta"].mean(),
            group["nontarget_delta"].mean(),
            epr(group),
        ]
    )


def bootstrap_samples(group, iterations, rng):
    columns = ["target_delta", "nontarget_delta"]
    for classifier in ("resnet18", "mobilenetv2", "vgg16", "vit_b_16"):
        columns.extend(
            [
                f"{classifier}_target_logit_abs",
                f"{classifier}_nontarget_logit_abs",
            ]
        )
    values = group[columns].to_numpy(dtype=np.float64)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    classifier_eprs = [
        means[:, index] / np.maximum(means[:, index + 1], 1e-8)
        for index in range(2, 10, 2)
    ]
    return np.column_stack(
        [means[:, 0], means[:, 1], np.mean(classifier_eprs, axis=0)]
    )


def bootstrap(group, iterations, rng):
    estimates = bootstrap_samples(group, iterations, rng)
    return np.quantile(estimates, [0.025, 0.975], axis=0)


def fmt(value, interval):
    return f"{value:.3f} ({interval[0]:.3f}, {interval[1]:.3f})"


def main():
    args = parse_args()
    frame = pd.read_csv(args.per_image, dtype={"image_id": str})
    rng = np.random.default_rng(args.seed)
    results = {}

    for (concept, method), group in frame.groupby(["concept", "method"]):
        point = metrics(group)
        interval = bootstrap(group, args.bootstrap, rng)
        results[(concept, method)] = (point, interval, len(group))

    # Macro confidence intervals resample images independently within concept.
    for method in args.methods:
        groups = [
            frame[(frame["concept"] == concept) & (frame["method"] == method)]
            for concept in CONCEPT_NAMES
        ]
        point = np.mean([metrics(group) for group in groups], axis=0)
        estimates = np.mean(
            [
                bootstrap_samples(group, args.bootstrap, rng)
                for group in groups
            ],
            axis=0,
        )
        interval = np.quantile(estimates, [0.025, 0.975], axis=0)
        results[("macro", method)] = (point, interval, sum(len(g) for g in groups))

    lines = [
        "| Concept | Method | $\\Delta$target $\\uparrow$ | "
        "$\\Delta$non-target $\\downarrow$ | EPR $\\uparrow$ |",
        "|---|---|---:|---:|---:|",
    ]
    for concept in [*CONCEPT_NAMES, "macro"]:
        available = {
            method: results[(concept, method)]
            for method in args.methods
            if (concept, method) in results
        }
        if not available:
            continue
        points = {method: value[0] for method, value in available.items()}
        winners = [
            max(points, key=lambda method: points[method][0]),
            min(points, key=lambda method: points[method][1]),
            max(points, key=lambda method: points[method][2]),
        ]
        for method, (point, interval, _) in available.items():
            cells = []
            for index in range(3):
                cell = fmt(point[index], interval[:, index])
                if method == winners[index]:
                    cell = f"**{cell}**"
                cells.append(cell)
            label = CONCEPT_NAMES.get(concept, "Macro avg.")
            lines.append(
                f"| {label} | {METHOD_NAMES[method]} | "
                f"{cells[0]} | {cells[1]} | {cells[2]} |"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
