#!/usr/bin/env python3
"""Evaluate equal-budget latent selection with targeted OS/MES."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CONCEPT_DIRECTORIES = {
    "Smiling": "smiling",
    "Young": "young",
    "Big_Nose": "big_nose",
    "Beards": "beards",
    "Blond_Hair": "blond_hair",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        type=Path,
        default=Path(
            "rebuttal_metrics/results/celeba_sae_omes/"
            "representations_t509_mean.npy"
        ),
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path("rebuttal_metrics/results/celeba_sae_omes/classes.csv"),
    )
    parser.add_argument(
        "--indices-root", type=Path, default=Path("sae")
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--inactive-threshold", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--random-seeds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rebuttal_metrics/results/celeba_sae_omes/selection"),
    )
    return parser.parse_args()


def read_classes(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    factors = [name for name in rows[0] if name != "image_id"]
    labels = np.asarray(
        [[int(row[factor]) for factor in factors] for row in rows],
        dtype=np.int8,
    )
    return factors, labels


def class_centroids(representation, labels):
    configurations, inverse = np.unique(labels, axis=0, return_inverse=True)
    centroids = np.zeros(
        (len(configurations), representation.shape[1]), dtype=np.float64
    )
    counts = np.bincount(inverse)
    for configuration_index in range(len(configurations)):
        centroids[configuration_index] = representation[
            inverse == configuration_index
        ].mean(axis=0, dtype=np.float64)
    return configurations, centroids, counts


def relation_matrix(configurations, centroids):
    n_dimensions = centroids.shape[1]
    relation = np.zeros((n_dimensions, configurations.shape[1]), dtype=np.float64)
    pair_counts = []
    for factor_index in range(configurations.shape[1]):
        groups = {}
        for index, configuration in enumerate(configurations):
            key = tuple(np.delete(configuration, factor_index).tolist())
            groups.setdefault(key, {})[int(configuration[factor_index])] = index
        pairs = [
            (values[0], values[1])
            for values in groups.values()
            if 0 in values and 1 in values
        ]
        if len(pairs) < 3:
            raise ValueError(
                f"Only {len(pairs)} exact pairs for factor {factor_index}"
            )
        x = centroids[[pair[0] for pair in pairs]]
        y = centroids[[pair[1] for pair in pairs]]
        x_centered = x - x.mean(axis=0)
        y_centered = y - y.mean(axis=0)
        denominator = np.sqrt(
            np.sum(x_centered**2, axis=0) * np.sum(y_centered**2, axis=0)
        )
        correlation = np.divide(
            np.sum(x_centered * y_centered, axis=0),
            denominator,
            out=np.zeros(n_dimensions, dtype=np.float64),
            where=denominator > 1e-12,
        )
        relation[:, factor_index] = 1.0 - np.abs(correlation)
        pair_counts.append(len(pairs))
    return relation, pair_counts


def weighted_average(values, weights):
    denominator = float(weights.sum())
    return float(np.sum(values * weights) / denominator) if denominator else 0.0


def targeted_scores(relation, selected, factor_index, alpha):
    selected_relation = relation[selected]
    ideal_factor = np.zeros(relation.shape[1])
    ideal_factor[factor_index] = 1.0
    os_per_dimension = 1.0 - np.mean(
        np.abs(selected_relation - ideal_factor[None, :]), axis=1
    )
    target_relation = selected_relation[:, factor_index]
    mes_per_dimension = np.empty(len(selected))
    for dimension in range(len(selected)):
        ideal_dimension = np.zeros(len(selected))
        ideal_dimension[dimension] = 1.0
        mes_per_dimension[dimension] = 1.0 - np.mean(
            np.abs(target_relation - ideal_dimension)
        )
    os_score = weighted_average(os_per_dimension, target_relation)
    mes_score = weighted_average(mes_per_dimension, target_relation)
    return os_score, mes_score, alpha * os_score + (1 - alpha) * mes_score


def load_rankings(root, factors):
    rankings = {}
    suffix = (
        "_LC_CelebA_HQ_P2_t999_ninv50_ngen50/"
        "top1024_indices_sae_model_ReLUNone_alpha32.0_scale128.pt"
    )
    for factor in factors:
        path = root / f"{CONCEPT_DIRECTORIES[factor]}{suffix}"
        rankings[factor] = torch.load(
            path, map_location="cpu", weights_only=False
        ).numpy()
    return rankings


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, rows):
    methods = ["Random active", "High-activation", "CASL"]
    labels = {
        "Random active": "Random active",
        "High-activation": "High activation",
        "CASL": "CASL top-k",
    }
    colors = {
        "Random active": "#7f7f7f",
        "High-activation": "#e69f00",
        "CASL": "#0072b2",
    }
    plt.figure(figsize=(5.5, 3.8), dpi=180)
    for method in methods:
        selected = [row for row in rows if row["selection"] == method]
        selected.sort(key=lambda row: int(row["k"]))
        plt.plot(
            [int(row["k"]) for row in selected],
            [float(row["OMES"]) for row in selected],
            marker="o",
            linewidth=2,
            label=labels[method],
            color=colors[method],
        )
    plt.xticks(sorted({int(row["k"]) for row in rows}))
    plt.xlabel("Latent budget per concept (k)")
    plt.ylabel("Targeted OMES ↑")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation = np.load(args.representation, mmap_mode="r")
    factors, labels = read_classes(args.classes)
    if len(representation) != len(labels):
        raise ValueError("Representation and class row counts differ")
    configurations, centroids, configuration_counts = class_centroids(
        representation, labels
    )
    relation, pair_counts = relation_matrix(configurations, centroids)

    scale = representation.std(axis=0, ddof=1, dtype=np.float64)
    active = np.flatnonzero(scale >= args.inactive_threshold)
    if len(active) < max(args.ks):
        raise ValueError(f"Only {len(active)} active dimensions")
    active_mask = np.zeros(representation.shape[1], dtype=bool)
    active_mask[active] = True
    mean_activation = representation.mean(axis=0, dtype=np.float64)
    activation_ranking = active[np.argsort(mean_activation[active])[::-1]]
    casl_rankings = load_rankings(args.indices_root, factors)
    casl_rankings = {
        factor: ranking[active_mask[ranking]] for factor, ranking in casl_rankings.items()
    }
    if any(len(ranking) < max(args.ks) for ranking in casl_rankings.values()):
        raise ValueError("A CASL ranking has too few active dimensions")

    factor_rows = []
    summary_rows = []
    rng = np.random.default_rng(args.seed)
    for k in args.ks:
        method_factor_scores = {"High-activation": [], "CASL": []}
        for factor_index, factor in enumerate(factors):
            selections = {
                "High-activation": activation_ranking[:k],
                "CASL": casl_rankings[factor][:k],
            }
            for method, selected in selections.items():
                os_score, mes_score, omes = targeted_scores(
                    relation, selected, factor_index, args.alpha
                )
                method_factor_scores[method].append((os_score, mes_score, omes))
                factor_rows.append(
                    {
                        "k": k,
                        "selection": method,
                        "concept": factor,
                        "OS": f"{os_score:.9f}",
                        "MES": f"{mes_score:.9f}",
                        "OMES": f"{omes:.9f}",
                        "n_repeats": 1,
                    }
                )

        random_macro = []
        random_by_factor = {factor: [] for factor in factors}
        for _ in range(args.random_seeds):
            current = []
            for factor_index, factor in enumerate(factors):
                selected = rng.choice(active, size=k, replace=False)
                score = targeted_scores(
                    relation, selected, factor_index, args.alpha
                )
                current.append(score)
                random_by_factor[factor].append(score)
            random_macro.append(np.mean(current, axis=0))
        random_macro = np.asarray(random_macro)
        for factor in factors:
            values = np.asarray(random_by_factor[factor])
            factor_rows.append(
                {
                    "k": k,
                    "selection": "Random active",
                    "concept": factor,
                    "OS": f"{values[:, 0].mean():.9f}",
                    "MES": f"{values[:, 1].mean():.9f}",
                    "OMES": f"{values[:, 2].mean():.9f}",
                    "n_repeats": args.random_seeds,
                }
            )

        for method, values in method_factor_scores.items():
            macro = np.mean(values, axis=0)
            summary_rows.append(
                {
                    "k": k,
                    "selection": method,
                    "OS": f"{macro[0]:.9f}",
                    "MES": f"{macro[1]:.9f}",
                    "OMES": f"{macro[2]:.9f}",
                    "OMES_ci_low": "",
                    "OMES_ci_high": "",
                    "n_repeats": 1,
                }
            )
        summary_rows.append(
            {
                "k": k,
                "selection": "Random active",
                "OS": f"{random_macro[:, 0].mean():.9f}",
                "MES": f"{random_macro[:, 1].mean():.9f}",
                "OMES": f"{random_macro[:, 2].mean():.9f}",
                "OMES_ci_low": f"{np.quantile(random_macro[:, 2], 0.025):.9f}",
                "OMES_ci_high": f"{np.quantile(random_macro[:, 2], 0.975):.9f}",
                "n_repeats": args.random_seeds,
            }
        )

    write_csv(args.output_dir / "selection_omes_per_concept.csv", factor_rows)
    write_csv(args.output_dir / "selection_omes_summary.csv", summary_rows)
    plot_summary(args.output_dir / "selection_omes_curve.png", summary_rows)
    audit = {
        "representation": str(args.representation.resolve()),
        "classes": str(args.classes.resolve()),
        "n_samples": int(len(representation)),
        "n_latents": int(representation.shape[1]),
        "n_active": int(len(active)),
        "inactive_threshold": args.inactive_threshold,
        "pooling": "average pooling in targeted OS/MES",
        "alpha": args.alpha,
        "factors": factors,
        "n_observed_factor_configurations": int(len(configurations)),
        "configuration_count_min": int(configuration_counts.min()),
        "configuration_count_max": int(configuration_counts.max()),
        "exact_centroid_pairs_per_factor": dict(zip(factors, pair_counts)),
        "random_seeds": args.random_seeds,
        "protocol": (
            "For each factor and k, compute official-style OS/MES on exactly k "
            "selected active rows of one shared full relation matrix; average "
            "factor scores. Class centroids marginalize identity nuisance."
        ),
    }
    with (args.output_dir / "selection_omes_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
