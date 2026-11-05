#!/usr/bin/env python3
"""Equal-budget OS/MES evaluation from independent paired factor probes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_selected_omes import plot_summary, targeted_scores, write_csv


DOMAIN_CONCEPTS = {
    "church": {
        "Gothic": "church_gothic",
        "Red_Brick": "church_red_brick",
        "Snowy": "church_snowy_church",
        "Wooden": "church_wooden_house",
    },
    "dog": {
        "Angry": "dog_angry",
        "Sleepy": "dog_sleepy",
        "Smiling": "dog_smiling",
        "Puppy_like": "dog_puppy",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAIN_CONCEPTS, required=True)
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--indices-root", type=Path, default=Path("sae"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--inactive-threshold", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--random-seeds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relation_matrix(pairs, concepts):
    n_dimensions = pairs[f"{concepts[0]}__source"].shape[1]
    relation = np.empty((n_dimensions, len(concepts)), dtype=np.float64)
    for factor_index, concept in enumerate(concepts):
        source = pairs[f"{concept}__source"].astype(np.float64)
        edited = pairs[f"{concept}__edit"].astype(np.float64)
        source -= source.mean(axis=0)
        edited -= edited.mean(axis=0)
        denominator = np.sqrt(
            np.sum(source**2, axis=0) * np.sum(edited**2, axis=0)
        )
        correlation = np.divide(
            np.sum(source * edited, axis=0),
            denominator,
            out=np.zeros(n_dimensions, dtype=np.float64),
            where=denominator > 1e-12,
        )
        relation[:, factor_index] = 1.0 - np.abs(correlation)
    return relation


def load_rankings(root, domain, concepts):
    rankings = {}
    category = "church_outdoor" if domain == "church" else "dog"
    suffix = (
        f"_LC_{category}_t999_ninv50_ngen50/"
        "top1024_indices_sae_model_ReLUNone_alpha32.0_scale128.pt"
    )
    for concept in concepts:
        path = root / f"{DOMAIN_CONCEPTS[domain][concept]}{suffix}"
        rankings[concept] = torch.load(
            path, map_location="cpu", weights_only=False
        ).numpy()
    return rankings


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.representation)
    concepts = list(DOMAIN_CONCEPTS[args.domain])
    pairs = {key: loaded[key] for key in loaded.files}
    relation = relation_matrix(pairs, concepts)
    source_keys = sorted(key for key in pairs if key.endswith("__source"))
    reference_source = pairs[source_keys[0]]
    if any(
        not np.array_equal(reference_source, pairs[key])
        for key in source_keys[1:]
    ):
        raise ValueError("Source representations differ across factor probes")
    # Selection heuristics must not inspect the target edits. The shared
    # unedited source representation defines both activity and the common
    # active-latent pool, matching the CelebA protocol.
    activity_representation = reference_source
    scale = activity_representation.std(axis=0, ddof=1, dtype=np.float64)
    active = np.flatnonzero(scale >= args.inactive_threshold)
    if len(active) < max(args.ks):
        raise ValueError(f"Only {len(active)} active dimensions")
    active_mask = np.zeros(activity_representation.shape[1], dtype=bool)
    active_mask[active] = True
    mean_activation = activity_representation.mean(axis=0, dtype=np.float64)
    activation_ranking = active[np.argsort(mean_activation[active])[::-1]]
    casl_rankings = load_rankings(args.indices_root, args.domain, concepts)
    casl_rankings = {
        concept: ranking[active_mask[ranking]]
        for concept, ranking in casl_rankings.items()
    }
    if any(len(ranking) < max(args.ks) for ranking in casl_rankings.values()):
        raise ValueError("A CASL ranking has too few active dimensions")

    factor_rows = []
    summary_rows = []
    rng = np.random.default_rng(args.seed)
    for k in args.ks:
        method_scores = {"High-activation": [], "CASL": []}
        for factor_index, concept in enumerate(concepts):
            for method, selected in {
                "High-activation": activation_ranking[:k],
                "CASL": casl_rankings[concept][:k],
            }.items():
                score = targeted_scores(
                    relation, selected, factor_index, args.alpha
                )
                method_scores[method].append(score)
                factor_rows.append(
                    {
                        "k": k,
                        "selection": method,
                        "concept": concept,
                        "OS": f"{score[0]:.9f}",
                        "MES": f"{score[1]:.9f}",
                        "OMES": f"{score[2]:.9f}",
                        "n_repeats": 1,
                    }
                )

        random_macro = []
        random_by_factor = {concept: [] for concept in concepts}
        for _ in range(args.random_seeds):
            current = []
            for factor_index, concept in enumerate(concepts):
                selected = rng.choice(active, size=k, replace=False)
                score = targeted_scores(
                    relation, selected, factor_index, args.alpha
                )
                current.append(score)
                random_by_factor[concept].append(score)
            random_macro.append(np.mean(current, axis=0))
        random_macro = np.asarray(random_macro)
        for concept in concepts:
            values = np.asarray(random_by_factor[concept])
            factor_rows.append(
                {
                    "k": k,
                    "selection": "Random active",
                    "concept": concept,
                    "OS": f"{values[:, 0].mean():.9f}",
                    "MES": f"{values[:, 1].mean():.9f}",
                    "OMES": f"{values[:, 2].mean():.9f}",
                    "n_repeats": args.random_seeds,
                }
            )
        for method, values in method_scores.items():
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
        "domain": args.domain,
        "representation": str(args.representation.resolve()),
        "probe_generator": "official Asyrp source/edit pairs",
        "n_pairs_per_factor": {
            concept: int(len(pairs[f"{concept}__source"])) for concept in concepts
        },
        "n_latents": int(activity_representation.shape[1]),
        "n_active": int(len(active)),
        "activity_reference": "shared unedited source representation only",
        "inactive_threshold": args.inactive_threshold,
        "alpha": args.alpha,
        "concepts": concepts,
        "random_seeds": args.random_seeds,
        "protocol": (
            "Selection-conditioned OMES on a shared factor-latent relation "
            "matrix; exactly k active dimensions per concept and selection rule."
        ),
    }
    with (args.output_dir / "selection_omes_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
