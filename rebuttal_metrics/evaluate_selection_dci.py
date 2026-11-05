#!/usr/bin/env python3
"""Equal-budget DCI evaluation for SAE latent-selection rules.

Each selection receives exactly k unique active dimensions per factor, hence
K*k dimensions in total. DCI is computed from gradient-boosted classifiers,
following the implementation bundled with transfer_disentanglement.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score


CONCEPT_DIRECTORIES = {
    "Smiling": "smiling",
    "Young": "young",
    "Big_Nose": "big_nose",
    "Beards": "beards",
    "Blond_Hair": "blond_hair",
}


def parse_args():
    parser = argparse.ArgumentParser()
    root = Path("rebuttal_metrics/results/celeba_sae_omes_independent")
    parser.add_argument(
        "--representation",
        type=Path,
        default=root / "representations_t509_mean.npy",
    )
    parser.add_argument("--classes", type=Path, default=root / "classes.csv")
    parser.add_argument("--indices-root", type=Path, default=Path("sae"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--inactive-threshold", type=float, default=0.01)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "dci_selection",
    )
    return parser.parse_args()


def load_rankings(root, factors):
    suffix = (
        "_LC_CelebA_HQ_P2_t999_ninv50_ngen50/"
        "top1024_indices_sae_model_ReLUNone_alpha32.0_scale128.pt"
    )
    import torch

    return {
        factor: torch.load(
            root / f"{CONCEPT_DIRECTORIES[factor]}{suffix}",
            map_location="cpu",
            weights_only=False,
        ).numpy()
        for factor in factors
    }


def balanced_unique_selection(rankings, factors, active_mask, k):
    """Give each factor k active dimensions, advancing past all duplicates."""
    selected = []
    selected_set = set()
    contributions = {}
    for factor in factors:
        contribution = []
        for index in rankings[factor]:
            index = int(index)
            if not active_mask[index] or index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            contribution.append(index)
            if len(contribution) == k:
                break
        if len(contribution) != k:
            raise ValueError(f"{factor} has fewer than {k} unique active latents")
        contributions[factor] = contribution
    return np.asarray(selected, dtype=np.int64), contributions


def dci_scores(x_train, y_train, x_test, y_test, seed):
    n_codes = x_train.shape[1]
    n_factors = y_train.shape[1]
    importance = np.zeros((n_codes, n_factors), dtype=np.float64)
    train_accuracy = []
    test_accuracy = []
    test_balanced_accuracy = []
    per_factor = {}
    for factor_index in range(n_factors):
        model = GradientBoostingClassifier(random_state=seed)
        model.fit(x_train, y_train[:, factor_index])
        importance[:, factor_index] = np.abs(model.feature_importances_)
        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)
        train_value = float(np.mean(train_prediction == y_train[:, factor_index]))
        test_value = float(np.mean(test_prediction == y_test[:, factor_index]))
        balanced_value = float(
            balanced_accuracy_score(y_test[:, factor_index], test_prediction)
        )
        train_accuracy.append(train_value)
        test_accuracy.append(test_value)
        test_balanced_accuracy.append(balanced_value)
        per_factor[factor_index] = {
            "train_accuracy": train_value,
            "test_accuracy": test_value,
            "test_balanced_accuracy": balanced_value,
        }

    safe = importance + 1e-11
    per_code_disentanglement = 1.0 - entropy(safe.T, base=n_factors)
    code_weight = importance.sum(axis=1)
    if code_weight.sum() == 0:
        code_weight = np.ones_like(code_weight)
    code_weight /= code_weight.sum()
    disentanglement = float(per_code_disentanglement @ code_weight)

    per_factor_completeness = 1.0 - entropy(safe, base=n_codes)
    factor_weight = importance.sum(axis=0)
    if factor_weight.sum() == 0:
        factor_weight = np.ones_like(factor_weight)
    factor_weight /= factor_weight.sum()
    completeness = float(per_factor_completeness @ factor_weight)
    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness_train": float(np.mean(train_accuracy)),
        "informativeness_test": float(np.mean(test_accuracy)),
        "balanced_informativeness_test": float(
            np.mean(test_balanced_accuracy)
        ),
        "per_factor": per_factor,
        "importance": importance,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation = np.load(args.representation, mmap_mode="r")
    classes = pd.read_csv(args.classes)
    factors = [column for column in classes.columns if column != "image_id"]
    labels = classes[factors].to_numpy(dtype=np.int8)
    if len(representation) != len(labels):
        raise ValueError("Representation and labels have different lengths")

    scale = representation.std(axis=0, ddof=1, dtype=np.float64)
    active = np.flatnonzero(scale >= args.inactive_threshold)
    active_mask = np.zeros(representation.shape[1], dtype=bool)
    active_mask[active] = True
    mean_activation = representation.mean(axis=0, dtype=np.float64)
    activation_ranking = active[np.argsort(mean_activation[active])[::-1]]
    rankings = load_rankings(args.indices_root, factors)

    split = int(np.ceil(len(labels) * (2.0 / 3.0)))
    y_train, y_test = labels[:split], labels[split:]
    rng = np.random.default_rng(args.seed)
    rows = []
    selection_audit = {}
    for k in args.ks:
        total_budget = len(factors) * k
        casl, contributions = balanced_unique_selection(
            rankings, factors, active_mask, k
        )
        selections = {
            "High-activation": [activation_ranking[:total_budget]],
            "CASL": [casl],
            "Random active": [
                rng.choice(active, size=total_budget, replace=False)
                for _ in range(args.random_repeats)
            ],
        }
        selection_audit[str(k)] = {
            "total_budget": total_budget,
            "casl_indices": casl.tolist(),
            "casl_contribution_indices": contributions,
            "high_activation_indices": activation_ranking[
                :total_budget
            ].tolist(),
        }
        tasks = []
        for selection, repeats in selections.items():
            for repeat, indices in enumerate(repeats):
                selected = np.asarray(
                    representation[:, indices], dtype=np.float32
                )
                tasks.append((selection, repeat, selected))
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(dci_scores)(
                selected[:split],
                y_train,
                selected[split:],
                y_test,
                args.seed + repeat,
            )
            for _, repeat, selected in tasks
        )
        for (selection, repeat, _), result in zip(tasks, results):
            row = {
                "k_per_factor": k,
                "total_latents": total_budget,
                "selection": selection,
                "repeat": repeat,
                "disentanglement": result["disentanglement"],
                "completeness": result["completeness"],
                "informativeness_test": result["informativeness_test"],
                "balanced_informativeness_test": result[
                    "balanced_informativeness_test"
                ],
                "informativeness_train": result["informativeness_train"],
            }
            for factor_index, factor in enumerate(factors):
                row[f"{factor}_accuracy"] = result["per_factor"][
                    factor_index
                ]["test_accuracy"]
                row[f"{factor}_balanced_accuracy"] = result["per_factor"][
                    factor_index
                ]["test_balanced_accuracy"]
            rows.append(row)
            np.save(
                args.output_dir
                / f"importance_k{k}_{selection.lower().replace(' ', '_')}"
                f"_r{repeat}.npy",
                result["importance"],
            )
            print(
                f"k={k} {selection} r={repeat}: "
                f"D={result['disentanglement']:.4f} "
                f"C={result['completeness']:.4f} "
                f"I={result['informativeness_test']:.4f}"
            )

    write_csv(args.output_dir / "dci_repeats.csv", rows)
    frame = pd.DataFrame(rows)
    summary_rows = []
    for (k, selection), group in frame.groupby(
        ["k_per_factor", "selection"], sort=True
    ):
        output = {
            "k_per_factor": k,
            "total_latents": int(group.total_latents.iloc[0]),
            "selection": selection,
            "n_repeats": len(group),
        }
        for metric in [
            "disentanglement",
            "completeness",
            "informativeness_test",
            "balanced_informativeness_test",
        ]:
            output[metric] = group[metric].mean()
            output[f"{metric}_q025"] = group[metric].quantile(0.025)
            output[f"{metric}_q975"] = group[metric].quantile(0.975)
        summary_rows.append(output)
    write_csv(args.output_dir / "dci_summary.csv", summary_rows)
    audit = {
        "representation": str(args.representation.resolve()),
        "classes": str(args.classes.resolve()),
        "n_samples": len(labels),
        "n_latents": int(representation.shape[1]),
        "n_active": len(active),
        "inactive_threshold": args.inactive_threshold,
        "factors": factors,
        "split": {
            "protocol": "first two-thirds train, final one-third test",
            "n_train": split,
            "n_test": len(labels) - split,
        },
        "regressor": "sklearn GradientBoostingClassifier",
        "importance": "absolute feature_importances_",
        "informativeness": "mean test classification accuracy",
        "balanced_informativeness_auxiliary": (
            "mean test balanced accuracy for imbalanced binary factors"
        ),
        "random_repeats": args.random_repeats,
        "selection_protocol": (
            "Exactly k unique active dimensions contributed per factor; "
            "all rules receive K*k total dimensions. CASL advances within "
            "each factor ranking to skip dimensions already selected."
        ),
        "selections": selection_audit,
    }
    with (args.output_dir / "dci_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
