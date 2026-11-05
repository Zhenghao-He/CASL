#!/usr/bin/env python3
"""Cross-attribute linear probes for CASL-selected and random SAE latents."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


CONCEPT_DIRECTORIES = {
    "Smiling": "smiling",
    "Young": "young",
    "Big_Nose": "big_nose",
    "Beards": "beards",
    "Blond_Hair": "blond_hair",
}


def parse_args():
    root = Path("rebuttal_metrics/results/celeba_sae_omes_independent")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        type=Path,
        default=root / "representations_t509_mean.npy",
    )
    parser.add_argument(
        "--active-indices",
        type=Path,
        help="Global indices for axis 2 of a spatial active-only cache.",
    )
    parser.add_argument("--classes", type=Path, default=root / "classes.csv")
    parser.add_argument("--indices-root", type=Path, default=Path("sae"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--inactive-threshold", type=float, default=0.01)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "cross_attribute_probe"
    )
    return parser.parse_args()


def load_rankings(root, concepts):
    suffix = (
        "_LC_CelebA_HQ_P2_t999_ninv50_ngen50/"
        "top1024_indices_sae_model_ReLUNone_alpha32.0_scale128.pt"
    )
    return {
        concept: torch.load(
            root / f"{CONCEPT_DIRECTORIES[concept]}{suffix}",
            map_location="cpu",
            weights_only=False,
        )
        .numpy()
        .astype(np.int64)
        for concept in concepts
    }


def probe_one(
    representation,
    labels,
    train_indices,
    test_indices,
    global_indices,
    columns,
    concepts,
    selection,
    selected_for,
    k,
    repeat,
    seed,
):
    if representation.ndim == 2:
        features = np.asarray(representation[:, columns], dtype=np.float32)
    else:
        features = np.asarray(
            representation[:, :, columns], dtype=np.float32
        ).reshape(len(representation), -1)
    x_train, x_test = features[train_indices], features[test_indices]
    rows = []
    for factor_index, predicted_attribute in enumerate(concepts):
        y_train = labels[train_indices, factor_index]
        y_test = labels[test_indices, factor_index]
        classifier = LinearSVC(
            dual="auto", random_state=seed, max_iter=20000
        )
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
        score = classifier.decision_function(x_test)
        rows.append(
            {
                "selection": selection,
                "repeat": repeat,
                "k": k,
                "selected_for": selected_for,
                "predicted_attribute": predicted_attribute,
                "is_target": int(selected_for == predicted_attribute),
                "accuracy": float(accuracy_score(y_test, prediction)),
                "auroc": float(roc_auc_score(y_test, score)),
                "indices": ";".join(map(str, global_indices)),
            }
        )
    return rows


def aggregate_per_set(frame):
    rows = []
    keys = ["selection", "repeat", "k", "selected_for"]
    for key, group in frame.groupby(keys, sort=False):
        target = group[group.is_target == 1].iloc[0]
        off = group[group.is_target == 0]
        rows.append(
            {
                "selection": key[0],
                "repeat": int(key[1]),
                "k": int(key[2]),
                "selected_for": key[3],
                "target_accuracy": target.accuracy,
                "target_auroc": target.auroc,
                "non_target_accuracy": off.accuracy.mean(),
                "non_target_auroc": off.auroc.mean(),
                "accuracy_gap": target.accuracy - off.accuracy.mean(),
                "auroc_gap": target.auroc - off.auroc.mean(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_macro(per_set):
    metrics = [
        "target_accuracy",
        "target_auroc",
        "non_target_accuracy",
        "non_target_auroc",
        "accuracy_gap",
        "auroc_gap",
    ]
    repeats = (
        per_set.groupby(["selection", "repeat", "k"], as_index=False)[metrics]
        .mean()
    )
    rows = []
    for (selection, k), group in repeats.groupby(
        ["selection", "k"], sort=False
    ):
        row = {
            "selection": selection,
            "k": int(k),
            "n_repeats": len(group),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[metric] = values.mean()
            row[f"{metric}_ci_low"] = np.quantile(values, 0.025)
            row[f"{metric}_ci_high"] = np.quantile(values, 0.975)
        rows.append(row)
    return repeats, pd.DataFrame(rows)


def write_markdown(path, macro):
    table = macro.copy()
    table["_order"] = table.selection.map({"CASL": 0, "Random active": 1})
    table = table.sort_values(["k", "_order"])
    lines = [
        "|Selection/$k$|Target Acc. $\\uparrow$|Target AUROC $\\uparrow$|"
        "Non-target Acc. $\\downarrow$|Non-target AUROC $\\downarrow$|"
        "AUROC gap $\\uparrow$|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"|{row.selection}/{row.k}|{row.target_accuracy:.3f}|"
            f"{row.target_auroc:.3f}|{row.non_target_accuracy:.3f}|"
            f"{row.non_target_auroc:.3f}|{row.auroc_gap:.3f}|"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation = np.load(args.representation, mmap_mode="r")
    classes = pd.read_csv(args.classes)
    concepts = list(CONCEPT_DIRECTORIES)
    if list(classes.columns[1:]) != concepts:
        raise ValueError(f"Unexpected class order: {list(classes.columns[1:])}")
    labels = classes[concepts].to_numpy(dtype=np.int8)
    if len(representation) != len(labels):
        raise ValueError("Representation and label counts differ")

    if representation.ndim == 2:
        scale = representation.std(axis=0, ddof=1, dtype=np.float64)
        active = np.flatnonzero(scale >= args.inactive_threshold)
        n_latents = representation.shape[1]
        global_to_column = {int(index): int(index) for index in active}
        pooling = "spatial mean"
    elif representation.ndim == 3 and args.active_indices:
        active = np.load(args.active_indices).astype(np.int64)
        if representation.shape[2] != len(active):
            raise ValueError("Spatial representation and active indices differ")
        # This cache stores selected columns from the paper's 65,536-unit SAE.
        n_latents = 65_536
        global_to_column = {
            int(index): column for column, index in enumerate(active)
        }
        pooling = "spatial flatten"
    else:
        raise ValueError("3-D representation requires --active-indices")
    active_mask = np.zeros(n_latents, dtype=bool)
    active_mask[active] = True

    all_indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )
    rankings = load_rankings(args.indices_root, concepts)
    rng = np.random.default_rng(args.seed)
    tasks, selection_audit = [], {}
    for k in args.ks:
        selection_audit[str(k)] = {"CASL": {}, "Random active": {}}
        for selected_for in concepts:
            casl = rankings[selected_for][:k]
            if not np.all(active_mask[casl]):
                raise ValueError(
                    f"CASL {selected_for}/k={k} contains inactive latents"
                )
            selection_audit[str(k)]["CASL"][selected_for] = casl.tolist()
            tasks.append(
                (
                    "CASL",
                    selected_for,
                    k,
                    0,
                    casl,
                    np.asarray([global_to_column[int(i)] for i in casl]),
                )
            )
            random_sets = [
                rng.choice(active, size=k, replace=False)
                for _ in range(args.random_repeats)
            ]
            selection_audit[str(k)]["Random active"][selected_for] = [
                indices.tolist() for indices in random_sets
            ]
            tasks.extend(
                (
                    "Random active",
                    selected_for,
                    k,
                    repeat,
                    indices,
                    np.asarray(
                        [global_to_column[int(i)] for i in indices]
                    ),
                )
                for repeat, indices in enumerate(random_sets)
            )

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(probe_one)(
            representation,
            labels,
            train_indices,
            test_indices,
            global_indices,
            columns,
            concepts,
            selection,
            selected_for,
            k,
            repeat,
            args.seed + repeat,
        )
        for (
            selection,
            selected_for,
            k,
            repeat,
            global_indices,
            columns,
        ) in tasks
    )
    frame = pd.DataFrame([row for result in results for row in result])
    per_set = aggregate_per_set(frame)
    macro_repeats, macro = aggregate_macro(per_set)
    frame.to_csv(args.output_dir / "cross_attribute_per_probe.csv", index=False)
    per_set.to_csv(args.output_dir / "cross_attribute_per_set.csv", index=False)
    macro_repeats.to_csv(
        args.output_dir / "cross_attribute_macro_repeats.csv", index=False
    )
    macro.to_csv(args.output_dir / "cross_attribute_macro.csv", index=False)
    write_markdown(args.output_dir / "CROSS_ATTRIBUTE_TABLE.md", macro)
    audit = {
        "representation": str(args.representation.resolve()),
        "active_indices": (
            str(args.active_indices.resolve()) if args.active_indices else None
        ),
        "classes": str(args.classes.resolve()),
        "n_samples": len(labels),
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "split_seed": args.seed,
        "timestep": 509,
        "pooling": pooling,
        "n_latents": n_latents,
        "n_active": len(active),
        "inactive_threshold": args.inactive_threshold,
        "concepts": concepts,
        "ks": args.ks,
        "random_repeats": args.random_repeats,
        "classifier": "LinearSVC(dual='auto', max_iter=20000)",
        "protocol": (
            "The identical fixed features selected for one concept predict "
            "all five attributes; random uses matched active-latent sets."
        ),
        "selection_indices": selection_audit,
    }
    with (args.output_dir / "cross_attribute_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
