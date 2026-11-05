#!/usr/bin/env python3
"""Prepare an SAE-only control without the Stage-2 CLIP alignment network."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONCEPTS = {
    "smiling": ("Smiling", False),
    "young": ("Young", False),
    "big_nose": ("Big_Nose", False),
    # The extracted column named Beards stores CelebA's No_Beard attribute.
    "beards": ("Beards", True),
    "blond_hair": ("Blond_Hair", False),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representations",
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
        "--sae",
        type=Path,
        default=Path(
            "sae/CelebA_HQ_P2/"
            "sae_model_ReLUNone_alpha32.0_scale128.pth"
        ),
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--selection-images", type=int, default=1000)
    parser.add_argument(
        "--experiment-prefix",
        default="sae_only_v2",
        help="Separate experiment namespace so earlier controls are preserved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("novelty_ablation/artifacts/sae_only"),
    )
    return parser.parse_args()


def read_labels(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        name: np.asarray([int(row[name]) for row in rows], dtype=np.int8)
        for name, _ in CONCEPTS.values()
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation = np.load(args.representations, mmap_mode="r")
    n = min(args.selection_images, len(representation))
    labels = read_labels(args.classes)
    values = np.asarray(representation[:n], dtype=np.float32)

    metadata = {
        "selection_rule": (
            "largest positive standardized target-vs-nontarget mean activation; "
            "label-assisted SAE unit selection"
        ),
        "stage2_supervision": False,
        "representations": str(args.representations.resolve()),
        "sae": str(args.sae.resolve()),
        "selection_images": n,
        "top_k": args.top_k,
        "concepts": {},
    }

    for concept, (label_name, invert_label) in CONCEPTS.items():
        y = labels[label_name][:n]
        if invert_label:
            y = 1 - y
        positive = values[y == 1]
        negative = values[y == 0]
        mean_delta = positive.mean(axis=0) - negative.mean(axis=0)
        pooled_std = np.sqrt(
            0.5 * (positive.var(axis=0) + negative.var(axis=0)) + 1e-8
        )
        score = mean_delta / pooled_std
        ranking = np.argsort(score)[::-1].astype(np.int64)
        selected = ranking[: args.top_k]

        exp_id = (
            f"{args.experiment_prefix}_{concept}_LC_CelebA_HQ_P2_"
            "t999_ninv50_ngen50"
        )
        index_dir = ROOT / "sae" / exp_id
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = (
            index_dir
            / "top1024_indices_sae_model_ReLUNone_alpha32.0_scale128.pt"
        )
        torch.save(torch.from_numpy(ranking[:1024].copy()), index_path)

        record = {
            "label": label_name,
            "label_inverted": invert_label,
            "selected_indices": selected.tolist(),
            "signed_scores": score[selected].astype(float).tolist(),
            "positive_count": int((y == 1).sum()),
            "negative_count": int((y == 0).sum()),
            "index_path": str(index_path.resolve()),
            "checkpoint_path": None,
            "decoder": "shared frozen SAE decoder",
        }
        metadata["concepts"][concept] = record
        print(concept, json.dumps(record))

    metadata_path = args.output_dir / "selection.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
