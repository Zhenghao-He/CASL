#!/usr/bin/env python3
"""
Train a linear probe on a concept's saved top-k SAE latent features.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/p/realai/zhenghao/LoRA_SAE")
SAE_MODULE_ROOT = REPO_ROOT / "Asyrp_SAE"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SAE_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAE_MODULE_ROOT))

from dit_preliminary_validation import extract_block_activations, load_dit_model, load_vae, set_seed  # type: ignore
from extract_concept_latent_indices import load_sae_checkpoint, parse_attr_file  # type: ignore


DEFAULT_ATTR_FILE = REPO_ROOT / "CelebA-HQ" / "list_attr_celeba_shift.txt"
DEFAULT_IMAGE_DIR = REPO_ROOT / "CelebA-HQ" / "data1024"
DEFAULT_INDEX_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_concept_latents"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_linear_probe"
DEFAULT_SAE_CKPT = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_sae" / "dit_sae_relu_lm8_l10.01.pt"


def sample_balanced_images(
    image_dir: Path,
    attr_file: Path,
    concept: str,
    positive_limit: int,
    negative_limit: int,
    seed: int,
) -> Tuple[List[Path], List[int]]:
    labeled_rows = parse_attr_file(attr_file, concept)
    positives = [image_dir / filename for filename, label in labeled_rows if label > 0 and (image_dir / filename).exists()]
    negatives = [image_dir / filename for filename, label in labeled_rows if label < 0 and (image_dir / filename).exists()]

    if not positives:
        raise FileNotFoundError(f"No positive samples found for concept '{concept}'")
    if not negatives:
        raise FileNotFoundError(f"No negative samples found for concept '{concept}'")

    rng = random.Random(seed)
    if positive_limit > 0 and len(positives) > positive_limit:
        positives = rng.sample(positives, positive_limit)
    if negative_limit > 0 and len(negatives) > negative_limit:
        negatives = rng.sample(negatives, negative_limit)

    image_paths = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    order = list(range(len(image_paths)))
    rng.shuffle(order)
    shuffled_paths = [image_paths[i] for i in order]
    shuffled_labels = [labels[i] for i in order]
    return shuffled_paths, shuffled_labels


def load_index_subset(index_file: Path) -> torch.Tensor:
    payload = torch.load(index_file, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        indices = payload.long().flatten()
    elif isinstance(payload, (list, tuple)):
        indices = torch.tensor(payload, dtype=torch.long)
    else:
        raise TypeError(f"Unsupported index file format: {index_file}")
    if indices.numel() == 0:
        raise ValueError(f"No indices found in {index_file}")
    return indices


def encode_selected_latents(
    sae,
    activations: torch.Tensor,
    selected_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        latents, _ = sae.encode(activations.to(device), t=None)
        features = latents[:, selected_indices.to(device)]
    return features.float().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear probe on top-k SAE latents for a concept.")
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--dit-repo", type=Path, required=True)
    parser.add_argument("--dit-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-source", type=str, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--index-file", type=Path, default=None, help="Defaults to <index-dir>/<concept>_top100_indices.pt")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--attr-file", type=Path, default=DEFAULT_ATTR_FILE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positive-limit", type=int, default=1000)
    parser.add_argument("--negative-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-k-features", type=int, default=0, help="If > 0, keep only the first k indices from the saved top-100 list.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-index", type=int, default=20)
    parser.add_argument("--timestep", type=int, default=250)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    index_file = args.index_file or (args.index_dir / f"{args.concept}_top100_indices.pt")
    selected_indices = load_index_subset(index_file)
    if args.top_k_features > 0:
        selected_indices = selected_indices[: min(args.top_k_features, selected_indices.numel())]

    image_paths, labels = sample_balanced_images(
        image_dir=args.image_dir,
        attr_file=args.attr_file,
        concept=args.concept,
        positive_limit=args.positive_limit,
        negative_limit=args.negative_limit,
        seed=args.seed,
    )
    print(f"Concept: {args.concept}")
    print(f"Selected images: total={len(image_paths)} pos={sum(labels)} neg={len(labels) - sum(labels)}")
    print(f"Feature subset size: {selected_indices.numel()}")

    vae = load_vae(args.vae_source, device)
    dit = load_dit_model(args.dit_repo, args.dit_checkpoint, args.image_size, device)
    sae = load_sae_checkpoint(args.sae_checkpoint, device)

    pooled_activations, used_paths, raw_shape = extract_block_activations(
        model=dit,
        vae=vae,
        image_paths=image_paths,
        block_index=args.block_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        timestep=args.timestep,
        class_label=args.class_label,
        device=device,
    )
    print(f"Raw block activation shape: {tuple(raw_shape)}")
    print(f"Pooled activation shape: {tuple(pooled_activations.shape)}")

    features = encode_selected_latents(sae, pooled_activations, selected_indices, device)
    y = torch.tensor(labels, dtype=torch.long).numpy()
    X = features.numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=args.seed)
    clf.fit(x_train_scaled, y_train)

    y_pred = clf.predict(x_test_scaled)
    y_score = clf.decision_function(x_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_score))

    weights = clf.coef_[0]
    ranked = sorted(
        [
            {
                "rank": rank + 1,
                "subset_position": int(pos),
                "latent_index": int(selected_indices[pos].item()),
                "weight": float(weights[pos]),
                "abs_weight": float(abs(weights[pos])),
            }
            for rank, pos in enumerate(sorted(range(len(weights)), key=lambda i: abs(weights[i]), reverse=True))
        ],
        key=lambda item: item["rank"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{selected_indices.numel()}" if args.top_k_features > 0 else ""
    stem = f"{args.concept}_linear_probe{suffix}"
    json_path = args.output_dir / f"{stem}.json"
    pt_path = args.output_dir / f"{stem}.pt"

    torch.save(
        {
            "selected_indices": selected_indices,
            "weights": torch.tensor(weights, dtype=torch.float32),
            "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
            "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
            "accuracy": acc,
            "auc": auc,
        },
        pt_path,
    )

    payload = {
        "concept": args.concept,
        "seed": args.seed,
        "num_samples": len(image_paths),
        "num_positive": int(sum(labels)),
        "num_negative": int(len(labels) - sum(labels)),
        "raw_block_shape": raw_shape,
        "pooled_shape": list(pooled_activations.shape),
        "feature_shape": list(features.shape),
        "top_k_features": int(selected_indices.numel()),
        "test_size": args.test_size,
        "accuracy": acc,
        "auc": auc,
        "index_file": str(index_file),
        "sae_checkpoint": str(args.sae_checkpoint),
        "top_ranked_latents": ranked[:20],
        "sampled_paths": used_paths[: min(20, len(used_paths))],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Top weighted latent indices: {[item['latent_index'] for item in ranked[:10]]}")
    print(f"Saved probe weights to: {pt_path}")
    print(f"Saved probe report to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
