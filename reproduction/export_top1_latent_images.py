#!/usr/bin/env python3
"""
Randomly sample images, score them by a concept's top-k linear-probe-ranked SAE
latents, and copy the highest-scoring images into an output folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


REPO_ROOT = Path("/p/realai/zhenghao/LoRA_SAE")
SAE_MODULE_ROOT = REPO_ROOT / "Asyrp_SAE"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SAE_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAE_MODULE_ROOT))

from dit_preliminary_validation import collect_image_paths, extract_block_activations, load_dit_model, load_vae, set_seed  # type: ignore
from extract_concept_latent_indices import load_sae_checkpoint  # type: ignore


DEFAULT_IMAGE_DIR = REPO_ROOT / "CelebA-HQ" / "data1024"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_top1_images"
DEFAULT_PROBE_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_linear_probe"
DEFAULT_SAE_CKPT = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_sae" / "dit_sae_relu_lm8_l10.01.pt"


def load_probe_topk(probe_file: Path, top_k_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(probe_file.read_text())
    ranked = payload.get("top_ranked_latents", [])
    if not ranked:
        raise ValueError(f"No top_ranked_latents found in {probe_file}")

    subset = ranked[: min(top_k_features, len(ranked))]
    indices = torch.tensor([item["latent_index"] for item in subset], dtype=torch.long)
    weights = torch.tensor([item["weight"] for item in subset], dtype=torch.float32)
    return indices, weights


def score_images_by_probe_features(
    sae,
    pooled_activations: torch.Tensor,
    latent_indices: torch.Tensor,
    latent_weights: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        latents, _ = sae.encode(pooled_activations.to(device), t=None)
        selected = latents[:, latent_indices.to(device)]
        normalized_weights = latent_weights.to(device)
        denom = normalized_weights.abs().sum().clamp_min(1e-8)
        scores = (selected * normalized_weights.unsqueeze(0)).sum(dim=1) / denom
    return scores.float().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top images using top-k linear-probe-ranked SAE latents.")
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--dit-repo", type=Path, required=True)
    parser.add_argument("--dit-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-source", type=str, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--probe-file", type=Path, default=None, help="Defaults to <probe-dir>/<concept>_linear_probe.json")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--top-images", type=int, default=32)
    parser.add_argument("--top-k-features", type=int, default=5, help="Use the top-k linear-probe-ranked latents for scoring.")
    parser.add_argument("--seed", type=int, default=42)
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

    probe_file = args.probe_file or (args.probe_dir / f"{args.concept}_linear_probe.json")
    latent_indices, latent_weights = load_probe_topk(probe_file, args.top_k_features)
    print(f"Concept: {args.concept}")
    print(f"Using top-{latent_indices.numel()} probe features: {latent_indices.tolist()}")

    sampled_paths = collect_image_paths(args.image_dir, args.sample_size, args.seed)
    print(f"Random images sampled: {len(sampled_paths)}")

    vae = load_vae(args.vae_source, device)
    dit = load_dit_model(args.dit_repo, args.dit_checkpoint, args.image_size, device)
    sae = load_sae_checkpoint(args.sae_checkpoint, device)

    pooled_activations, used_paths, raw_shape = extract_block_activations(
        model=dit,
        vae=vae,
        image_paths=sampled_paths,
        block_index=args.block_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        timestep=args.timestep,
        class_label=args.class_label,
        device=device,
    )
    print(f"Raw block activation shape: {tuple(raw_shape)}")
    print(f"Pooled activation shape: {tuple(pooled_activations.shape)}")

    scores = score_images_by_probe_features(sae, pooled_activations, latent_indices, latent_weights, device)
    top_n = min(args.top_images, scores.numel())
    top_values, top_indices = torch.topk(scores, k=top_n)

    concept_dir = args.output_dir / f"{args.concept}_probe_top{latent_indices.numel()}_top{top_n}"
    concept_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for rank, (score, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
        src = Path(used_paths[idx])
        dst = concept_dir / f"{rank:02d}_score_{score:.4f}_{src.name}"
        shutil.copy2(src, dst)
        exported.append({"rank": rank, "score": score, "source": str(src), "copied_to": str(dst)})

    metadata = {
        "concept": args.concept,
        "seed": args.seed,
        "sample_size": len(used_paths),
        "top_images": top_n,
        "top_k_features": int(latent_indices.numel()),
        "probe_file": str(probe_file),
        "latent_indices": latent_indices.tolist(),
        "latent_weights": [float(v) for v in latent_weights.tolist()],
        "sae_checkpoint": str(args.sae_checkpoint),
        "raw_block_shape": raw_shape,
        "pooled_shape": list(pooled_activations.shape),
        "exported": exported,
    }
    meta_path = concept_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Copied top-{top_n} images to: {concept_dir}")
    print(f"Saved metadata to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
