#!/usr/bin/env python3
"""
Sample concept-positive CelebA-HQ images, extract DiT activations, run a trained
SAE, and save the top-k latent indices with highest mean activation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch


REPO_ROOT = Path("/p/realai/zhenghao/LoRA_SAE")
SAE_MODULE_ROOT = REPO_ROOT / "Asyrp_SAE"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SAE_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAE_MODULE_ROOT))

from dit_preliminary_validation import extract_block_activations, load_dit_model, load_vae, set_seed  # type: ignore
from models.sae.sae import Autoencoder  # type: ignore


DEFAULT_ATTR_FILE = REPO_ROOT / "CelebA-HQ" / "list_attr_celeba_shift.txt"
DEFAULT_IMAGE_DIR = REPO_ROOT / "CelebA-HQ" / "data1024"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_concept_latents"
DEFAULT_SAE_CKPT = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_sae" / "dit_sae_relu_lm8_l10.01.pt"


def parse_attr_file(attr_file: Path, concept: str) -> List[Tuple[str, int]]:
    lines = attr_file.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Unexpected attr file format: {attr_file}")

    header = lines[1].split()
    if concept not in header:
        raise ValueError(f"Concept '{concept}' not found in attr header.")
    concept_idx = header.index(concept)

    labeled_rows: List[Tuple[str, int]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) != len(header) + 1:
            continue
        filename = parts[0]
        label = int(parts[concept_idx + 1])
        labeled_rows.append((filename, label))
    return labeled_rows


def sample_positive_images(
    image_dir: Path,
    attr_file: Path,
    concept: str,
    limit: int,
    seed: int,
) -> List[Path]:
    labeled_rows = parse_attr_file(attr_file, concept)
    positives = [image_dir / filename for filename, label in labeled_rows if label > 0 and (image_dir / filename).exists()]
    if not positives:
        raise FileNotFoundError(f"No positive samples found for concept '{concept}' in {image_dir}")

    if limit <= 0 or len(positives) <= limit:
        return sorted(positives)

    rng = random.Random(seed)
    return sorted(rng.sample(positives, limit))


def load_sae_checkpoint(checkpoint_path: Path, device: torch.device) -> Autoencoder:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected SAE checkpoint format: {checkpoint_path}")
    model = Autoencoder.from_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def top_latent_indices(model: Autoencoder, activations: torch.Tensor, top_k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        latents, _ = model.encode(activations.to(device), t=None)
        mean_activation = latents.mean(dim=0)
        values, indices = torch.topk(mean_activation, k=min(top_k, mean_activation.numel()))
    return indices.cpu(), values.cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract top SAE latent indices for a CelebA-HQ concept.")
    parser.add_argument("--concept", type=str, required=True, help="Attribute name in list_attr_celeba_shift.txt, e.g. Smiling")
    parser.add_argument("--dit-repo", type=Path, required=True)
    parser.add_argument("--dit-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-source", type=str, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--attr-file", type=Path, default=DEFAULT_ATTR_FILE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
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

    image_paths = sample_positive_images(
        image_dir=args.image_dir,
        attr_file=args.attr_file,
        concept=args.concept,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"Concept: {args.concept}")
    print(f"Positive images selected: {len(image_paths)}")
    print(f"Device: {device}")

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

    indices, values = top_latent_indices(sae, pooled_activations, args.top_k, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.concept
    pt_path = args.output_dir / f"{stem}_top{args.top_k}_indices.pt"
    json_path = args.output_dir / f"{stem}_top{args.top_k}_indices.json"

    torch.save(indices, pt_path)
    payload = {
        "concept": args.concept,
        "num_images": len(used_paths),
        "seed": args.seed,
        "block_index": args.block_index,
        "timestep": args.timestep,
        "raw_block_shape": raw_shape,
        "pooled_shape": list(pooled_activations.shape),
        "sae_checkpoint": str(args.sae_checkpoint),
        "top_k": args.top_k,
        "top_indices": indices.tolist(),
        "top_values": [float(v) for v in values.tolist()],
        "sampled_paths": used_paths[: min(20, len(used_paths))],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved top-{args.top_k} latent indices to: {pt_path}")
    print(f"Saved metadata to: {json_path}")
    print(f"Top indices: {indices.tolist()[:10]} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
