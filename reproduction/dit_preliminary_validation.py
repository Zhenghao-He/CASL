#!/usr/bin/env python3
"""
Minimal DiT activation extractor.

This version is intentionally narrow:
- randomly sample local images from the current workspace
- load a local VAE and a local DiT checkpoint
- hook one DiT block
- print the raw activation shape for that block
- save pooled activations for later SAE training

By default, it saves pooled activations with shape [N, D], not the full
token tensor [N, T, D], because full-token activations become very large.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image


DEFAULT_IMAGE_DIR = Path("/p/realai/zhenghao/LoRA_SAE/CelebA-HQ/data1024")
DEFAULT_OUTPUT_DIR = Path("/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/results/dit_activation_dump")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_dit_repo_to_path(dit_repo: Path) -> None:
    repo_str = str(dit_repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_dit_model(dit_repo: Path, checkpoint: Path, image_size: int, device: torch.device):
    models_py = dit_repo / "models.py"
    if not models_py.exists():
        raise RuntimeError(f"Could not import DiT from {dit_repo}: missing {models_py}")

    spec = importlib.util.spec_from_file_location("dit_models_local", models_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import DiT from {dit_repo}: failed to load import spec")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Could not import DiT from {dit_repo}") from exc

    if not hasattr(module, "DiT_models"):
        raise RuntimeError(f"Could not import DiT from {dit_repo}: DiT_models not found in models.py")

    DiT_models = module.DiT_models

    if "DiT-XL/2" not in DiT_models:
        raise RuntimeError("This script expects DiT-XL/2 in models.DiT_models.")

    latent_size = image_size // 8
    model = DiT_models["DiT-XL/2"](input_size=latent_size, num_classes=1000)

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("ema", "model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected checkpoint keys: {len(unexpected)}")

    model.eval()
    model.to(device)
    return model


def load_vae(vae_source: str, device: torch.device):
    try:
        from diffusers.models import AutoencoderKL
    except ImportError as exc:
        raise RuntimeError("diffusers is required to encode images into DiT latents.") from exc

    vae = AutoencoderKL.from_pretrained(vae_source)
    vae.eval()
    vae.to(device)
    return vae


def encode_with_vae(vae, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
    return latents * 0.18215


def collect_image_paths(image_dir: Path, limit: int, seed: int) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = [p for p in sorted(image_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise FileNotFoundError(f"No images found under: {image_dir}")

    if limit <= 0 or limit >= len(paths):
        return paths

    rng = random.Random(seed)
    return sorted(rng.sample(paths, limit))


def center_crop_square(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    short = min(width, height)
    left = (width - short) // 2
    top = (height - short) // 2
    image = image.crop((left, top, left + short, top + short))
    return image.resize((image_size, image_size), Image.BICUBIC)


def image_to_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = center_crop_square(image.convert("RGB"), image_size)
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    return data * 2.0 - 1.0


def make_batch(image_paths: Sequence[Path], image_size: int, device: torch.device) -> torch.Tensor:
    batch = torch.stack([image_to_tensor(path, image_size) for path in image_paths], dim=0)
    return batch.to(device)


def extract_block_activations(
    model,
    vae,
    image_paths: Sequence[Path],
    block_index: int,
    image_size: int,
    batch_size: int,
    timestep: int,
    class_label: int,
    device: torch.device,
) -> tuple[torch.Tensor, List[str], List[int]]:
    if not hasattr(model, "blocks"):
        raise RuntimeError("Loaded model has no .blocks attribute.")
    if block_index < 0 or block_index >= len(model.blocks):
        raise ValueError(f"block_index={block_index} is out of range (num_blocks={len(model.blocks)})")

    captured: dict[str, torch.Tensor] = {}
    raw_shape: List[int] | None = None
    pooled_batches: List[torch.Tensor] = []
    used_paths: List[str] = []

    def hook_fn(_module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        captured["hidden"] = tensor.detach()

    handle = model.blocks[block_index].register_forward_hook(hook_fn)
    model_dtype = next(model.parameters()).dtype

    try:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = make_batch(batch_paths, image_size, device)
            latents = encode_with_vae(vae, images).to(dtype=model_dtype)
            t = torch.full((latents.shape[0],), timestep, device=device, dtype=torch.long)
            y = torch.full((latents.shape[0],), class_label, device=device, dtype=torch.long)

            with torch.no_grad():
                _ = model(latents, t, y)

            hidden = captured["hidden"]
            if hidden.ndim != 3:
                raise RuntimeError(f"Expected [B, T, D] block output, got {tuple(hidden.shape)}")

            if raw_shape is None:
                raw_shape = list(hidden.shape)
                print(f"Raw block activation shape: {tuple(raw_shape)}")

            pooled_batches.append(hidden.mean(dim=1).float().cpu())
            used_paths.extend(str(path) for path in batch_paths)
    finally:
        handle.remove()

    assert raw_shape is not None
    pooled = torch.cat(pooled_batches, dim=0)
    return pooled, used_paths, raw_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DiT block activation extractor.")
    parser.add_argument("--dit-repo", type=Path, required=True, help="Local clone of the DiT repo.")
    parser.add_argument("--dit-checkpoint", type=Path, required=True, help="Local DiT checkpoint.")
    parser.add_argument("--vae-source", type=str, required=True, help="Path or HF source for AutoencoderKL.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-index", type=int, default=20)
    parser.add_argument("--timestep", type=int, default=250)
    parser.add_argument("--class-label", type=int, default=0, help="Dummy class label used for all inputs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    image_paths = collect_image_paths(args.image_dir, args.limit, args.seed)

    print(f"Using image dir: {args.image_dir}")
    print(f"Images sampled: {len(image_paths)}")
    print(f"Sampling seed: {args.seed}")
    print(f"Device: {device}")

    vae = load_vae(args.vae_source, device)
    model = load_dit_model(args.dit_repo, args.dit_checkpoint, args.image_size, device)

    pooled_activations, used_paths, raw_shape = extract_block_activations(
        model=model,
        vae=vae,
        image_paths=image_paths,
        block_index=args.block_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        timestep=args.timestep,
        class_label=args.class_label,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    activation_path = args.output_dir / f"block_{args.block_index}_pooled_activations.pt"
    meta_path = args.output_dir / f"block_{args.block_index}_metadata.json"

    torch.save(
        {
            "pooled_activations": pooled_activations,
            "paths": used_paths,
            "raw_block_shape": raw_shape,
            "pooled_shape": list(pooled_activations.shape),
            "block_index": args.block_index,
            "timestep": args.timestep,
            "class_label": args.class_label,
        },
        activation_path,
    )

    metadata = {
        "image_dir": str(args.image_dir),
        "num_images": len(used_paths),
        "seed": args.seed,
        "block_index": args.block_index,
        "timestep": args.timestep,
        "class_label": args.class_label,
        "raw_block_shape": raw_shape,
        "pooled_shape": list(pooled_activations.shape),
        "activation_file": str(activation_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Pooled activation shape: {tuple(pooled_activations.shape)}")
    print(f"Saved activations to: {activation_path}")
    print(f"Saved metadata to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
