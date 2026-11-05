#!/usr/bin/env python3
"""Split torchvision vertical strength grids into per-strength images."""

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", required=True)
    parser.add_argument("--min-strength", type=float, required=True)
    parser.add_argument("--max-strength", type=float, required=True)
    parser.add_argument("--num-strengths", type=int, required=True)
    args = parser.parse_args()

    input_dir = (
        ROOT
        / "runs"
        / (
            f"novelty_dense_grid_{args.concept}"
            "_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
        )
        / "sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse"
        / "xblock_test_images"
        / "100"
    )
    output_root = ROOT / "novelty_ablation" / "artifacts" / "dense" / args.concept
    strengths = np.linspace(
        args.min_strength, args.max_strength, args.num_strengths
    )
    paths = sorted(input_dir.glob("test_*_ngen50.png"))
    if not paths:
        raise FileNotFoundError(f"No dense grids in {input_dir}")
    for path in paths:
        match = re.match(r"test_(\d{5})_", path.name)
        if match is None:
            continue
        image_id = match.group(1)
        grid = Image.open(path).convert("RGB")
        # torchvision.save_image writes a bare 256 x 256 image when the grid
        # contains one tile, but adds one-pixel outer/inter-tile padding for
        # multi-image grids.
        if args.num_strengths == 1 and grid.size == (256, 256):
            crops = [grid]
        else:
            expected_height = args.num_strengths * 256 + (
                args.num_strengths + 1
            )
            if grid.width != 258 or grid.height != expected_height:
                raise ValueError(
                    f"Unexpected grid shape {grid.size}; expected "
                    f"(258, {expected_height})"
                )
            crops = [
                grid.crop((1, 1 + index * 257, 257, 257 + index * 257))
                for index in range(args.num_strengths)
            ]
        for crop, strength in zip(crops, strengths):
            tag = f"{float(strength):g}"
            output_dir = output_root / f"strength_{tag}"
            output_dir.mkdir(parents=True, exist_ok=True)
            crop.save(output_dir / f"{image_id}.png")
    print(
        f"Split {len(paths)} grids x {len(strengths)} strengths into {output_root}"
    )


if __name__ == "__main__":
    main()
