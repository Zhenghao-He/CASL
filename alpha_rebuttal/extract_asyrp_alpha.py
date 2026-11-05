#!/usr/bin/env python3
"""Split official ASYRP interpolation grids into coefficient directories."""

import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "Asyrp_official/alpha_sweep",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results/asyrp",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=("smiling", "young", "big_nose", "beards", "blond_hair"),
    )
    parser.add_argument(
        "--coefficients", type=float, nargs="+", default=(0, 0.5, 1, 1.5, 2)
    )
    return parser.parse_args()


def main():
    args = parse_args()
    count = 0
    for concept in args.concepts:
        candidates = list(
            args.input_root.glob(
                f"{concept}_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
            )
        )
        if not candidates:
            continue
        input_dir = candidates[0] / "test_images/100"
        for path in sorted(input_dir.glob("*.png")):
            image = Image.open(path).convert("RGB")
            n_tiles = (image.height - 1) // 257
            if image.width != 258 or n_tiles != len(args.coefficients):
                raise ValueError(
                    f"Unexpected ASYRP grid {image.size}, expected "
                    f"258x{257 * len(args.coefficients) + 1}: {path}"
                )
            image_id = path.stem.split("_")[0]
            for index, coefficient in enumerate(args.coefficients):
                tile = image.crop((1, 1 + 257 * index, 257, 257 + 257 * index))
                output = (
                    args.output_root
                    / concept
                    / f"coeff_{coefficient:g}"
                    / f"{image_id}.png"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                tile.save(output)
                count += 1
    print(f"Wrote {count} ASYRP interpolation images")


if __name__ == "__main__":
    main()
