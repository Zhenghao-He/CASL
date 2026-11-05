#!/usr/bin/env python3
"""Split official Asyrp source/edit grids into paired 256px images."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ID_PATTERN = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--precomputed-pairs",
        type=Path,
        help="Official inversion pairs; use pair[1] as the source reconstruction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.output_dir / "source"
    edit_dir = args.output_dir / "edit"
    source_dir.mkdir(parents=True, exist_ok=True)
    edit_dir.mkdir(parents=True, exist_ok=True)
    inversion_pairs = None
    if args.precomputed_pairs is not None:
        inversion_pairs = torch.load(
            args.precomputed_pairs, map_location="cpu", weights_only=False
        )

    count = 0
    for path in sorted(args.input_dir.glob("*.png")):
        match = ID_PATTERN.search(path.stem)
        if match is None:
            continue
        image_id = int(match.group(1))
        image = Image.open(path).convert("RGB")
        if image.size != (258, 515):
            raise ValueError(f"Expected a 258x515 two-image grid, got {image.size}: {path}")
        # torchvision.make_grid(..., nrow=1, padding=1) places 256px images at
        # x=[1,257), y=[1,257) and y=[258,514), respectively.
        if inversion_pairs is None:
            source = image.crop((1, 1, 257, 257))
        else:
            reconstruction = inversion_pairs[image_id][1][0].clamp(-1, 1)
            array = (
                reconstruction.add(1)
                .mul(127.5)
                .round()
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            source = Image.fromarray(np.asarray(array), mode="RGB")
        edited = image.crop((1, 258, 257, 514))
        source.save(source_dir / f"{image_id}.png")
        edited.save(edit_dir / f"{image_id}.png")
        count += 1

    if count == 0:
        raise RuntimeError(f"No PNG grids found in {args.input_dir}")
    print(f"Wrote {count} source/edit pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
