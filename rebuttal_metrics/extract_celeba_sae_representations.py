#!/usr/bin/env python3
"""Encode cached CelebA-HQ diffusion activations with the frozen CASL SAE."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.sae.sae import Autoencoder


FACTORS = {
    "Smiling": "Smiling",
    "Young": "Young",
    "Big_Nose": "Big_Nose",
    "Beards": "No_Beard",
    "Blond_Hair": "Blond_Hair",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("sample_h/CelebA_HQ_P2_t999_ngen50_nim1000.pth"),
    )
    parser.add_argument(
        "--sae",
        type=Path,
        default=Path(
            "sae/CelebA_HQ_P2/"
            "sae_model_ReLUNone_alpha32.0_scale128.pth"
        ),
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("../CelebA-HQ/list_attr_celeba_shift.txt"),
    )
    parser.add_argument(
        "--image-dir", type=Path, default=Path("../CelebA-HQ/data1024")
    )
    parser.add_argument("--n-images", type=int, default=1000)
    parser.add_argument("--timestep", type=int, default=509)
    parser.add_argument("--pooling", choices=("mean", "max"), default="mean")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rebuttal_metrics/results/celeba_sae_omes"),
    )
    return parser.parse_args()


def read_attributes(path):
    with path.open() as handle:
        lines = handle.readlines()
    names = lines[1].strip().split()
    rows = {}
    for line in lines[2:]:
        fields = line.split()
        rows[fields[0]] = np.asarray(fields[1:], dtype=np.int8)
    return names, rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_names = sorted(
        path.name
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.n_images]
    attribute_names, attributes = read_attributes(args.attributes)
    missing = [name for name in image_names if name not in attributes]
    if missing:
        raise ValueError(f"{len(missing)} images have no attributes; first={missing[0]}")

    print(f"Loading cached activations from {args.activations}")
    samples = torch.load(args.activations, map_location="cpu", weights_only=False)
    if len(samples) % len(image_names):
        raise ValueError(
            f"{len(samples)} activations are not divisible by {len(image_names)} images"
        )
    steps_per_image = len(samples) // len(image_names)
    first_timesteps = [
        int(samples[index]["timestep"]) for index in range(steps_per_image)
    ]
    if args.timestep not in first_timesteps:
        raise ValueError(
            f"timestep {args.timestep} unavailable; choices are {first_timesteps}"
        )
    timestep_offset = first_timesteps.index(args.timestep)
    selected = [
        samples[image_index * steps_per_image + timestep_offset]["middle_h"]
        for image_index in range(len(image_names))
    ]
    del samples

    state = torch.load(args.sae, map_location="cpu", weights_only=False)
    sae = Autoencoder.from_state_dict(state).eval().to(args.device)
    n_latents = sae.encoder.out_features
    output_path = (
        args.output_dir
        / f"representations_t{args.timestep}_{args.pooling}.npy"
    )
    representations = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), n_latents),
    )

    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            batch_items = selected[start : start + args.batch_size]
            token_count = batch_items[0].shape[0]
            if any(item.shape != batch_items[0].shape for item in batch_items):
                raise ValueError("Cached activation shapes differ across images")
            batch = torch.cat(batch_items, dim=0).to(args.device)
            timesteps = torch.full(
                (batch.shape[0],),
                args.timestep,
                dtype=torch.long,
                device=args.device,
            )
            latents, _ = sae.encode(batch, timesteps)
            latents = latents.reshape(len(batch_items), token_count, n_latents)
            if args.pooling == "mean":
                pooled = latents.mean(dim=1)
            else:
                pooled = latents.amax(dim=1)
            representations[start : start + len(batch_items)] = (
                pooled.float().cpu().numpy()
            )
            if start % (10 * args.batch_size) == 0:
                print(f"Encoded {start + len(batch_items)}/{len(selected)}")
    representations.flush()

    label_path = args.output_dir / "classes.csv"
    with label_path.open("w", newline="") as handle:
        fieldnames = ["image_id", *FACTORS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_name in image_names:
            values = attributes[image_name]
            row = {"image_id": image_name}
            for display_name, attribute_name in FACTORS.items():
                row[display_name] = int(values[attribute_names.index(attribute_name)] > 0)
            writer.writerow(row)

    metadata = {
        "activations": str(args.activations.resolve()),
        "sae": str(args.sae.resolve()),
        "attributes": str(args.attributes.resolve()),
        "n_images": len(image_names),
        "steps_per_image": steps_per_image,
        "timestep": args.timestep,
        "pooling": args.pooling,
        "n_latents": n_latents,
        "representation": str(output_path.resolve()),
        "classes": str(label_path.resolve()),
    }
    with (args.output_dir / "representation_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
