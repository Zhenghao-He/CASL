#!/usr/bin/env python3
"""Extract frozen diffusion/SAE representations on an independent image slice."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.guided_diffusion.script_util import guided_Diffusion
from models.sae.sae import Autoencoder
from utils.diffusion_utils import get_beta_schedule


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
        "--diffusion-checkpoint",
        type=Path,
        default=Path("pretrained/celebahq_p2.pt"),
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
    parser.add_argument("--start-index", type=int, default=20000)
    parser.add_argument("--n-images", type=int, default=4096)
    parser.add_argument("--timestep", type=int, default=509)
    parser.add_argument("--pooling", choices=("mean", "max"), default="mean")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rebuttal_metrics/results/celeba_sae_omes_independent"),
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
    all_images = sorted(
        path.name
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    image_names = all_images[args.start_index : args.start_index + args.n_images]
    if len(image_names) != args.n_images:
        raise ValueError("Requested image slice extends beyond the dataset")
    attribute_names, attributes = read_attributes(args.attributes)

    print(f"Loading diffusion checkpoint {args.diffusion_checkpoint}", flush=True)
    diffusion = guided_Diffusion("CelebA_HQ_P2")
    diffusion.load_state_dict(
        torch.load(
            args.diffusion_checkpoint, map_location="cpu", weights_only=False
        ),
        strict=False,
    )
    diffusion = diffusion.eval().to(args.device)
    sae = Autoencoder.from_state_dict(
        torch.load(args.sae, map_location="cpu", weights_only=False)
    ).eval().to(args.device)

    betas = get_beta_schedule(
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
    )
    alpha_bar = float(np.cumprod(1.0 - betas)[args.timestep])
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    output_path = (
        args.output_dir
        / f"representations_t{args.timestep}_{args.pooling}.npy"
    )
    representations = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(image_names), sae.encoder.out_features),
    )

    with torch.inference_mode():
        for start in range(0, len(image_names), args.batch_size):
            names = image_names[start : start + args.batch_size]
            x0 = torch.stack(
                [
                    transform(Image.open(args.image_dir / name).convert("RGB"))
                    for name in names
                ]
            ).to(args.device)
            noise = torch.randn(
                x0.shape,
                generator=generator,
                device=args.device,
                dtype=x0.dtype,
            )
            xt = alpha_bar**0.5 * x0 + (1.0 - alpha_bar) ** 0.5 * noise
            timesteps = torch.full(
                (len(names),),
                args.timestep,
                dtype=torch.long,
                device=args.device,
            )
            _, _, _, middle_h, _ = diffusion(xt, timesteps)
            tokens = middle_h.permute(0, 2, 3, 1).reshape(
                -1, middle_h.shape[1]
            )
            token_timesteps = timesteps.repeat_interleave(
                middle_h.shape[2] * middle_h.shape[3]
            )
            latents, _ = sae.encode(tokens, token_timesteps)
            latents = latents.reshape(
                len(names), -1, sae.encoder.out_features
            )
            pooled = (
                latents.mean(dim=1)
                if args.pooling == "mean"
                else latents.amax(dim=1)
            )
            representations[start : start + len(names)] = (
                pooled.float().cpu().numpy()
            )
            if start % (10 * args.batch_size) == 0:
                print(f"Encoded {start + len(names)}/{len(image_names)}", flush=True)
    representations.flush()

    class_path = args.output_dir / "classes.csv"
    with class_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", *FACTORS])
        writer.writeheader()
        for image_name in image_names:
            values = attributes[image_name]
            row = {"image_id": image_name}
            for display_name, attribute_name in FACTORS.items():
                row[display_name] = int(values[attribute_names.index(attribute_name)] > 0)
            writer.writerow(row)
    metadata = {
        "protocol": "single deterministic q(x_t|x_0) forward pass",
        "diffusion_checkpoint": str(args.diffusion_checkpoint.resolve()),
        "sae": str(args.sae.resolve()),
        "start_index_zero_based": args.start_index,
        "first_image": image_names[0],
        "last_image": image_names[-1],
        "n_images": len(image_names),
        "timestep": args.timestep,
        "alpha_bar": alpha_bar,
        "noise_seed": args.seed,
        "pooling": args.pooling,
        "n_latents": sae.encoder.out_features,
        "representation": str(output_path.resolve()),
        "classes": str(class_path.resolve()),
    }
    with (args.output_dir / "representation_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
