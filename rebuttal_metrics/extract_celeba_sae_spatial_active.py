#!/usr/bin/env python3
"""Extract spatial SAE features for the active latent pool only."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.guided_diffusion.script_util import guided_Diffusion
from models.sae.sae import Autoencoder
from utils.diffusion_utils import get_beta_schedule


def parse_args():
    root = Path("rebuttal_metrics/results/celeba_sae_omes_independent")
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
        "--image-dir", type=Path, default=Path("../CelebA-HQ/data1024")
    )
    parser.add_argument(
        "--activity-representation",
        type=Path,
        default=root / "representations_t509_mean.npy",
    )
    parser.add_argument("--start-index", type=int, default=20000)
    parser.add_argument("--n-images", type=int, default=4096)
    parser.add_argument("--timestep", type=int, default=509)
    parser.add_argument("--inactive-threshold", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=root)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activity = np.load(args.activity_representation, mmap_mode="r")
    scale = activity.std(axis=0, ddof=1, dtype=np.float64)
    active = np.flatnonzero(scale >= args.inactive_threshold).astype(np.int64)
    if len(activity) != args.n_images:
        raise ValueError("Activity representation and requested image count differ")

    all_images = sorted(
        path.name
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    image_names = all_images[args.start_index : args.start_index + args.n_images]
    if len(image_names) != args.n_images:
        raise ValueError("Requested image slice extends beyond the dataset")

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
    active_tensor = torch.as_tensor(active, device=args.device)
    selected_weight = sae.encoder.weight.index_select(0, active_tensor)
    selected_bias = sae.latent_bias.index_select(0, active_tensor)

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
    output_path = args.output_dir / f"representations_t{args.timestep}_spatial_active.npy"
    index_path = args.output_dir / f"active_indices_threshold{args.inactive_threshold:.2f}.npy"
    np.save(index_path, active)
    representations = None
    token_count = None

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
            current_tokens = middle_h.shape[2] * middle_h.shape[3]
            if representations is None:
                token_count = current_tokens
                representations = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(image_names), token_count, len(active)),
                )
            elif current_tokens != token_count:
                raise ValueError("Bottleneck token count changed between batches")
            tokens = middle_h.permute(0, 2, 3, 1).reshape(
                -1, middle_h.shape[1]
            )
            token_timesteps = timesteps.repeat_interleave(current_tokens)
            if not sae.ignore_timesteps:
                tokens = tokens + sae.timestep_embedding(token_timesteps)
            tokens, _ = sae.preprocess(tokens)
            selected = sae.activation(
                F.linear(tokens - sae.pre_bias, selected_weight, selected_bias)
            )
            selected = selected.reshape(len(names), token_count, len(active))
            representations[start : start + len(names)] = (
                selected.float().cpu().numpy()
            )
            if start % (10 * args.batch_size) == 0:
                print(f"Encoded {start + len(names)}/{len(image_names)}", flush=True)
    representations.flush()
    metadata = {
        "protocol": "single deterministic q(x_t|x_0) forward pass",
        "diffusion_checkpoint": str(args.diffusion_checkpoint.resolve()),
        "sae": str(args.sae.resolve()),
        "activity_representation": str(args.activity_representation.resolve()),
        "start_index_zero_based": args.start_index,
        "n_images": len(image_names),
        "timestep": args.timestep,
        "noise_seed": args.seed,
        "n_tokens": token_count,
        "n_active": len(active),
        "inactive_threshold": args.inactive_threshold,
        "representation": str(output_path.resolve()),
        "active_indices": str(index_path.resolve()),
    }
    with (
        args.output_dir / "spatial_active_representation_metadata.json"
    ).open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
