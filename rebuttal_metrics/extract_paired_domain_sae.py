#!/usr/bin/env python3
"""Extract frozen SAE codes for independent source/edit factor probes."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from models.sae.sae import Autoencoder
from utils.diffusion_utils import get_beta_schedule


DOMAIN_DEFAULTS = {
    "church": {
        "config": "rebuttal_metrics/configs/lsun_church_table12.json",
        "diffusion_config": "configs/church.yml",
        "diffusion_checkpoint": "pretrained/ddpm_church.pt",
        "sae": "sae/church_outdoor/sae_model_ReLUNone_alpha32.0_scale128.pth",
        "output_dir": "rebuttal_metrics/results/lsun_church_selection_omes",
    },
    "dog": {
        "config": "rebuttal_metrics/configs/afhq_dog_table13.json",
        "diffusion_config": "configs/afhq.yml",
        "diffusion_checkpoint": "pretrained/afhqdog_p2.pt",
        "sae": "sae/dog/sae_model_ReLUNone_alpha32.0_scale128.pth",
        "output_dir": "rebuttal_metrics/results/afhq_dog_selection_omes",
    },
}


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAIN_DEFAULTS, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--diffusion-config", type=Path)
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--sae", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timestep", type=int, default=509)
    parser.add_argument("--pooling", choices=("mean", "max"), default="mean")
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    defaults = DOMAIN_DEFAULTS[args.domain]
    for name in (
        "config",
        "diffusion_config",
        "diffusion_checkpoint",
        "sae",
        "output_dir",
    ):
        if getattr(args, name) is None:
            setattr(args, name, Path(defaults[name]))
    return args


def index_images(directory):
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def load_diffusion(args):
    with args.diffusion_config.open() as handle:
        config = namespace(yaml.safe_load(handle))
    if args.domain == "church":
        model = DDPM(config)
    else:
        model = i_DDPM("AFHQ")
    state = torch.load(
        args.diffusion_checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(state, strict=False)
    return model.eval().to(args.device)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.config.open() as handle:
        evaluation_config = json.load(handle)
    probes = evaluation_config["methods"]["Asyrp"]

    print(f"Loading {args.domain} diffusion and SAE", flush=True)
    diffusion = load_diffusion(args)
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

    arrays = {}
    pairing_audit = {}
    with torch.inference_mode():
        for concept, directories in probes.items():
            source_dir = (REPOSITORY_ROOT / directories["source_dir"]).resolve()
            edited_dir = (REPOSITORY_ROOT / directories["edited_dir"]).resolve()
            source_images = index_images(source_dir)
            edited_images = index_images(edited_dir)
            image_ids = sorted(
                set(source_images) & set(edited_images),
                key=lambda value: int(value) if value.isdigit() else value,
            )
            if not image_ids:
                raise ValueError(f"No pairs for {concept}")
            source_codes = []
            edited_codes = []
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            for start in range(0, len(image_ids), args.pair_batch_size):
                batch_ids = image_ids[start : start + args.pair_batch_size]
                source = torch.stack(
                    [
                        transform(Image.open(source_images[item]).convert("RGB"))
                        for item in batch_ids
                    ]
                ).to(args.device)
                edited = torch.stack(
                    [
                        transform(Image.open(edited_images[item]).convert("RGB"))
                        for item in batch_ids
                    ]
                ).to(args.device)
                noise = torch.randn(
                    source.shape,
                    generator=generator,
                    device=args.device,
                    dtype=source.dtype,
                )
                source_xt = alpha_bar**0.5 * source + (1 - alpha_bar) ** 0.5 * noise
                edited_xt = alpha_bar**0.5 * edited + (1 - alpha_bar) ** 0.5 * noise
                model_input = torch.cat([source_xt, edited_xt])
                timesteps = torch.full(
                    (len(model_input),),
                    args.timestep,
                    dtype=torch.long,
                    device=args.device,
                )
                _, _, _, middle_h, _ = diffusion(model_input, timesteps)
                tokens_per_image = middle_h.shape[2] * middle_h.shape[3]
                tokens = middle_h.permute(0, 2, 3, 1).reshape(
                    -1, middle_h.shape[1]
                )
                token_timesteps = timesteps.repeat_interleave(tokens_per_image)
                latents, _ = sae.encode(tokens, token_timesteps)
                latents = latents.reshape(
                    len(model_input), tokens_per_image, sae.encoder.out_features
                )
                pooled = (
                    latents.mean(dim=1)
                    if args.pooling == "mean"
                    else latents.amax(dim=1)
                )
                source_pooled, edited_pooled = pooled.split(len(batch_ids))
                source_codes.append(source_pooled.float().cpu().numpy())
                edited_codes.append(edited_pooled.float().cpu().numpy())
            arrays[f"{concept}__source"] = np.concatenate(source_codes)
            arrays[f"{concept}__edit"] = np.concatenate(edited_codes)
            pairing_audit[concept] = {
                "n_pairs": len(image_ids),
                "image_ids": image_ids,
                "source_dir": str(source_dir),
                "edited_dir": str(edited_dir),
            }
            print(f"Encoded {concept}: {len(image_ids)} pairs", flush=True)

    representation_path = args.output_dir / "paired_representations.npz"
    np.savez_compressed(representation_path, **arrays)
    metadata = {
        "domain": args.domain,
        "probe_generator": "official Asyrp source/edit outputs",
        "probe_independence": (
            "CASL edits are not used to construct the factor intervention pairs"
        ),
        "diffusion_checkpoint": str(args.diffusion_checkpoint.resolve()),
        "sae": str(args.sae.resolve()),
        "timestep": args.timestep,
        "alpha_bar": alpha_bar,
        "noise_seed": args.seed,
        "paired_noise": True,
        "pooling": args.pooling,
        "n_latents": sae.encoder.out_features,
        "representation": str(representation_path.resolve()),
        "pairing": pairing_audit,
    }
    with (args.output_dir / "representation_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
