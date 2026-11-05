#!/usr/bin/env python3
"""Efficient Concept Slider strength sweep using the repository implementation."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

SLIDER_ROOT = Path(__file__).resolve().parents[2] / "sliders"
sys.path.insert(0, str(SLIDER_ROOT))
from trainscripts.textsliders.lora import (  # noqa: E402
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
    LoRANetwork,
)


MODELS = {
    "smiling": "smilinglider_alpha1.0_rank4_noxattn/smilinglider_alpha1.0_rank4_noxattn_last.safetensors",
    "young": "ageslider_alpha1.0_rank4_noxattn/ageslider_alpha1.0_rank4_noxattn_last.safetensors",
    "big_nose": "bignoseslider_alpha1.0_rank4_noxattn/bignoseslider_alpha1.0_rank4_noxattn_last.safetensors",
    "beards": "beardslider_alpha1.0_rank4_noxattn/beardslider_alpha1.0_rank4_noxattn_last.safetensors",
    "blond_hair": "blondeslider_alpha1.0_rank4_noxattn/blondeslider_alpha1.0_rank4_noxattn_last.safetensors",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", choices=tuple(MODELS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--scales", type=float, nargs="+", default=(0, 32, 64, 128, 192))
    parser.add_argument("--start-noise", type=int, default=500)
    parser.add_argument("--prompt", default="a photo of a person")
    parser.add_argument(
        "--input-dir", type=Path, default=SLIDER_ROOT / "datasets/celeba-hq"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results/slider",
    )
    return parser.parse_args()


def image_to_latent(image, vae, device):
    array = np.asarray(image.resize((512, 512)), dtype=np.float32)
    tensor = torch.from_numpy(array).div(127.5).sub(1)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return vae.encode(tensor).latent_dist.mean * 0.18215


def latent_to_image(latents, vae):
    image = vae.decode(latents / 0.18215).sample
    image = image.div(2).add(0.5).clamp(0, 1)
    array = image[0].detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype(np.uint8))


def main():
    args = parse_args()
    device = torch.device(args.device)
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    modules = list(DEFAULT_TARGET_REPLACE) + list(UNET_TARGET_REPLACE_MODULE_CONV)
    network = LoRANetwork(
        unet,
        rank=4,
        multiplier=1.0,
        alpha=1.0,
        train_method="noxattn",
    ).to(device)
    checkpoint = SLIDER_ROOT / "models" / MODELS[args.concept]
    network.load_state_dict(load_file(checkpoint), strict=False)

    text_input = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        conditioned = text_encoder(text_input.input_ids.to(device))[0]
        unconditioned = text_encoder(uncond_input.input_ids.to(device))[0]
    embeddings = torch.cat([unconditioned, conditioned])
    scheduler.set_timesteps(50)

    ids = [f"{value:05d}" for value in range(29501, 29501 + args.n_images)]
    for image_id in tqdm(ids, desc=args.concept):
        source = Image.open(args.input_dir / f"{image_id}.jpg").convert("RGB")
        with torch.inference_mode():
            latent = image_to_latent(source, vae, device) * scheduler.init_noise_sigma
            for scale in args.scales:
                current = latent.clone()
                for timestep in scheduler.timesteps:
                    network.set_lora_slider(
                        scale=0 if timestep > args.start_noise else scale
                    )
                    model_input = torch.cat([current, current])
                    model_input = scheduler.scale_model_input(
                        model_input, timestep=timestep
                    )
                    with network:
                        prediction = unet(
                            model_input,
                            timestep,
                            encoder_hidden_states=embeddings,
                        ).sample
                    unconditional, conditional = prediction.chunk(2)
                    prediction = unconditional + 7.5 * (
                        conditional - unconditional
                    )
                    current = scheduler.step(
                        prediction, timestep, current
                    ).prev_sample
                output_dir = (
                    args.output_root / args.concept / f"scale_{scale:g}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                latent_to_image(current, vae).save(output_dir / f"{image_id}.png")


if __name__ == "__main__":
    main()
