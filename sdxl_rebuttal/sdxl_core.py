from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    StableDiffusionXLPipeline,
)

from .utils import batched, image_path, load_square_image


@dataclass
class PromptCondition:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    time_ids: torch.Tensor

    def batch(self, size: int) -> "PromptCondition":
        def repeat(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[0] == size:
                return tensor
            if tensor.shape[0] != 1:
                raise ValueError(f"Cannot repeat conditioning batch {tensor.shape[0]} to {size}")
            return tensor.repeat(size, *([1] * (tensor.ndim - 1)))

        return PromptCondition(
            repeat(self.prompt_embeds),
            repeat(self.pooled_prompt_embeds),
            repeat(self.time_ids),
        )


class MidBlockController:
    """Single hook used for capture, intervention, and hook-integrity checks."""

    def __init__(self, mid_block: nn.Module):
        self.capture = False
        self.last_activation: torch.Tensor | None = None
        self.intervention: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.active_indices: set[int] = set()
        self.step_index = -1
        self.call_count = 0
        self.handle = mid_block.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor mid-block output, got {type(output)}")
        if self.capture:
            self.last_activation = output.detach()
        if self.intervention is not None and self.step_index in self.active_indices:
            temb = _inputs[1] if len(_inputs) > 1 else None
            output = self.intervention(output, temb)
        return output

    def configure(
        self,
        intervention: Callable[[torch.Tensor], torch.Tensor] | None,
        active_indices: set[int] | None = None,
    ) -> None:
        self.intervention = intervention
        self.active_indices = active_indices or set()

    def close(self) -> None:
        self.handle.remove()


def load_pipeline(
    model_id: str,
    device: torch.device,
    *,
    local_files_only: bool = False,
) -> StableDiffusionXLPipeline:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
        local_files_only=local_files_only,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    # SDXL's VAE is numerically safer in fp32; the U-Net remains bf16.
    pipe.vae.to(dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    for module in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return pipe


@torch.no_grad()
def make_condition(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    resolution: int,
    device: torch.device,
) -> PromptCondition:
    prompt_embeds, _, pooled, _ = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    projection_dim = pipe.text_encoder_2.config.projection_dim
    time_ids = pipe._get_add_time_ids(
        (resolution, resolution),
        (0, 0),
        (resolution, resolution),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=projection_dim,
    ).to(device)
    return PromptCondition(prompt_embeds, pooled, time_ids)


def release_text_encoders(pipe: StableDiffusionXLPipeline) -> None:
    for name in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
        component = getattr(pipe, name, None)
        if isinstance(component, nn.Module):
            component.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unet_eps(
    pipe: StableDiffusionXLPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    condition: PromptCondition,
) -> torch.Tensor:
    condition = condition.batch(latents.shape[0])
    latent_input = pipe.scheduler.scale_model_input(latents, timestep)
    return pipe.unet(
        latent_input,
        timestep,
        encoder_hidden_states=condition.prompt_embeds,
        added_cond_kwargs={
            "text_embeds": condition.pooled_prompt_embeds,
            "time_ids": condition.time_ids,
        },
        return_dict=False,
    )[0]


def encode_images(pipe: StableDiffusionXLPipeline, images_m11: torch.Tensor) -> torch.Tensor:
    vae_dtype = pipe.vae.dtype
    images_m11 = images_m11.to(device=pipe.vae.device, dtype=vae_dtype)
    encoded = pipe.vae.encode(images_m11).latent_dist.mean
    return encoded * float(pipe.vae.config.scaling_factor)


def decode_latents(pipe: StableDiffusionXLPipeline, latents: torch.Tensor) -> torch.Tensor:
    vae_dtype = pipe.vae.dtype
    scaled = latents.to(device=pipe.vae.device, dtype=vae_dtype)
    scaled = scaled / float(pipe.vae.config.scaling_factor)
    return pipe.vae.decode(scaled, return_dict=False)[0].clamp(-1, 1)


@torch.no_grad()
def invert_latent(
    pipe: StableDiffusionXLPipeline,
    latent_x0: torch.Tensor,
    condition: PromptCondition,
    steps: int,
    controller: MidBlockController,
) -> torch.Tensor:
    inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inverse.set_timesteps(steps, device=latent_x0.device)
    # Keep unet_eps and scheduler scale_model_input on the same scheduler.
    forward_scheduler = pipe.scheduler
    pipe.scheduler = inverse
    latent = latent_x0.to(dtype=pipe.unet.dtype)
    controller.configure(None)
    try:
        for index, timestep in enumerate(inverse.timesteps):
            controller.step_index = index
            eps = unet_eps(pipe, latent, timestep, condition)
            latent = inverse.step(eps, timestep, latent, return_dict=False)[0]
    finally:
        pipe.scheduler = forward_scheduler
    return latent


def denoise_latent(
    pipe: StableDiffusionXLPipeline,
    latent_xT: torch.Tensor,
    condition: PromptCondition,
    steps: int,
    controller: MidBlockController,
    intervention: Callable[[torch.Tensor], torch.Tensor] | None = None,
    active_indices: set[int] | None = None,
    *,
    require_grad: bool = False,
) -> torch.Tensor:
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(steps, device=latent_xT.device)
    previous_scheduler = pipe.scheduler
    pipe.scheduler = scheduler
    latent = latent_xT.to(dtype=pipe.unet.dtype)
    controller.configure(intervention, active_indices)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    try:
        with context:
            for index, timestep in enumerate(scheduler.timesteps):
                controller.step_index = index
                eps = unet_eps(pipe, latent, timestep, condition)
                latent = scheduler.step(eps, timestep, latent, return_dict=False)[0]
    finally:
        controller.configure(None)
        pipe.scheduler = previous_scheduler
    return latent


def predict_x0(
    scheduler: DDIMScheduler,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    alpha = scheduler.alphas_cumprod[timestep].to(device=sample.device, dtype=sample.dtype)
    beta = 1 - alpha
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return (sample - beta.sqrt() * model_output) / alpha.sqrt()
    if prediction_type == "v_prediction":
        return alpha.sqrt() * sample - beta.sqrt() * model_output
    if prediction_type == "sample":
        return model_output
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


@torch.no_grad()
def collect_activation_cache(
    pipe: StableDiffusionXLPipeline,
    cfg: dict,
    ids: list[str],
    condition: PromptCondition,
    controller: MidBlockController,
    *,
    image_batch_size: int = 2,
) -> dict:
    resolution = int(cfg["resolution"])
    steps = int(cfg["num_inference_steps"])
    step_indices = [int(value) for value in cfg["activation_step_indices"]]
    tokens_per = int(cfg["tokens_per_activation"])
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    execution_device = pipe.unet.device
    scheduler.set_timesteps(steps, device=execution_device)
    prior_scheduler = pipe.scheduler
    pipe.scheduler = scheduler
    all_tokens: list[torch.Tensor] = []
    all_latents: list[torch.Tensor] = []
    controller.capture = True
    controller.configure(None)
    try:
        for id_batch in batched(ids, image_batch_size):
            images = torch.stack(
                [load_square_image(image_path(cfg, image_id), resolution) for image_id in id_batch]
            ).to(execution_device)
            latents = encode_images(pipe, images).to(dtype=pipe.unet.dtype)
            all_latents.append(latents.detach().cpu().to(torch.float16))
            for step_index in step_indices:
                timestep = scheduler.timesteps[step_index]
                noise_parts = []
                for image_id in id_batch:
                    numeric_id = int(Path(image_id).stem)
                    generator = torch.Generator(device=execution_device).manual_seed(
                        int(cfg["seed"]) + numeric_id * 1009 + step_index
                    )
                    noise_parts.append(
                        torch.randn(
                            latents[0].shape,
                            generator=generator,
                            device=execution_device,
                            dtype=latents.dtype,
                        )
                    )
                noise = torch.stack(noise_parts)
                noisy = scheduler.add_noise(latents, noise, timestep)
                controller.step_index = step_index
                _ = unet_eps(pipe, noisy, timestep, condition)
                if controller.last_activation is None:
                    raise RuntimeError("Mid-block hook did not capture an activation")
                activation = controller.last_activation
                batch, channels, height, width = activation.shape
                flat = activation.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
                for local_index, image_id in enumerate(id_batch):
                    token_gen = torch.Generator(device=flat.device).manual_seed(
                        int(cfg["seed"]) + int(Path(image_id).stem) * 917 + step_index
                    )
                    count = min(tokens_per, height * width)
                    indices = torch.randperm(height * width, generator=token_gen, device=flat.device)[:count]
                    all_tokens.append(flat[local_index, indices].detach().cpu().to(torch.float16))
    finally:
        controller.capture = False
        controller.last_activation = None
        pipe.scheduler = prior_scheduler
    tokens = torch.cat(all_tokens)
    latents = torch.cat(all_latents)
    return {
        "tokens": tokens,
        "latents": latents,
        "image_ids": ids,
        "step_indices": step_indices,
        "input_dim": int(tokens.shape[1]),
        "resolution": resolution,
        "prompt": cfg["neutral_prompt"],
        "model_id": cfg["model_id"],
    }
