from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

import clip
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from tqdm import tqdm

from .models import (
    AlignmentMap,
    AsyrpDeltaBlock,
    FactorizedAlignmentMap,
    HSpaceIntervention,
    InterventionSpec,
    SparseAutoencoder,
    load_mapper,
    load_sae,
)
from .sdxl_core import (
    MidBlockController,
    collect_activation_cache,
    decode_latents,
    denoise_latent,
    encode_images,
    invert_latent,
    load_pipeline,
    make_condition,
    predict_x0,
    release_text_encoders,
    unet_eps,
)
from .utils import (
    append_event,
    atomic_torch_save,
    batched,
    ensure_dirs,
    image_path,
    load_config,
    load_square_image,
    parse_csv_floats,
    parse_csv_ints,
    seed_everything,
    tensor_to_pil,
    test_ids,
    train_ids,
    write_json,
)


def range_slug(bounds: list[int] | tuple[int, int]) -> str:
    return f"steps{int(bounds[0]):02d}-{int(bounds[1]):02d}"


def strength_slug(value: float) -> str:
    return "a" + f"{value:.5g}".replace("-", "m").replace(".", "p")


class DirectionalCLIP:
    def __init__(self, device: torch.device):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dtype = next(self.model.parameters()).dtype
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def text_features(self, source: str, target: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize([source, target]).to(self.device)
            return F.normalize(self.model.encode_text(tokens).float(), dim=-1)

    def text_direction(self, source: str, target: str) -> torch.Tensor:
        features = self.text_features(source, target)
        return F.normalize(features[1:2] - features[0:1], dim=-1)

    def image_features(self, image_m11: torch.Tensor) -> torch.Tensor:
        image = (image_m11.float() + 1) / 2
        image = F.interpolate(image, (224, 224), mode="bicubic", align_corners=False, antialias=True)
        image = (image - self.mean) / self.std
        return F.normalize(self.model.encode_image(image.to(self.dtype)).float(), dim=-1)

    def directional_loss(
        self,
        reference_m11: torch.Tensor,
        edited_m11: torch.Tensor,
        text_direction: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            reference_features = self.image_features(reference_m11)
        edited_features = self.image_features(edited_m11)
        image_direction = edited_features - reference_features
        return (1 - F.cosine_similarity(image_direction, text_direction.expand_as(image_direction), eps=1e-4)).mean()

    def alignment_losses(
        self,
        reference_m11: torch.Tensor,
        edited_m11: torch.Tensor,
        text_pair: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Directional CLIP plus a binary source-vs-target text objective.

        The second term is deliberately identical for CASL and dense. It
        prevents a high directional score from being obtained by a generic
        perceptual change that never reaches the requested target concept.
        """
        with torch.no_grad():
            reference_features = self.image_features(reference_m11)
        edited_features = self.image_features(edited_m11)
        text_direction = F.normalize(text_pair[1:2] - text_pair[0:1], dim=-1)
        image_direction = edited_features - reference_features
        directional = (
            1
            - F.cosine_similarity(
                image_direction,
                text_direction.expand_as(image_direction),
                eps=1e-4,
            )
        ).mean()
        logits = edited_features @ text_pair.T / max(float(temperature), 1e-4)
        target_labels = torch.ones(
            edited_features.shape[0], dtype=torch.long, device=edited_features.device
        )
        target = F.cross_entropy(logits, target_labels)
        return directional, target


class Experiment:
    def __init__(
        self,
        cfg: dict,
        device: str,
        *,
        concepts: list[str] | None = None,
        methods: list[str] | None = None,
        force: bool = False,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.force = force
        ensure_dirs(cfg)
        seed_everything(int(cfg["seed"]))
        default_concepts = ["Smiling"] if cfg["stage"] == "smoke" else list(cfg["concepts"])
        self.concepts = concepts or default_concepts
        self.methods = methods or ["casl", "dense"]
        for concept in self.concepts:
            if concept not in cfg["concepts"]:
                raise KeyError(f"Concept {concept!r} is not in the config")
        for method in self.methods:
            if method not in {"casl", "casl_lowrank", "dense", "asyrp_cnn"}:
                raise ValueError(f"Unsupported method {method!r}")
        self._pipe = None
        self._condition = None
        self._controller = None

    @property
    def work_dir(self) -> Path:
        return Path(self.cfg["work_dir"])

    @property
    def result_dir(self) -> Path:
        return Path(self.cfg["result_dir"])

    @property
    def activation_path(self) -> Path:
        return self.work_dir / "activation_cache.pt"

    @property
    def sae_path(self) -> Path:
        return self.work_dir / "sae.pt"

    def alignment_path(self, concept: str, method: str) -> Path:
        return self.work_dir / "alignment" / concept / f"{method}.pt"

    def inversion_path(self, image_id: str) -> Path:
        return self.work_dir / "inversions" / f"{Path(image_id).stem}.pt"

    def pipe_components(self):
        if self._pipe is None:
            started = time.time()
            self._pipe = load_pipeline(self.cfg["model_id"], self.device)
            self._condition = make_condition(
                self._pipe,
                self.cfg["neutral_prompt"],
                int(self.cfg["resolution"]),
                self.device,
            )
            self._controller = MidBlockController(self._pipe.unet.mid_block)
            release_text_encoders(self._pipe)
            append_event(self.cfg, "model_loaded", seconds=time.time() - started)
        return self._pipe, self._condition, self._controller

    def collect(self) -> None:
        if self.activation_path.exists() and not self.force:
            print(f"[reuse] {self.activation_path}")
            return
        pipe, condition, controller = self.pipe_components()
        ids = train_ids(self.cfg)
        started = time.time()
        cache = collect_activation_cache(pipe, self.cfg, ids, condition, controller)
        atomic_torch_save(cache, self.activation_path)
        append_event(
            self.cfg,
            "activation_cache_complete",
            path=str(self.activation_path),
            images=len(ids),
            tokens=int(cache["tokens"].shape[0]),
            shape=list(cache["tokens"].shape),
            seconds=time.time() - started,
        )
        print(f"[saved] {self.activation_path} {tuple(cache['tokens'].shape)}")

    def train_sae(self) -> None:
        if self.sae_path.exists() and not self.force:
            print(f"[reuse] {self.sae_path}")
            return
        if not self.activation_path.exists():
            self.collect()
        cache = torch.load(self.activation_path, map_location="cpu", weights_only=False)
        tokens = cache["tokens"]
        input_dim = int(tokens.shape[1])
        latent_dim = input_dim * int(self.cfg["sae"]["expansion"])
        sae = SparseAutoencoder(input_dim, latent_dim).to(self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam(sae.parameters(), lr=float(self.cfg["sae"]["lr"]))
        steps = int(self.cfg["sae"]["steps"])
        batch_size = min(int(self.cfg["sae"]["batch_tokens"]), int(tokens.shape[0]))
        l1_weight = float(self.cfg["sae"]["l1"])
        generator = torch.Generator(device="cpu").manual_seed(int(self.cfg["seed"]) + 41)
        variance = tokens.float().var(dim=0).mean().clamp_min(1e-6).item()
        curve: list[dict] = []
        started = time.time()
        iterator = tqdm(range(steps), desc="train SAE")
        for step in iterator:
            indices = torch.randint(tokens.shape[0], (batch_size,), generator=generator)
            batch = tokens[indices].to(self.device, dtype=torch.float32)
            reconstruction, code = sae(batch)
            mse = F.mse_loss(reconstruction, batch)
            l1 = code.abs().mean()
            loss = mse / variance + l1_weight * l1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sae.remove_decoder_parallel_gradients()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            sae.normalize_decoder()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite SAE loss at step {step}")
            if step % max(1, steps // 50) == 0 or step == steps - 1:
                active = (code > 0).float().sum(dim=1).mean()
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "mse": float(mse.detach()),
                    "l1": float(l1.detach()),
                    "mean_active": float(active.detach()),
                    "explained_variance": float(1 - mse.detach() / variance),
                }
                curve.append(row)
                iterator.set_postfix(loss=f"{row['loss']:.4f}", active=f"{row['mean_active']:.1f}")
        payload = {
            "state_dict": {key: value.detach().cpu() for key, value in sae.state_dict().items()},
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "config": self.cfg["sae"],
            "curve": curve,
            "training_seconds": time.time() - started,
        }
        atomic_torch_save(payload, self.sae_path)
        write_json(curve, self.result_dir / "sae_curve.json")
        append_event(
            self.cfg,
            "sae_complete",
            path=str(self.sae_path),
            input_dim=input_dim,
            latent_dim=latent_dim,
            seconds=time.time() - started,
            final=curve[-1],
        )
        del cache, tokens, sae, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def invert(self) -> None:
        pipe, condition, controller = self.pipe_components()
        recon_dir = self.result_dir / "reconstruction"
        recon_dir.mkdir(parents=True, exist_ok=True)
        records = []
        started = time.time()
        for image_id in tqdm(test_ids(self.cfg), desc="DDIM inversion + reconstruction"):
            latent_path = self.inversion_path(image_id)
            output_path = recon_dir / image_id
            if latent_path.exists() and output_path.exists() and not self.force:
                records.append({"image_id": image_id, "status": "reused"})
                continue
            image = load_square_image(image_path(self.cfg, image_id), int(self.cfg["resolution"]))
            with torch.no_grad():
                latent_x0 = encode_images(pipe, image.unsqueeze(0).to(self.device))
                latent_xT = invert_latent(
                    pipe,
                    latent_x0,
                    condition,
                    int(self.cfg["num_inference_steps"]),
                    controller,
                )
                reconstruction_latent = denoise_latent(
                    pipe,
                    latent_xT,
                    condition,
                    int(self.cfg["num_inference_steps"]),
                    controller,
                )
                reconstruction = decode_latents(pipe, reconstruction_latent)
            if not torch.isfinite(latent_xT).all() or not torch.isfinite(reconstruction).all():
                raise FloatingPointError(f"NaN/Inf during inversion for {image_id}")
            atomic_torch_save(
                {
                    "image_id": image_id,
                    "latent_xT": latent_xT.detach().cpu().to(torch.float16),
                    "encoded_x0": latent_x0.detach().cpu().to(torch.float16),
                    "prompt": self.cfg["neutral_prompt"],
                    "steps": int(self.cfg["num_inference_steps"]),
                    "resolution": int(self.cfg["resolution"]),
                    "model_id": self.cfg["model_id"],
                },
                latent_path,
            )
            tensor_to_pil(reconstruction).save(output_path, quality=95)
            records.append({"image_id": image_id, "status": "complete"})
        write_json(records, self.result_dir / "inversion_manifest.json")
        append_event(
            self.cfg,
            "inversion_complete",
            images=len(records),
            seconds=time.time() - started,
        )

    def _load_training_latents(self) -> tuple[torch.Tensor, list[str]]:
        if not self.activation_path.exists():
            self.collect()
        cache = torch.load(self.activation_path, map_location="cpu", weights_only=False)
        limit = min(int(self.cfg["alignment"]["train_images"]), len(cache["image_ids"]))
        return cache["latents"][:limit], cache["image_ids"][:limit]

    def _make_intervention(
        self,
        concept: str,
        method: str,
        *,
        strength: float,
        topk: int | None,
        normalize_delta: bool,
        train: bool = False,
    ) -> HSpaceIntervention:
        pipe, _, _ = self.pipe_components()
        dtype = torch.float32
        mapper = load_mapper(str(self.alignment_path(concept, method)), self.device, dtype)
        if train:
            mapper.train()
        sae = None
        if method.startswith("casl"):
            sae = load_sae(str(self.sae_path), self.device, torch.float32)
        spec = InterventionSpec(method, strength, topk, normalize_delta)
        return HSpaceIntervention(spec, mapper, sae).to(self.device)

    def train_alignment(self) -> None:
        if not self.sae_path.exists():
            self.train_sae()
        pipe, condition, controller = self.pipe_components()
        latents_cpu, latent_ids = self._load_training_latents()
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        scheduler.set_timesteps(int(self.cfg["num_inference_steps"]), device=self.device)
        previous_scheduler = pipe.scheduler
        pipe.scheduler = scheduler
        clip_loss = DirectionalCLIP(self.device)
        sae = load_sae(str(self.sae_path), self.device, torch.float32)
        channels = int(sae.input_dim)
        steps = int(self.cfg["alignment"]["steps"])
        train_alpha = float(self.cfg["alignment"]["train_alpha"])
        normalize = bool(self.cfg["alignment"]["normalize_delta"])
        try:
            for concept in self.concepts:
                alignment_source_prompt = self.cfg["concepts"][concept].get(
                    "source_prompt", self.cfg["neutral_prompt"]
                )
                text_pair = clip_loss.text_features(
                    alignment_source_prompt,
                    self.cfg["concepts"][concept]["target_prompt"],
                )
                for method in self.methods:
                    output_path = self.alignment_path(concept, method)
                    if output_path.exists() and not self.force:
                        print(f"[reuse] {output_path}")
                        continue
                    input_dim = sae.latent_dim if method.startswith("casl") else channels
                    architecture = "linear"
                    rank = None
                    emb_dim = None
                    if method == "asyrp_cnn":
                        architecture = "asyrp_delta_block"
                        emb_dim = int(self.cfg["alignment"].get("emb_dim", channels))
                        mapper = AsyrpDeltaBlock(channels, emb_dim)
                    elif method == "casl_lowrank":
                        architecture = "factorized"
                        rank = int(self.cfg["alignment"].get("lowrank_rank", 256))
                        mapper = FactorizedAlignmentMap(input_dim, channels, rank)
                    else:
                        mapper = AlignmentMap(input_dim, channels)
                        # A tiny nonzero initialization makes normalized training well-defined.
                        torch.nn.init.normal_(mapper.linear.weight, std=1e-5)
                    mapper = mapper.to(self.device, dtype=torch.float32)
                    intervention = HSpaceIntervention(
                        InterventionSpec(method, train_alpha, None, normalize),
                        mapper,
                        sae if method.startswith("casl") else None,
                    ).to(self.device)
                    method_lr = float(
                        self.cfg["alignment"].get("lr_by_method", {}).get(
                            method, self.cfg["alignment"]["lr"]
                        )
                    )
                    optimizer_name = self.cfg["alignment"].get(
                        "optimizer_by_method", {}
                    ).get(method, "adam")
                    if optimizer_name.lower() == "sgd":
                        optimizer = torch.optim.SGD(
                            mapper.parameters(), lr=method_lr, weight_decay=0
                        )
                    else:
                        optimizer = torch.optim.Adam(mapper.parameters(), lr=method_lr)
                    curve = []
                    started = time.time()
                    iterator = tqdm(range(steps), desc=f"align {concept}/{method}")
                    for step in iterator:
                        image_index = step % len(latents_cpu)
                        choices = [int(value) for value in self.cfg["activation_step_indices"]]
                        step_index = choices[(step // len(latents_cpu)) % len(choices)]
                        timestep = scheduler.timesteps[step_index]
                        latent_x0 = latents_cpu[image_index : image_index + 1].to(
                            self.device, dtype=pipe.unet.dtype
                        )
                        generator = torch.Generator(device=self.device).manual_seed(
                            int(self.cfg["seed"]) + image_index * 1009 + step_index
                        )
                        noise = torch.randn(
                            latent_x0.shape,
                            generator=generator,
                            device=self.device,
                            dtype=latent_x0.dtype,
                        )
                        noisy = scheduler.add_noise(latent_x0, noise, timestep)
                        controller.configure(None)
                        controller.step_index = step_index
                        with torch.no_grad():
                            eps_reference = unet_eps(pipe, noisy, timestep, condition)
                            x0_reference = predict_x0(scheduler, noisy, eps_reference, timestep)
                            image_reference = decode_latents(pipe, x0_reference)
                        controller.configure(intervention, {step_index})
                        controller.step_index = step_index
                        eps_edited = unet_eps(pipe, noisy, timestep, condition)
                        controller.configure(None)
                        x0_edited = predict_x0(scheduler, noisy, eps_edited, timestep)
                        image_edited = decode_latents(pipe, x0_edited)
                        directional, target_clip = clip_loss.alignment_losses(
                            image_reference,
                            image_edited,
                            text_pair,
                            float(self.cfg["alignment"].get("target_temperature", 0.05)),
                        )
                        preservation = F.l1_loss(x0_edited.float(), x0_reference.float())
                        loss = (
                            float(self.cfg["alignment"]["clip_weight"]) * directional
                            + float(self.cfg["alignment"].get("target_clip_weight", 0.0))
                            * target_clip
                            + float(self.cfg["alignment"]["preserve_weight"]) * preservation
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
                        optimizer.step()
                        mapper_finite = all(
                            bool(torch.isfinite(parameter).all())
                            for parameter in mapper.parameters()
                        )
                        if not torch.isfinite(loss) or not mapper_finite:
                            raise FloatingPointError(
                                f"Non-finite alignment state at {concept}/{method} step {step}"
                            )
                        row = {
                            "step": step,
                            "image_id": latent_ids[image_index],
                            "timestep_index": step_index,
                            "loss": float(loss.detach()),
                            "clip_directional": float(directional.detach()),
                            "clip_target": float(target_clip.detach()),
                            "latent_l1": float(preservation.detach()),
                            "grad_norm": float(grad_norm.detach()),
                        }
                        curve.append(row)
                        iterator.set_postfix(
                            loss=f"{row['loss']:.4f}",
                            clip=f"{row['clip_directional']:.3f}",
                            target=f"{row['clip_target']:.3f}",
                            l1=f"{row['latent_l1']:.4f}",
                        )
                        del eps_reference, eps_edited, x0_reference, x0_edited
                        del image_reference, image_edited, loss
                    atomic_torch_save(
                        {
                            "state_dict": {
                                key: value.detach().cpu() for key, value in mapper.state_dict().items()
                            },
                            "input_dim": input_dim,
                            "output_dim": channels,
                            "architecture": architecture,
                            "rank": rank,
                            "emb_dim": emb_dim,
                            "parameter_count": sum(
                                parameter.numel() for parameter in mapper.parameters()
                            ),
                            "method": method,
                            "concept": concept,
                            "config": self.cfg["alignment"],
                            "neutral_prompt": self.cfg["neutral_prompt"],
                            "alignment_source_prompt": alignment_source_prompt,
                            "target_prompt": self.cfg["concepts"][concept]["target_prompt"],
                            "curve": curve,
                            "training_seconds": time.time() - started,
                        },
                        output_path,
                    )
                    write_json(curve, self.result_dir / "alignment" / concept / f"{method}_curve.json")
                    append_event(
                        self.cfg,
                        "alignment_complete",
                        concept=concept,
                        method=method,
                        path=str(output_path),
                        seconds=time.time() - started,
                        final=curve[-1],
                    )
                    del mapper, intervention, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            pipe.scheduler = previous_scheduler
            del clip_loss
            gc.collect()
            torch.cuda.empty_cache()

    def _iter_edit_configs(
        self,
        *,
        strengths: list[float] | None = None,
        topks: list[int] | None = None,
    ) -> Iterable[tuple[str, str, list[int], float, int | None]]:
        default_strengths = [
            float(value) for value in self.cfg["edit"]["strengths"]
        ]
        strengths_by_method = self.cfg["edit"].get("strengths_by_method", {})
        topks = topks or [int(value) for value in self.cfg["edit"]["topks"]]
        for concept in self.concepts:
            for method in self.methods:
                method_strengths = (
                    strengths
                    if strengths is not None
                    else [
                        float(value)
                        for value in strengths_by_method.get(method, default_strengths)
                    ]
                )
                method_topks: list[int | None] = (
                    topks if method.startswith("casl") else [None]
                )
                for bounds in self.cfg["edit"]["timestep_ranges"]:
                    for topk in method_topks:
                        for strength in method_strengths:
                            yield concept, method, bounds, strength, topk

    def edit(
        self,
        *,
        strengths: list[float] | None = None,
        topks: list[int] | None = None,
    ) -> None:
        pipe, condition, controller = self.pipe_components()
        records = []
        for concept, method, bounds, strength, topk in self._iter_edit_configs(
            strengths=strengths, topks=topks
        ):
            if not self.alignment_path(concept, method).exists():
                raise FileNotFoundError(self.alignment_path(concept, method))
            topk_slug = f"topk{topk}" if topk is not None else "dense"
            output_dir = (
                self.result_dir
                / "edits"
                / concept
                / method
                / range_slug(bounds)
                / topk_slug
                / strength_slug(strength)
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            intervention = self._make_intervention(
                concept,
                method,
                strength=strength,
                topk=topk,
                normalize_delta=bool(
                    self.cfg["edit"].get("normalize_delta_by_method", {}).get(
                        method,
                        self.cfg["edit"].get(
                            "normalize_delta",
                            self.cfg["alignment"]["normalize_delta"],
                        ),
                    )
                ),
            )
            active = set(range(int(bounds[0]), int(bounds[1]) + 1))
            started = time.time()
            pending: list[str] = []
            for image_id in test_ids(self.cfg):
                output_path = output_dir / image_id
                if output_path.exists() and not self.force:
                    records.append(
                        {
                            "image_id": image_id,
                            "concept": concept,
                            "method": method,
                            "range": list(bounds),
                            "topk": topk,
                            "strength": strength,
                            "path": str(output_path),
                            "status": "reused",
                        }
                    )
                else:
                    pending.append(image_id)
            edit_batch_size = int(self.cfg["edit"].get("batch_size", 1))
            if float(strength) == 0:
                for image_id in pending:
                    output_path = output_dir / image_id
                    shutil.copy2(self.result_dir / "reconstruction" / image_id, output_path)
                    records.append(
                        {
                            "image_id": image_id,
                            "concept": concept,
                            "method": method,
                            "range": list(bounds),
                            "topk": topk,
                            "strength": strength,
                            "path": str(output_path),
                            "status": "copied_zero",
                        }
                    )
                pending = []
            for image_ids in tqdm(
                list(batched(pending, edit_batch_size)),
                desc=f"edit {concept}/{method}/{range_slug(bounds)}/{topk_slug}/{strength_slug(strength)}",
            ):
                latent_xT = torch.cat(
                    [
                        torch.load(
                            self.inversion_path(image_id),
                            map_location="cpu",
                            weights_only=False,
                        )["latent_xT"]
                        for image_id in image_ids
                    ]
                ).to(self.device, dtype=pipe.unet.dtype)
                edited_latent = denoise_latent(
                    pipe,
                    latent_xT,
                    condition,
                    int(self.cfg["num_inference_steps"]),
                    controller,
                    intervention,
                    active,
                )
                edited = decode_latents(pipe, edited_latent)
                if not torch.isfinite(edited).all():
                    raise FloatingPointError(
                        f"Non-finite edit: {concept}/{method}/{strength}/{image_ids}"
                    )
                for local_index, image_id in enumerate(image_ids):
                    output_path = output_dir / image_id
                    tensor_to_pil(edited[local_index]).save(output_path, quality=95)
                    records.append(
                        {
                            "image_id": image_id,
                            "concept": concept,
                            "method": method,
                            "range": list(bounds),
                            "topk": topk,
                            "strength": strength,
                            "path": str(output_path),
                            "status": "complete",
                        }
                    )
            append_event(
                self.cfg,
                "edit_configuration_complete",
                concept=concept,
                method=method,
                timestep_range=list(bounds),
                topk=topk,
                strength=strength,
                seconds=time.time() - started,
            )
            del intervention
            gc.collect()
            torch.cuda.empty_cache()
        write_json(records, self.result_dir / "edit_manifest.json")

    def verify(self) -> None:
        pipe, condition, controller = self.pipe_components()
        first_id = test_ids(self.cfg)[0]
        if not self.inversion_path(first_id).exists():
            self.invert()
        concept = self.concepts[0]
        method = self.methods[0]
        if not self.alignment_path(concept, method).exists():
            self.train_alignment()
        inversion = torch.load(self.inversion_path(first_id), map_location="cpu", weights_only=False)
        latent_xT = inversion["latent_xT"].to(self.device, dtype=pipe.unet.dtype)
        bounds = self.cfg["edit"]["timestep_ranges"][0]
        active = set(range(int(bounds[0]), int(bounds[1]) + 1))
        controller.call_count = 0
        reference = denoise_latent(
            pipe,
            latent_xT,
            condition,
            int(self.cfg["num_inference_steps"]),
            controller,
        )
        calls_without = controller.call_count
        zero_intervention = self._make_intervention(
            concept,
            method,
            strength=0.0,
            topk=int(self.cfg["edit"]["topks"][0]) if method == "casl" else None,
            normalize_delta=bool(self.cfg["alignment"]["normalize_delta"]),
        )
        controller.call_count = 0
        zero = denoise_latent(
            pipe,
            latent_xT,
            condition,
            int(self.cfg["num_inference_steps"]),
            controller,
            zero_intervention,
            active,
        )
        calls_with = controller.call_count
        max_abs = float((reference.float() - zero.float()).abs().max())
        report = {
            "image_id": first_id,
            "hook_calls_without_intervention": calls_without,
            "hook_calls_with_intervention": calls_with,
            "expected_calls": int(self.cfg["num_inference_steps"]),
            "zero_intervention_max_abs_latent_error": max_abs,
            "zero_intervention_exact": bool(torch.equal(reference, zero)),
            "reference_shape": list(reference.shape),
            "finite": bool(torch.isfinite(reference).all() and torch.isfinite(zero).all()),
            "same_inversion_path": str(self.inversion_path(first_id)),
        }
        if calls_without != int(self.cfg["num_inference_steps"]) or calls_with != int(
            self.cfg["num_inference_steps"]
        ):
            raise RuntimeError(f"Hook call count mismatch: {report}")
        if max_abs != 0:
            raise RuntimeError(f"Zero intervention does not match inversion-only: {report}")
        if not report["finite"]:
            raise FloatingPointError(f"Verification found NaN/Inf: {report}")
        write_json(report, self.result_dir / "verification.json")
        append_event(self.cfg, "verification_complete", **report)
        print(json.dumps(report, indent=2))

    def all(
        self,
        *,
        strengths: list[float] | None = None,
        topks: list[int] | None = None,
    ) -> None:
        write_json(
            {
                "config": self.cfg,
                "device": str(self.device),
                "concepts": self.concepts,
                "methods": self.methods,
                "torch_version": torch.__version__,
                "cuda": torch.version.cuda,
            },
            self.result_dir / "run_manifest.json",
        )
        self.collect()
        self.train_sae()
        self.invert()
        self.train_alignment()
        self.verify()
        self.edit(strengths=strengths, topks=topks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["collect", "train-sae", "invert", "train-align", "verify", "edit", "all"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--concepts", help="Comma-separated config concept names")
    parser.add_argument(
        "--methods",
        default="casl,dense",
        help="casl,casl_lowrank,dense,asyrp_cnn",
    )
    parser.add_argument("--strengths", help="Optional comma-separated strength override")
    parser.add_argument("--topks", help="Optional comma-separated CASL top-k override")
    parser.add_argument(
        "--edit-batch-size",
        type=int,
        help="Optional edit batch-size override for memory-constrained shared GPUs",
    )
    parser.add_argument(
        "--result-dir",
        help="Optional result-directory override, useful for isolated ablations",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        help="Optional fixed test-image count override",
    )
    parser.add_argument(
        "--normalize-delta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override edit-time delta RMS normalization for the selected methods",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    concepts = args.concepts.split(",") if args.concepts else None
    methods = [value for value in args.methods.split(",") if value]
    if args.result_dir:
        cfg["result_dir"] = str(Path(args.result_dir).resolve())
    if args.test_count is not None:
        if args.test_count < 1:
            raise ValueError("--test-count must be at least 1")
        cfg["test_count"] = args.test_count
    if args.edit_batch_size is not None:
        if args.edit_batch_size < 1:
            raise ValueError("--edit-batch-size must be at least 1")
        cfg["edit"]["batch_size"] = args.edit_batch_size
    if args.normalize_delta is not None:
        normalization = cfg["edit"].setdefault("normalize_delta_by_method", {})
        for method in methods:
            normalization[method] = bool(args.normalize_delta)
    experiment = Experiment(
        cfg,
        args.device,
        concepts=concepts,
        methods=methods,
        force=args.force,
    )
    command = args.command.replace("-", "_")
    if command == "train_align":
        command = "train_alignment"
    if command in {"edit", "all"}:
        getattr(experiment, command)(
            strengths=parse_csv_floats(args.strengths),
            topks=parse_csv_ints(args.topks),
        )
    else:
        getattr(experiment, command)()


if __name__ == "__main__":
    main()
