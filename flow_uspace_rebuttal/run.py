from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


class SAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, activation_topk: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_topk = activation_topk
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.normal_(self.decoder.weight, std=input_dim**-0.5)
        self.normalize_decoder()

    def encode(self, x):
        z = F.relu(self.encoder(x - self.bias))
        if self.activation_topk is not None and self.activation_topk < z.shape[-1]:
            indices = z.topk(self.activation_topk, dim=-1).indices
            mask = torch.zeros_like(z).scatter_(-1, indices, 1)
            z = z * mask
        return z

    def decode(self, z):
        return self.decoder(z) + self.bias

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    @torch.no_grad()
    def remove_parallel_grad(self):
        if self.decoder.weight.grad is None:
            return
        w, g = self.decoder.weight.data, self.decoder.weight.grad
        self.decoder.weight.grad = g - w * (w * g).sum(0, keepdim=True)


class CLIPDirection(nn.Module):
    def __init__(self, device):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        )

    @torch.no_grad()
    def text_direction(self, source: str, target: str):
        tokens = self.tokenizer([source, target], padding=True, return_tensors="pt")
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        feat = F.normalize(self.model.get_text_features(**tokens), dim=-1)
        return F.normalize(feat[1] - feat[0], dim=-1)

    def image_features(self, images):
        x = F.interpolate(images, (224, 224), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        return F.normalize(self.model.get_image_features(pixel_values=x), dim=-1)

    def loss(self, before, after, text_direction):
        direction = F.normalize(self.image_features(after) - self.image_features(before), dim=-1)
        return 1 - (direction * text_direction[None]).sum(-1).mean()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_cfg(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def list_image_paths(cfg):
    paths = sorted(resolve(cfg["source_dir"]).glob("*"))
    return [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def load_path_batch(cfg, paths):
    tensors = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize(
            (int(cfg["resolution"]), int(cfg["resolution"])), Image.Resampling.LANCZOS
        )
        tensors.append(TF.pil_to_tensor(image).float() / 127.5 - 1)
    return torch.stack(tensors)


def load_images(cfg, limit=None, offset=0):
    ids_file = cfg.get("edit", {}).get("test_ids_file")
    if ids_file:
        with resolve(ids_file).open() as handle:
            image_ids = [line.strip() for line in handle if line.strip()]
        if limit is not None:
            image_ids = image_ids[: int(limit)]
        paths = [resolve(cfg["source_dir"]) / image_id for image_id in image_ids]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing test image: {missing[0]}")
        return paths, load_path_batch(cfg, paths)
    paths = list_image_paths(cfg)
    requested = limit or int(cfg["image_limit"])
    paths = paths[int(offset) : int(offset) + requested]
    if not paths:
        raise FileNotFoundError(f"No images under {resolve(cfg['source_dir'])}")
    return paths, load_path_batch(cfg, paths)


def load_backbone(cfg, device):
    lfm_root = resolve(cfg["lfm_root"])
    sys.path.insert(0, str(lfm_root))
    from diffusers.models import AutoencoderKL
    from models.DiT import DiT_models

    model = DiT_models["DiT-L/2"](
        img_resolution=int(cfg["resolution"]) // 8,
        in_channels=4,
        num_classes=1,
        label_dropout=0.0,
    ).to(device)
    state = torch.load(resolve(cfg["checkpoint"]), map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(cfg["vae"]).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return model, vae


def encode(vae, images, scale):
    return vae.encode(images).latent_dist.mean * scale


def decode(vae, latents, scale):
    return (vae.decode(latents / scale).sample / 2 + 0.5).clamp(0, 1)


def euler(model, x, start, end, steps, intervention=None, record=False):
    times = torch.linspace(start, end, steps + 1, device=x.device, dtype=x.dtype)
    trajectory = [x.detach().cpu()] if record else None
    for current, nxt in zip(times[:-1], times[1:]):
        batch_t = current.expand(x.shape[0])
        model_input = x
        if intervention is not None:
            model_input = intervention(model_input, float(current))
        velocity = model(batch_t, model_input)
        x = x + (nxt - current) * velocity
        if record:
            trajectory.append(x.detach().cpu())
    return (x, trajectory) if record else x


def save_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.detach().cpu().clamp(0, 1)).save(path)


def collect(cfg, model, vae, device):
    out = resolve(cfg["output_dir"])
    path = out / "uspace_train.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    image_paths = list_image_paths(cfg)[: int(cfg["image_limit"])]
    if not image_paths:
        raise FileNotFoundError(f"No images under {resolve(cfg['source_dir'])}")
    scale = float(cfg["scale_factor"])
    samples = []
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]) + 1)
    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), 16), desc="encode SAE images"):
            batch = load_path_batch(cfg, image_paths[start : start + 16])
            z0 = encode(vae, batch.to(device), scale)
            for _ in range(int(cfg["sae"]["samples_per_image"])):
                t = torch.rand((z0.shape[0], 1, 1, 1), generator=generator, device=device)
                noise = torch.randn(z0.shape, generator=generator, device=device)
                samples.append(((1 - t) * z0 + (1e-5 + (1 - 1e-5) * t) * noise).cpu())
    payload = {
        "u": torch.cat(samples).flatten(1).half(),
        "shape": [4, int(cfg["resolution"]) // 8, int(cfg["resolution"]) // 8],
        "image_ids": [p.name for p in image_paths],
        "definition": "pre-patchification DiT input (u-space)",
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def train_sae(cfg, payload, device):
    out = resolve(cfg["output_dir"])
    suffix = str(cfg["sae"].get("name", "")).strip()
    stem = "sae" if not suffix else f"sae_{suffix}"
    path = out / f"{stem}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        sae = SAE(
            saved["input_dim"],
            saved["latent_dim"],
            saved.get("activation_topk"),
        )
        sae.load_state_dict(saved["state_dict"])
        return sae.to(device).eval()
    data = payload["u"].float()
    if cfg["sae"].get("input_mode", "flatten") == "tokenwise":
        channels, height, width = payload["shape"]
        data = (
            data.reshape(-1, channels, height, width)
            .permute(0, 2, 3, 1)
            .reshape(-1, channels)
        )
    input_dim = data.shape[1]
    latent_dim = input_dim * int(cfg["sae"]["expansion"])
    sae = SAE(input_dim, latent_dim, cfg["sae"].get("activation_topk")).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=float(cfg["sae"]["lr"]))
    curve = []
    gen = torch.Generator().manual_seed(int(cfg["seed"]) + 2)
    permutation = torch.randperm(data.shape[0], generator=gen)
    val_count = max(1, round(data.shape[0] * float(cfg["sae"].get("val_fraction", 0.1))))
    val_ids = permutation[:val_count]
    train_ids = permutation[val_count:]
    train_data = data[train_ids]
    val_data = data[val_ids]
    train_variance = train_data.var().clamp_min(1e-6)

    @torch.no_grad()
    def evaluate_split(split):
        squared_error = 0.0
        active = 0.0
        count = 0
        eval_batch_size = 262144 if input_dim <= 16 else 256
        for begin in range(0, split.shape[0], eval_batch_size):
            batch = split[begin : begin + eval_batch_size].to(device)
            recon, code = sae(batch)
            squared_error += float(F.mse_loss(recon, batch, reduction="sum"))
            active += float((code > 0).sum())
            count += batch.shape[0]
        mse = squared_error / (count * input_dim)
        variance = float(split.var().clamp_min(1e-6))
        return {
            "mse": mse,
            "l0": active / count,
            "explained_variance": 1 - mse / variance,
        }

    for step in tqdm(range(int(cfg["sae"]["steps"])), desc="u-space SAE"):
        ids = torch.randint(
            train_data.shape[0],
            (min(int(cfg["sae"]["batch_size"]), train_data.shape[0]),),
            generator=gen,
        )
        batch = train_data[ids].to(device)
        recon, code = sae(batch)
        mse = F.mse_loss(recon, batch)
        l1 = code.abs().mean()
        loss = mse / train_variance + float(cfg["sae"]["l1"]) * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1)
        opt.step()
        sae.normalize_decoder()
        if step % 50 == 0 or step + 1 == int(cfg["sae"]["steps"]):
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "mse": float(mse.detach()),
                    "l0": float((code > 0).sum(-1).float().mean()),
                    "explained_variance": float(1 - mse.detach() / train_variance),
                }
            )
    train_metrics = evaluate_split(train_data)
    val_metrics = evaluate_split(val_data)
    torch.save(
        {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "activation_topk": sae.activation_topk,
            "state_dict": {k: v.detach().cpu() for k, v in sae.state_dict().items()},
            "curve": curve,
            "config": cfg["sae"],
            "split": {
                "train_samples": train_data.shape[0],
                "validation_samples": val_data.shape[0],
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
            },
        },
        path,
    )
    with (out / f"{stem}_curve.json").open("w") as f:
        json.dump(curve, f, indent=2)
    with (out / f"{stem}_validation.json").open("w") as f:
        json.dump(
            {
                "train_samples": train_data.shape[0],
                "validation_samples": val_data.shape[0],
                "train": train_metrics,
                "validation": val_metrics,
            },
            f,
            indent=2,
        )
    return sae.eval()


def sparse_shift(sae, direction, topk, shape):
    indices = direction.detach().abs().topk(min(topk, direction.numel())).indices
    selected = torch.zeros_like(direction)
    selected[indices] = direction[indices]
    zero = torch.zeros_like(selected)
    shift = sae.decode(selected[None]) - sae.decode(zero[None])
    if sae.input_dim == shape[0]:
        shift = shift.reshape(1, shape[0], 1, 1).expand(1, *shape)
    else:
        shift = shift.reshape(1, *shape)
    return shift, indices


@torch.no_grad()
def omp_projection(dictionary, target, count):
    """Sparse least-squares initialization of a dense semantic direction."""
    residual = target.clone()
    chosen = []
    coefficients = None
    for _ in range(min(count, dictionary.shape[1])):
        scores = dictionary.T @ residual
        if chosen:
            scores[torch.tensor(chosen, device=scores.device)] = 0
        chosen.append(int(scores.abs().argmax()))
        atoms = dictionary[:, chosen]
        coefficients = torch.linalg.lstsq(atoms, target[:, None]).solution[:, 0]
        residual = target - atoms @ coefficients
    direction = torch.zeros(dictionary.shape[1], device=dictionary.device)
    direction[torch.tensor(chosen, device=dictionary.device)] = coefficients
    return direction, torch.tensor(chosen, device=dictionary.device)


def train_directions(cfg, model, vae, sae, device, selected_concepts=None):
    out = resolve(cfg["output_dir"])
    paths, images = load_images(
        cfg,
        int(cfg["alignment"]["train_images"]),
        int(cfg["alignment"].get("train_offset", 0)),
    )
    images = images.to(device)
    scale = float(cfg["scale_factor"])
    t = float(cfg["alignment"]["t"])
    shape = (4, int(cfg["resolution"]) // 8, int(cfg["resolution"]) // 8)
    with torch.no_grad():
        z0 = encode(vae, images, scale)
        x_t = euler(model, z0, 0.0, t, max(1, round(int(cfg["euler_steps"]) * t)))
        v_ref = model(torch.full((len(images),), t, device=device), x_t)
        pred_ref = x_t - t * v_ref
        image_ref = decode(vae, pred_ref, scale)
        if sae.input_dim == shape[0]:
            token_codes = sae.encode(x_t.permute(0, 2, 3, 1).reshape(-1, shape[0]))
        else:
            token_codes = None
    clip = CLIPDirection(device).to(device)
    records = {}
    concept_items = cfg["alignment"]["concepts"].items()
    if selected_concepts:
        wanted = set(selected_concepts)
        concept_items = [(key, value) for key, value in concept_items if key in wanted]
    casl_variant = cfg["alignment"].get("casl_variant", "straight_through")
    alignment_name = str(cfg["alignment"].get("name", "")).strip()
    for concept, prompts in concept_items:
        concept_dir = out / "alignment" / concept
        concept_dir.mkdir(parents=True, exist_ok=True)
        text_dir = clip.text_direction(prompts["source"], prompts["target"])
        records[concept] = {}
        for method in ("dense", "casl"):
            variant_suffix = casl_variant + (f"_{alignment_name}" if alignment_name else "")
            filename = (
                f"casl_{variant_suffix}.pt"
                if method == "casl" and casl_variant != "straight_through"
                else f"{method}.pt"
            )
            path = concept_dir / filename
            if path.exists():
                records[concept][method] = path
                continue
            dim = math.prod(shape) if method == "dense" else sae.latent_dim
            fixed_support = None
            mapping = None
            if method == "casl" and casl_variant == "mapping":
                if token_codes is None:
                    raise ValueError("mapping CASL requires a token-wise SAE")
                mapping = nn.Linear(sae.latent_dim, shape[0]).to(device)
                nn.init.normal_(mapping.weight, std=1e-4)
                nn.init.zeros_(mapping.bias)
                parameters = list(mapping.parameters())
                direction = None
            elif method == "casl" and casl_variant == "omp":
                dense_saved = torch.load(
                    concept_dir / "dense.pt", map_location=device, weights_only=False
                )
                initial, fixed_support = omp_projection(
                    sae.decoder.weight,
                    dense_saved["direction"].to(device),
                    int(cfg["alignment"]["topk"]),
                )
                direction = nn.Parameter(initial)
                parameters = [direction]
            else:
                direction = nn.Parameter(torch.randn(dim, device=device) * 1e-4)
                parameters = [direction]
            opt = torch.optim.Adam(parameters, lr=float(cfg["alignment"]["lr"]))
            curve = []
            for step in tqdm(range(int(cfg["alignment"]["steps"])), desc=f"{concept}/{method}"):
                if method == "dense":
                    shift = direction.reshape(1, *shape)
                    indices = None
                elif casl_variant == "mapping":
                    shift = (
                        mapping(token_codes)
                        .reshape(len(images), shape[1], shape[2], shape[0])
                        .permute(0, 3, 1, 2)
                    )
                    indices = mapping.weight.detach().square().sum(0).topk(
                        min(int(cfg["alignment"]["topk"]), sae.latent_dim)
                    ).indices
                elif casl_variant == "omp":
                    mask = torch.zeros_like(direction)
                    mask[fixed_support] = 1
                    selected = direction * mask
                    zero = torch.zeros_like(direction)
                    indices = fixed_support
                    shift = sae.decode(selected[None]) - sae.decode(zero[None])
                    if sae.input_dim == shape[0]:
                        shift = shift.reshape(1, shape[0], 1, 1).expand(1, *shape)
                    else:
                        shift = shift.reshape(1, *shape)
                else:
                    # Straight-through top-k: the forward pass is the exact sparse
                    # intervention evaluated at test time, while the backward pass
                    # sends gradients to all alignment coordinates. This avoids
                    # hard-top-k gradient starvation without training a dense edit.
                    zero = torch.zeros_like(direction)
                    count = min(int(cfg["alignment"]["topk"]), direction.numel())
                    indices = direction.detach().abs().topk(count).indices
                    selected = torch.zeros_like(direction)
                    selected[indices] = direction[indices]
                    straight_through = direction + (selected - direction).detach()
                    shift = sae.decode(straight_through[None]) - sae.decode(zero[None])
                    if sae.input_dim == shape[0]:
                        shift = shift.reshape(1, shape[0], 1, 1).expand(1, *shape)
                    else:
                        shift = shift.reshape(1, *shape)
                rms = shift.square().mean().sqrt().clamp_min(1e-6)
                shift = shift / rms * float(cfg["alignment"]["train_strength"])
                v_edit = model(torch.full((len(images),), t, device=device), x_t + shift)
                pred_edit = x_t - t * v_edit
                image_edit = decode(vae, pred_edit, scale)
                clip_loss = clip.loss(image_ref, image_edit, text_dir)
                preserve = F.l1_loss(pred_edit, pred_ref)
                loss = clip_loss + float(cfg["alignment"]["preserve_weight"]) * preserve
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                opt.step()
                if step % 10 == 0 or step + 1 == int(cfg["alignment"]["steps"]):
                    curve.append(
                        {
                            "step": step,
                            "loss": float(loss.detach()),
                            "clip": float(clip_loss.detach()),
                            "latent_l1": float(preserve.detach()),
                        }
                    )
            checkpoint = {
                    "direction": None if direction is None else direction.detach().cpu(),
                    "method": method,
                    "concept": concept,
                    "topk_train": cfg["alignment"]["topk"] if method == "casl" else None,
                    "selected_train": indices.detach().cpu() if indices is not None else None,
                    "casl_variant": casl_variant if method == "casl" else None,
                    "prompts": prompts,
                    "curve": curve,
                }
            if mapping is not None:
                checkpoint["direction"] = None
                checkpoint["mapping_state_dict"] = {
                    key: value.detach().cpu() for key, value in mapping.state_dict().items()
                }
                checkpoint["mapping_in"] = sae.latent_dim
                checkpoint["mapping_out"] = shape[0]
                checkpoint["importance"] = mapping.weight.detach().square().sum(0).cpu()
            torch.save(checkpoint, path)
            records[concept][method] = path
    return records


def edit(cfg, model, vae, sae, directions, device, selected_concepts=None):
    out = resolve(cfg["output_dir"])
    paths, images = load_images(
        cfg,
        int(cfg["edit"]["test_images"]),
        int(cfg["edit"].get("test_offset", 0)),
    )
    images = images.to(device)
    scale = float(cfg["scale_factor"])
    steps = int(cfg["euler_steps"])
    shape = (4, int(cfg["resolution"]) // 8, int(cfg["resolution"]) // 8)
    rows = []
    with torch.no_grad():
        z0 = encode(vae, images, scale)
        noise, _ = euler(model, z0, 0.0, 1.0, steps, record=True)
        recon = euler(model, noise, 1.0, 0.0, steps)
        recon_images = decode(vae, recon, scale)
        for index, image_id in enumerate(paths):
            save_image(images[index] / 2 + 0.5, out / "source" / image_id.name)
            save_image(recon_images[index], out / "inversion" / image_id.name)
        methods = cfg["edit"].get("methods", ["dense", "casl"])
        for concept in (selected_concepts or cfg["edit"]["concepts"]):
            for method in methods:
                saved = torch.load(directions[concept][method], map_location=device, weights_only=False)
                is_mapping = method == "casl" and saved.get("casl_variant") == "mapping"
                if is_mapping:
                    mapping = nn.Linear(saved["mapping_in"], saved["mapping_out"]).to(device)
                    mapping.load_state_dict(saved["mapping_state_dict"])
                    mapping.eval()
                    importance = saved["importance"].to(device)
                    direction = None
                else:
                    direction = saved["direction"].to(device)
                topks = cfg["edit"]["topks"] if method == "casl" else [None]
                for topk in topks:
                    selected_indices = None
                    if is_mapping:
                        selected_indices = importance.topk(min(int(topk), importance.numel())).indices
                        raw_shift = None
                    elif method == "dense":
                        raw_shift = direction.reshape(1, *shape)
                    else:
                        raw_shift, _ = sparse_shift(sae, direction, int(topk), shape)
                    if raw_shift is not None:
                        raw_shift = (
                            raw_shift
                            / raw_shift.square().mean().sqrt().clamp_min(1e-6)
                            * float(cfg["edit"].get("base_rms", 1.0))
                        )
                    strengths = cfg["edit"].get(
                        f"{method}_strengths", cfg["edit"]["strengths"]
                    )
                    for strength in strengths:
                        def intervention(
                            x,
                            time_value,
                            shift=raw_shift,
                            alpha=float(strength),
                            indices=selected_indices,
                        ):
                            if float(cfg["edit"]["edit_t_min"]) <= time_value <= float(
                                cfg["edit"]["edit_t_max"]
                            ):
                                if indices is not None:
                                    tokens = x.permute(0, 2, 3, 1).reshape(-1, shape[0])
                                    codes = sae.encode(tokens)
                                    sparse_codes = torch.zeros_like(codes)
                                    sparse_codes[:, indices] = codes[:, indices]
                                    shift = (
                                        mapping(sparse_codes)
                                        .reshape(x.shape[0], shape[1], shape[2], shape[0])
                                        .permute(0, 3, 1, 2)
                                    )
                                    shift = (
                                        shift
                                        / shift.square().mean().sqrt().clamp_min(1e-6)
                                        * float(cfg["edit"].get("base_rms", 1.0))
                                    )
                                return x + alpha * shift
                            return x

                        edited = euler(model, noise, 1.0, 0.0, steps, intervention)
                        decoded = decode(vae, edited, scale)
                        slug = f"topk{topk}" if topk is not None else "dense"
                        strength_slug = str(strength).replace("-", "m").replace(".", "p")
                        for index, image_id in enumerate(paths):
                            edit_root = out / "edits"
                            output_tag = str(cfg["edit"].get("manifest_tag", "")).strip()
                            if output_tag:
                                edit_root = edit_root / output_tag
                            target = (
                                edit_root
                                / concept
                                / method
                                / slug
                                / f"strength_{strength_slug}"
                                / image_id.name
                            )
                            save_image(decoded[index], target)
                            rows.append(
                                {
                                    "image_id": image_id.name,
                                    "concept": concept,
                                    "method": method,
                                    "topk": topk,
                                    "strength": strength,
                                    "path": str(target.relative_to(ROOT)),
                                }
                            )
    manifest_name = (
        "manifest.csv"
        if not selected_concepts
        else "manifest_" + "_".join(selected_concepts) + ".csv"
    )
    manifest_tag = str(cfg["edit"].get("manifest_tag", "")).strip()
    if manifest_tag:
        manifest_name = manifest_name.removesuffix(".csv") + f"_{manifest_tag}.csv"
    if set(methods) != {"dense", "casl"}:
        manifest_name = manifest_name.removesuffix(".csv") + "_" + "_".join(methods) + ".csv"
    with (out / manifest_name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def verify(cfg, model, vae, sae, device):
    _, images = load_images(cfg, 1)
    scale = float(cfg["scale_factor"])
    with torch.no_grad():
        z0 = encode(vae, images.to(device), scale)
        noise = euler(model, z0, 0, 1, int(cfg["euler_steps"]))
        base = euler(model, noise, 1, 0, int(cfg["euler_steps"]))
        zero = euler(
            model,
            noise,
            1,
            0,
            int(cfg["euler_steps"]),
            lambda x, _: x + torch.zeros_like(x),
        )
    result = {
        "zero_intervention_max_abs": float((base - zero).abs().max()),
        "finite": bool(torch.isfinite(base).all()),
        "u_shape": list(z0.shape),
        "patch_tokens": int(model.x_embedder.num_patches),
        "sae_input": sae.input_dim,
        "sae_latent": sae.latent_dim,
    }
    with (resolve(cfg["output_dir"]) / "verification.json").open("w") as f:
        json.dump(result, f, indent=2)
    if result["zero_intervention_max_abs"] != 0 or not result["finite"]:
        raise RuntimeError(result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["collect", "sae", "align", "verify", "edit", "all"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_smoke.yaml")))
    parser.add_argument("--concepts", nargs="+")
    parser.add_argument("--test-images", type=int)
    parser.add_argument("--sae-l1", type=float)
    parser.add_argument("--sae-name")
    parser.add_argument("--sae-activation-topk", type=int)
    parser.add_argument("--casl-variant", choices=["straight_through", "omp", "mapping"])
    parser.add_argument("--sae-input-mode", choices=["flatten", "tokenwise"])
    parser.add_argument("--alignment-name")
    parser.add_argument("--train-strength", type=float)
    parser.add_argument("--topks", type=int, nargs="+")
    parser.add_argument("--dense-strengths", type=float, nargs="+")
    parser.add_argument("--casl-strengths", type=float, nargs="+")
    parser.add_argument("--methods", choices=["dense", "casl"], nargs="+")
    parser.add_argument("--test-ids-file")
    parser.add_argument("--manifest-tag")
    parser.add_argument("--euler-steps", type=int)
    args = parser.parse_args()
    cfg = load_cfg(Path(args.config))
    if args.test_images is not None:
        cfg["edit"]["test_images"] = args.test_images
    if args.sae_l1 is not None:
        cfg["sae"]["l1"] = args.sae_l1
    if args.sae_name is not None:
        cfg["sae"]["name"] = args.sae_name
    if args.sae_activation_topk is not None:
        cfg["sae"]["activation_topk"] = args.sae_activation_topk
    if args.casl_variant is not None:
        cfg["alignment"]["casl_variant"] = args.casl_variant
    if args.sae_input_mode is not None:
        cfg["sae"]["input_mode"] = args.sae_input_mode
    if args.alignment_name is not None:
        cfg["alignment"]["name"] = args.alignment_name
    if args.train_strength is not None:
        cfg["alignment"]["train_strength"] = args.train_strength
    if args.topks is not None:
        cfg["edit"]["topks"] = args.topks
    if args.dense_strengths is not None:
        cfg["edit"]["dense_strengths"] = args.dense_strengths
    if args.casl_strengths is not None:
        cfg["edit"]["casl_strengths"] = args.casl_strengths
    if args.methods is not None:
        cfg["edit"]["methods"] = args.methods
    if args.test_ids_file is not None:
        cfg["edit"]["test_ids_file"] = args.test_ids_file
    if args.manifest_tag is not None:
        cfg["edit"]["manifest_tag"] = args.manifest_tag
    if args.euler_steps is not None:
        cfg["euler_steps"] = args.euler_steps
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    device = torch.device("cuda")
    model, vae = load_backbone(cfg, device)
    sae_suffix = str(cfg["sae"].get("name", "")).strip()
    sae_stem = "sae" if not sae_suffix else f"sae_{sae_suffix}"
    saved_sae = resolve(cfg["output_dir"]) / f"{sae_stem}.pt"
    # Evaluation/editing only needs the saved SAE. Avoid reloading the multi-GB
    # activation cache on every parallel worker.
    payload = (
        None
        if args.stage not in {"collect", "sae", "all"} and saved_sae.exists()
        else collect(cfg, model, vae, device)
    )
    if args.stage == "collect":
        return
    sae = train_sae(cfg, payload, device)
    if args.stage == "sae":
        return
    directions = train_directions(cfg, model, vae, sae, device, args.concepts)
    if args.stage == "align":
        return
    verify(cfg, model, vae, sae, device)
    if args.stage == "verify":
        return
    edit(cfg, model, vae, sae, directions, device, args.concepts)


if __name__ == "__main__":
    main()
