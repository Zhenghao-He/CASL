from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .experiment import strength_slug
from .sdxl_core import (
    MidBlockController,
    decode_latents,
    denoise_latent,
    load_pipeline,
    make_condition,
    release_text_encoders,
)
from .utils import (
    append_event,
    batched,
    file_sha256,
    load_config,
    tensor_to_pil,
    test_ids,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDER_SOURCE = REPO_ROOT / "third_party" / "sliders" / "trainscripts" / "textsliders"
sys.path.insert(0, str(SLIDER_SOURCE))
import lora as concept_slider_lora  # noqa: E402

LoRANetwork = concept_slider_lora.LoRANetwork


def slider_config(cfg: dict, concept: str) -> dict:
    external = cfg.get("concept_slider", {})
    if concept not in external.get("weights", {}):
        raise KeyError(f"No official Concept Slider weight configured for {concept}")
    return {
        "weight": Path(external["weights"][concept]),
        "sign": float(external.get("signs", {}).get(concept, 1.0)),
        "strengths": [float(value) for value in external["strengths"]],
    }


def merge_manifests(cfg: dict) -> None:
    result_dir = Path(cfg["result_dir"])
    core_path = result_dir / "edit_manifest.json"
    records = []
    if core_path.is_file():
        records.extend(json.loads(core_path.read_text(encoding="utf-8")))
    for path in sorted(result_dir.glob("concept_slider_manifest_*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    unique = {
        (
            row["concept"],
            row["method"],
            str(row["range"]),
            str(row.get("topk")),
            float(row["strength"]),
            row["image_id"],
        ): row
        for row in records
    }
    merged = list(unique.values())
    write_json(merged, core_path)
    append_event(cfg, "manifests_merged", rows=len(merged))
    print(f"[saved] {core_path} ({len(merged)} rows)")


def edit(cfg: dict, device_name: str, concept: str, force: bool) -> None:
    setup = slider_config(cfg, concept)
    weight_path = setup["weight"]
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    device = torch.device(device_name)
    pipe = load_pipeline(cfg["model_id"], device)
    condition = make_condition(pipe, cfg["neutral_prompt"], int(cfg["resolution"]), device)
    controller = MidBlockController(pipe.unet.mid_block)
    release_text_encoders(pipe)

    # The released weights use the official c3lier setup. The official
    # training/inference code extends this module-level target list before
    # constructing LoRANetwork.
    for module_name in concept_slider_lora.UNET_TARGET_REPLACE_MODULE_CONV:
        if module_name not in concept_slider_lora.DEFAULT_TARGET_REPLACE:
            concept_slider_lora.DEFAULT_TARGET_REPLACE.append(module_name)
    network = LoRANetwork(
        pipe.unet,
        rank=4,
        multiplier=0.0,
        alpha=1.0,
        train_method="noxattn",
    ).to(device=device, dtype=pipe.unet.dtype)
    payload = torch.load(weight_path, map_location="cpu", weights_only=False)
    network.load_state_dict(payload, strict=True)
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    result_dir = Path(cfg["result_dir"])
    records = []
    started = time.time()
    for strength in setup["strengths"]:
        applied_scale = setup["sign"] * strength
        output_dir = (
            result_dir
            / "edits"
            / concept
            / "concept_slider"
            / "steps00-49"
            / "dense"
            / strength_slug(strength)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        pending: list[str] = []
        for image_id in test_ids(cfg):
            output_path = output_dir / image_id
            if output_path.is_file() and not force:
                status = "reused"
            elif strength == 0:
                # The zero-slider path is exactly the already verified inversion
                # reconstruction, with no extra JPEG round trip.
                shutil.copy2(result_dir / "reconstruction" / image_id, output_path)
                status = "copied_zero"
            else:
                pending.append(image_id)
                continue
            records.append(
                {
                    "image_id": image_id,
                    "concept": concept,
                    "method": "concept_slider",
                    "range": [0, int(cfg["num_inference_steps"]) - 1],
                    "topk": None,
                    "strength": strength,
                    "applied_scale": applied_scale,
                    "path": str(output_path),
                    "status": status,
                }
            )
        batch_size = int(cfg["edit"].get("batch_size", 1))
        for image_ids in tqdm(
            list(batched(pending, batch_size)),
            desc=f"Concept Slider {concept}/scale{applied_scale:g}",
        ):
            latent_xT = torch.cat(
                [
                    torch.load(
                        Path(cfg["work_dir"])
                        / "inversions"
                        / f"{Path(image_id).stem}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )["latent_xT"]
                    for image_id in image_ids
                ]
            ).to(device, dtype=pipe.unet.dtype)
            network.set_lora_slider(applied_scale)
            with network:
                edited_latent = denoise_latent(
                    pipe,
                    latent_xT,
                    condition,
                    int(cfg["num_inference_steps"]),
                    controller,
                )
            if not torch.isfinite(edited_latent).all():
                raise FloatingPointError(
                    f"Non-finite Concept Slider output: {concept}/{image_ids}/{strength}"
                )
            edited = decode_latents(pipe, edited_latent)
            for local_index, image_id in enumerate(image_ids):
                output_path = output_dir / image_id
                tensor_to_pil(edited[local_index]).save(output_path, quality=95)
                records.append(
                    {
                        "image_id": image_id,
                        "concept": concept,
                        "method": "concept_slider",
                        "range": [0, int(cfg["num_inference_steps"]) - 1],
                        "topk": None,
                        "strength": strength,
                        "applied_scale": applied_scale,
                        "path": str(output_path),
                        "status": "complete",
                    }
                )
    write_json(records, result_dir / f"concept_slider_manifest_{concept}.json")
    append_event(
        cfg,
        "concept_slider_complete",
        concept=concept,
        official_weight=str(weight_path),
        official_weight_sha256=file_sha256(weight_path),
        rows=len(records),
        seconds=time.time() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--concept")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.merge_only:
        merge_manifests(cfg)
        return
    if not args.concept:
        parser.error("--concept is required unless --merge-only is used")
    edit(cfg, args.device, args.concept, args.force)


if __name__ == "__main__":
    main()
