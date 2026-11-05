#!/usr/bin/env python3
"""Evaluate the quick linear-vs-MLP Stage-2 alignment ablation."""

from pathlib import Path
import sys

import lpips
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novelty_ablation.evaluate_attribute_metrics import CLASSIFIERS, CONCEPTS
from sdxl_rebuttal.evaluate import attribute_names, classify_paths


PROMPTS = {
    "smiling": "a portrait photo of a smiling face",
    "young": "a portrait photo of a young person",
    "big_nose": "a portrait photo of a face with a big nose",
    "beards": "a portrait photo of a person with a beard",
    "blond_hair": "a portrait photo of a person with blond hair",
}
SAE_NAME = "delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse"


def image_dir(mapping, concept, kind):
    base = (
        ROOT
        / "runs"
        / f"mapping_{mapping}_{concept}_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
        / SAE_NAME
        / "xblock_test_images"
        / "20"
    )
    deterministic = base / "iter0_ngen50" / kind / "images"
    archived = base / "iter0_ngen50_stochastic" / kind / "images"
    return deterministic if deterministic.is_dir() else archived


def load_lpips_tensor(path):
    image = Image.open(path).convert("RGB").resize((256, 256))
    return pil_to_tensor(image).float().div(127.5).sub(1.0)


def compute_lpips(pairs, device):
    model = lpips.LPIPS(net="alex").eval().to(device)
    result = {}
    with torch.inference_mode():
        for reference, edited in pairs:
            score = model(
                load_lpips_tensor(reference).unsqueeze(0).to(device),
                load_lpips_tensor(edited).unsqueeze(0).to(device),
            )
            result[(reference, edited)] = float(score.item())
    del model
    torch.cuda.empty_cache()
    return result


def compute_clip(paths, device):
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    text_inputs = processor(
        text=[PROMPTS[key] for key in PROMPTS],
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.inference_mode():
        text = model.get_text_features(**text_inputs)
        text = text / text.norm(dim=-1, keepdim=True)
    indices = {name: index for index, name in enumerate(PROMPTS)}
    result = {}
    with torch.inference_mode():
        for path in paths:
            inputs = processor(
                images=Image.open(path).convert("RGB"), return_tensors="pt"
            ).to(device)
            image = model.get_image_features(**inputs)
            image = image / image.norm(dim=-1, keepdim=True)
            similarities = image @ text.T
            result[path] = {
                name: float(similarities[0, index].item())
                for name, index in indices.items()
            }
    return result


def main():
    output_dir = ROOT / "mapping_ablation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for mapping in ("linear", "mlp"):
        for concept in CONCEPTS:
            references = {path.stem: path for path in image_dir(mapping, concept, "origin").glob("*.jpg")}
            edits = {path.stem: path for path in image_dir(mapping, concept, "delta_h").glob("*.jpg")}
            for image_id in sorted(set(references) & set(edits)):
                records.append(
                    {
                        "mapping": mapping,
                        "concept": concept,
                        "image_id": image_id,
                        "reference": str(references[image_id]),
                        "edited": str(edits[image_id]),
                    }
                )
    if len(records) != 80:
        raise RuntimeError(f"Expected 80 records, found {len(records)}")

    device = torch.device("cuda:0")
    paths = sorted({row[key] for row in records for key in ("reference", "edited")})
    predictions = classify_paths(paths, device, batch_size=32)
    pairs = sorted({(row["reference"], row["edited"]) for row in records})
    lpips_scores = compute_lpips(pairs, device)
    clip_scores = compute_clip(sorted({row["edited"] for row in records}), device)
    names = attribute_names()

    rows = []
    for record in records:
        attribute, sign, _ = CONCEPTS[record["concept"]]
        target = names.index(attribute)
        non_target = [index for index in range(len(names)) if index != target]
        row = dict(record)
        row["lpips"] = lpips_scores[(record["reference"], record["edited"])]
        row["clip_score"] = clip_scores[record["edited"]][record["concept"]]
        for classifier in CLASSIFIERS:
            before = predictions[record["reference"]]
            after = predictions[record["edited"]]
            prob_delta = after[f"{classifier}_probs"] - before[f"{classifier}_probs"]
            logit_delta = after[f"{classifier}_logits"] - before[f"{classifier}_logits"]
            row[f"{classifier}_target_delta"] = float(sign * prob_delta[target])
            row[f"{classifier}_target_logit_abs"] = float(abs(logit_delta[target]))
            row[f"{classifier}_nontarget_delta"] = float(
                np.abs(prob_delta[non_target]).mean()
            )
            row[f"{classifier}_nontarget_logit_abs"] = float(
                np.abs(logit_delta[non_target]).mean()
            )
        row["target_delta"] = float(
            np.mean([row[f"{name}_target_delta"] for name in CLASSIFIERS])
        )
        row["nontarget_delta"] = float(
            np.mean([row[f"{name}_nontarget_delta"] for name in CLASSIFIERS])
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    summaries = []
    for (concept, mapping), group in frame.groupby(["concept", "mapping"]):
        result = {
            "concept": concept,
            "mapping": mapping,
            "n": len(group),
            "target_delta": group["target_delta"].mean(),
            "nontarget_delta": group["nontarget_delta"].mean(),
            "epr": np.mean(
                [
                    group[f"{name}_target_logit_abs"].mean()
                    / max(group[f"{name}_nontarget_logit_abs"].mean(), 1e-8)
                    for name in CLASSIFIERS
                ]
            ),
            "clip_score": group["clip_score"].mean(),
            "lpips": group["lpips"].mean(),
        }
        summaries.append(result)
    summary = pd.DataFrame(summaries).sort_values(["concept", "mapping"])
    macro = (
        summary.groupby("mapping")[["target_delta", "nontarget_delta", "epr", "clip_score", "lpips"]]
        .mean()
        .reset_index()
    )
    macro.insert(0, "concept", "Macro")
    frame.to_csv(output_dir / "per_image.csv", index=False)
    pd.concat([summary, macro], ignore_index=True).to_csv(
        output_dir / "summary.csv", index=False
    )
    print(pd.concat([summary, macro], ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
