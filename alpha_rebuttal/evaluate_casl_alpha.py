#!/usr/bin/env python3
"""Minimal CASL alpha sweep for the rebuttal.

All edit-induced quantities use the concept-matched alpha=0 inversion as
reference. OMES/OS/MES describe the fixed selected top-1 representation and
are therefore constant across inference alpha.
"""

import argparse
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

from novelty_ablation.evaluate_attribute_metrics import (
    CLASSIFIERS,
    CONCEPTS,
    casl_directory,
    id_path_map,
)
from sdxl_rebuttal.evaluate import attribute_names, classify_paths


PROMPTS = {
    "smiling": "a portrait photo of a smiling face",
    "young": "a portrait photo of a young person",
    "big_nose": "a portrait photo of a face with a big nose",
    "beards": "a portrait photo of a person with a beard",
    "blond_hair": "a portrait photo of a person with blond hair",
}
STRENGTHS = (0, 32, 64, 96, 128, 160, 192)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument(
        "--omes",
        type=Path,
        default=ROOT
        / "rebuttal_metrics/results/celeba_sae_omes/selection/"
        "selection_omes_per_concept.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "alpha_rebuttal/results/casl_alpha",
    )
    return parser.parse_args()


def load_tensor(path):
    image = Image.open(path).convert("RGB").resize((256, 256))
    return pil_to_tensor(image).float().div(127.5).sub(1.0)


def compute_lpips(pairs, device, batch_size):
    model = lpips.LPIPS(net="alex").eval().to(device)
    values = {}
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            before = torch.stack([load_tensor(a) for a, _ in chunk]).to(device)
            after = torch.stack([load_tensor(b) for _, b in chunk]).to(device)
            scores = model(before, after).flatten().cpu().numpy()
            for pair, score in zip(chunk, scores):
                values[pair] = float(score)
    return values


def compute_clip(paths, prompts, device, batch_size):
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    text_inputs = processor(
        text=[prompts[key] for key in prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.inference_mode():
        text = model.get_text_features(**text_inputs)
        text = text / text.norm(dim=-1, keepdim=True)
    prompt_index = {key: index for index, key in enumerate(prompts)}
    scores = {}
    unique_paths = sorted(set(paths))
    with torch.inference_mode():
        for start in range(0, len(unique_paths), batch_size):
            chunk = unique_paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in chunk]
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            similarities = features @ text.T
            for row, path in enumerate(chunk):
                scores[path] = {
                    key: float(similarities[row, prompt_index[key]].item())
                    for key in prompts
                }
    return scores


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = [f"{value:05d}" for value in range(29501, 29501 + args.n_images)]
    records = []
    for concept in CONCEPTS:
        reference = id_path_map(casl_directory(concept, 0), "casl")
        for alpha in STRENGTHS:
            edited = id_path_map(casl_directory(concept, alpha), "casl")
            for image_id in ids:
                if image_id not in reference or image_id not in edited:
                    continue
                records.append(
                    {
                        "concept": concept,
                        "alpha": alpha,
                        "image_id": image_id,
                        "reference": str(reference[image_id]),
                        "edited": str(edited[image_id]),
                    }
                )
    if not records:
        raise RuntimeError("No CASL alpha outputs found")

    device = torch.device(args.device)
    paths = sorted(
        {row["reference"] for row in records} | {row["edited"] for row in records}
    )
    predictions = classify_paths(paths, device, args.batch_size)
    pairs = sorted({(row["reference"], row["edited"]) for row in records})
    lpips_values = compute_lpips(pairs, device, args.batch_size)
    clip_values = compute_clip(
        [row["edited"] for row in records], PROMPTS, device, args.batch_size
    )

    names = attribute_names()
    rows = []
    for record in records:
        concept = record["concept"]
        attribute, sign, _ = CONCEPTS[concept]
        target = names.index(attribute)
        non_target = [index for index in range(len(names)) if index != target]
        row = dict(record)
        row["lpips"] = lpips_values[(record["reference"], record["edited"])]
        row["clip_score"] = clip_values[record["edited"]][concept]
        for classifier in CLASSIFIERS:
            before = predictions[record["reference"]]
            after = predictions[record["edited"]]
            probability_delta = (
                after[f"{classifier}_probs"] - before[f"{classifier}_probs"]
            )
            logit_delta = (
                after[f"{classifier}_logits"] - before[f"{classifier}_logits"]
            )
            row[f"{classifier}_target_delta"] = float(
                sign * probability_delta[target]
            )
            row[f"{classifier}_target_logit_abs"] = float(
                abs(logit_delta[target])
            )
            row[f"{classifier}_nontarget_delta"] = float(
                np.abs(probability_delta[non_target]).mean()
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
    omes = pd.read_csv(args.omes)
    omes = omes[(omes["selection"] == "CASL") & (omes["k"] == 1)].copy()
    concept_names = {
        "smiling": "Smiling",
        "young": "Young",
        "big_nose": "Big_Nose",
        "beards": "Beards",
        "blond_hair": "Blond_Hair",
    }
    omes_lookup = omes.set_index("concept")[["OS", "MES", "OMES"]].to_dict("index")
    summary_rows = []
    for (concept, alpha), group in frame.groupby(["concept", "alpha"]):
        result = {
            "concept": concept,
            "alpha": alpha,
            "n": len(group),
            "target_delta": group["target_delta"].mean(),
            "nontarget_delta": group["nontarget_delta"].mean(),
            "clip_score": group["clip_score"].mean(),
            "lpips": group["lpips"].mean(),
        }
        eprs = []
        for classifier in CLASSIFIERS:
            numerator = group[f"{classifier}_target_logit_abs"].mean()
            denominator = group[f"{classifier}_nontarget_logit_abs"].mean()
            eprs.append(float(numerator / max(float(denominator), 1e-8)))
        result["epr"] = float(np.mean(eprs))
        result.update(omes_lookup[concept_names[concept]])
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows).sort_values(["concept", "alpha"])
    frame.to_csv(args.output_dir / "per_image.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
