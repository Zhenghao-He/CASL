#!/usr/bin/env python3
"""Four-classifier target, non-target, and EPR evaluation for the ablation."""

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdxl_rebuttal.evaluate import attribute_names, classify_paths


CLASSIFIERS = ("resnet18", "mobilenetv2", "vgg16", "vit_b_16")
CONCEPTS = {
    "smiling": ("Smiling", 1.0, 128.0),
    "young": ("Young", 1.0, 128.0),
    "big_nose": ("Big_Nose", 1.0, 64.0),
    "beards": ("No_Beard", -1.0, 96.0),
    "blond_hair": ("Blond_Hair", 1.0, 128.0),
}
CASL_STRENGTHS = (0, 32, 64, 80, 96, 112, 128, 144, 160, 176, 192)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sae-prefix",
        default="sae_only",
        help="Run namespace for SAE-only outputs, e.g. sae_only_v2.",
    )
    parser.add_argument(
        "--exclude-dense",
        action="store_true",
        help="Exclude the archived Dense-CLIP diagnostic.",
    )
    parser.add_argument(
        "--include-official-asyrp",
        action="store_true",
        help="Evaluate the fixed official ASYRP iter4 outputs alongside the ablation.",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("novelty_ablation/results/selected_strengths.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("novelty_ablation/results"),
    )
    return parser.parse_args()


def image_ids(split):
    values = [f"{value:05d}" for value in range(29501, 29533)]
    return values[:8] if split == "validation" else values[8:]


def strength_from_name(name):
    match = re.search(r"scale(-?[0-9]+(?:\.[0-9]+)?)$", name)
    if match is None:
        raise ValueError(name)
    return float(match.group(1))


def highest_iteration_directory(root, strength):
    candidates = []
    for path in root.glob("test_*_scale*"):
        if not np.isclose(strength_from_name(path.name), float(strength)):
            continue
        match = re.match(r"test_(\d+)_scale", path.name)
        candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates)[1]


def sae_directory(concept, strength, prefix="sae_only"):
    root = (
        ROOT
        / "runs"
        / f"{prefix}_{concept}_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
        / "delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse"
        / "test_xblock_editted_delta_images_topk1"
        / "100"
    )
    return highest_iteration_directory(root, strength)


def casl_directory(concept, strength):
    root = (
        ROOT
        / "runs"
        / f"{concept}_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
        / "delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse"
        / "test_xblock_editted_delta_images_topk1"
        / "100"
    )
    return highest_iteration_directory(root, strength)


def dense_tag(strength):
    tag = f"{strength:g}".replace("-", "_neg_").replace(".", "p")
    return tag


def dense_directory(concept, strength):
    extracted = (
        ROOT
        / "novelty_ablation"
        / "artifacts"
        / "dense"
        / concept
        / f"strength_{strength:g}"
    )
    if extracted.is_dir():
        return extracted
    return (
        ROOT
        / "runs"
        / (
            f"novelty_dense_sweep_{concept}_{dense_tag(strength)}"
            "_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
        )
        / "sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse"
        / "xblock_test_images"
        / "100"
    )


def asyrp_directory(concept):
    return (
        ROOT.parent
        / "Asyrp_official"
        / "runs"
        / f"{concept}_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
        / "test_images"
        / "100"
        / "iter4"
        / "delta_h"
        / "images"
    )


def id_path_map(directory, method):
    if directory is None or not directory.is_dir():
        return {}
    if method in {"casl", "sae_only"}:
        candidates = list((directory / "delta_h" / "images").glob("*.jpg"))
        return {path.stem: path for path in candidates}
    if method == "asyrp":
        candidates = list(directory.glob("*.jpg"))
        return {path.stem: path for path in candidates}
    extracted = list(directory.glob("*.png"))
    if extracted:
        return {path.stem: path for path in extracted}
    candidates = list(directory.glob("test_*_ngen50.png"))
    result = {}
    for path in candidates:
        match = re.match(r"test_(\d{5})_", path.name)
        if match:
            result[match.group(1)] = path
    return result


def available_sae_strengths(concept, prefix="sae_only"):
    root = (
        ROOT
        / "runs"
        / f"{prefix}_{concept}_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
        / "delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse"
        / "test_xblock_editted_delta_images_topk1"
        / "100"
    )
    return sorted({strength_from_name(path.name) for path in root.glob("test_*_scale*")})


def available_dense_strengths(concept):
    extracted_root = (
        ROOT / "novelty_ablation" / "artifacts" / "dense" / concept
    )
    extracted = []
    for path in extracted_root.glob("strength_*"):
        extracted.append(float(path.name.removeprefix("strength_")))
    if extracted:
        return sorted(set(extracted))
    prefix = f"novelty_dense_sweep_{concept}_"
    suffix = "_LC_CelebA_HQ_P2_t999_ninv50_ngen50"
    strengths = []
    for path in (ROOT / "runs").glob(f"{prefix}*{suffix}"):
        tag = path.name[len(prefix) : -len(suffix)]
        value = tag.replace("_neg_", "-").replace("p", ".")
        strengths.append(float(value))
    return sorted(set(strengths))


def build_records(
    split,
    selection,
    include_official_asyrp=False,
    sae_prefix="sae_only",
    exclude_dense=False,
):
    ids = image_ids(split)
    records = []
    # A single canonical zero-intervention reconstruction is shared by every
    # concept and method. This avoids concept-specific reference drift.
    reference = id_path_map(sae_directory("smiling", 0.0), "sae_only")
    for concept, (_, _, table_strength) in CONCEPTS.items():
        if split == "validation":
            method_strengths = {
                "casl": list(CASL_STRENGTHS),
                "sae_only": available_sae_strengths(concept, sae_prefix),
            }
            if not exclude_dense:
                method_strengths["dense_clip"] = available_dense_strengths(concept)
        else:
            chosen = selection[concept]
            method_strengths = {
                method: [float(chosen[method])] for method in chosen
            }
            if include_official_asyrp:
                # ASYRP's official iter4 output is a single, already tuned
                # operating point; strength is a label rather than CASL alpha.
                method_strengths["asyrp"] = [1.0]
        for method, strengths in method_strengths.items():
            for strength in strengths:
                if method == "casl":
                    directory = casl_directory(concept, strength)
                elif method == "sae_only":
                    directory = sae_directory(concept, strength, sae_prefix)
                elif method == "asyrp":
                    directory = asyrp_directory(concept)
                else:
                    directory = dense_directory(concept, strength)
                if method == "sae_only" and np.isclose(strength, 0.0):
                    # Zero intervention is the canonical inversion itself,
                    # not a separately sampled rerun with fresh eta noise.
                    edited = reference
                else:
                    edited = id_path_map(directory, method)
                for image_id in ids:
                    if image_id not in reference or image_id not in edited:
                        continue
                    records.append(
                        {
                            "concept": concept,
                            "method": method,
                            "strength": float(strength),
                            "image_id": image_id,
                            "reference": str(reference[image_id]),
                            "edited": str(edited[image_id]),
                            "table_strength": table_strength,
                        }
                    )
    return records


def score_records(records, device, batch_size):
    paths = sorted(
        {row["reference"] for row in records} | {row["edited"] for row in records}
    )
    predictions = classify_paths(paths, device, batch_size)
    names = attribute_names()
    rows = []
    for record in records:
        attribute, sign, _ = CONCEPTS[record["concept"]]
        target = names.index(attribute)
        non_target = [index for index in range(len(names)) if index != target]
        row = dict(record)
        for classifier in CLASSIFIERS:
            before = predictions[record["reference"]]
            after = predictions[record["edited"]]
            prob_delta = (
                after[f"{classifier}_probs"] - before[f"{classifier}_probs"]
            )
            logit_delta = (
                after[f"{classifier}_logits"] - before[f"{classifier}_logits"]
            )
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
    return pd.DataFrame(rows)


def aggregate(frame):
    rows = []
    for key, group in frame.groupby(["concept", "method", "strength"]):
        concept, method, strength = key
        row = {
            "concept": concept,
            "method": method,
            "strength": strength,
            "n": len(group),
            "target_delta": group["target_delta"].mean(),
            "nontarget_delta": group["nontarget_delta"].mean(),
        }
        eprs = []
        for classifier in CLASSIFIERS:
            numerator = group[f"{classifier}_target_logit_abs"].mean()
            denominator = group[f"{classifier}_nontarget_logit_abs"].mean()
            value = float(numerator / max(float(denominator), 1e-8))
            row[f"{classifier}_epr"] = value
            eprs.append(value)
        row["epr"] = float(np.mean(eprs))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_strengths(summary):
    selection = {}
    for concept, (_, _, casl_strength) in CONCEPTS.items():
        subset = summary[summary["concept"] == concept]
        casl = subset[
            (subset["method"] == "casl")
            & np.isclose(subset["strength"], casl_strength)
        ]
        if casl.empty:
            raise ValueError(f"Missing fixed CASL strength for {concept}")
        target = float(casl.iloc[0]["target_delta"])
        selected = {"casl": casl_strength}
        for method in ("sae_only", "dense_clip"):
            candidates = subset[subset["method"] == method].copy()
            if candidates.empty:
                continue
            if method == "sae_only":
                # This is deliberately favorable to the heuristic baseline:
                # choose its strongest positive target edit using the same
                # eight validation images, breaking ties by preservation.
                candidates = candidates.sort_values(
                    ["target_delta", "nontarget_delta"],
                    ascending=[False, True],
                )
                selected[method] = float(candidates.iloc[0]["strength"])
            else:
                candidates["distance"] = np.abs(candidates["target_delta"] - target)
                selected[method] = float(
                    candidates.sort_values("distance").iloc[0]["strength"]
                )
        selection[concept] = selected
    return selection


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = {}
    if args.split == "test":
        with args.selection.open() as handle:
            selection = json.load(handle)
    records = build_records(
        args.split,
        selection,
        include_official_asyrp=args.include_official_asyrp,
        sae_prefix=args.sae_prefix,
        exclude_dense=args.exclude_dense,
    )
    if not records:
        raise RuntimeError("No complete edit records found")
    frame = score_records(records, torch.device(args.device), args.batch_size)
    summary = aggregate(frame)
    suffix = "_with_asyrp" if args.include_official_asyrp else ""
    frame.to_csv(
        args.output_dir / f"{args.split}{suffix}_per_image.csv", index=False
    )
    summary.to_csv(
        args.output_dir / f"{args.split}{suffix}_summary.csv", index=False
    )
    if args.split == "validation":
        selection = choose_strengths(summary)
        with args.selection.open("w") as handle:
            json.dump(selection, handle, indent=2)
        print(json.dumps(selection, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
