"""Minimal classifier/EPR/preservation evaluation for the flow rebuttal.

Edit-induced metrics use the Euler reconstruction as reference.  The original
image is used only for inversion LPIPS and ArcFace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdxl_rebuttal.evaluate import (
    CLASSIFIERS,
    IdentityEvaluator,
    attribute_names,
    classify_paths,
)


CONCEPTS = {
    "smiling": ("Smiling", 1.0),
    "young": ("Young", 1.0),
    "big_nose": ("Big_Nose", 1.0),
    "beard": ("No_Beard", -1.0),
    "blond_hair": ("Blond_Hair", 1.0),
    "eyeglasses": ("Eyeglasses", 1.0),
    "male": ("Male", 1.0),
    "mustache": ("Mustache", 1.0),
    "no_beard": ("No_Beard", 1.0),
    "bangs": ("Bangs", 1.0),
    "heavy_makeup": ("Heavy_Makeup", 1.0),
    "chubby": ("Chubby", 1.0),
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def lpips_tensor(path: str | Path, device: torch.device, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="flow_uspace_rebuttal/config_large.yaml")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--concepts", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--with-preservation", action="store_true")
    args = parser.parse_args()

    with resolve(args.config).open() as handle:
        cfg = yaml.safe_load(handle)
    result_dir = resolve(cfg["output_dir"])
    tables = [pd.read_csv(result_dir / name) for name in args.manifests]
    table = pd.concat(tables, ignore_index=True).drop_duplicates(
        ["image_id", "concept", "method", "topk", "strength", "path"]
    )
    table = table[table["method"].isin(["dense", "casl"])].copy()
    if args.concepts:
        table = table[table.concept.isin(args.concepts)].copy()
    unknown = sorted(set(table.concept) - set(CONCEPTS))
    if unknown:
        raise ValueError(f"Missing concept definitions: {unknown}")

    references = {
        image_id: str(result_dir / "inversion" / image_id)
        for image_id in table.image_id.unique()
    }
    edit_paths = [str(resolve(path)) for path in table.path]
    paths = sorted(set(references.values()) | set(edit_paths))
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    device = torch.device(args.device)
    resolution = int(cfg["resolution"])
    predictions = classify_paths(paths, device, args.batch_size)
    names = attribute_names()
    lpips_model = (
        lpips.LPIPS(net="alex").to(device).eval() if args.with_preservation else None
    )
    identity = IdentityEvaluator(device) if args.with_preservation else None
    if args.with_preservation:
        inversion_rows = []
        for image_id in sorted(table.image_id.unique()):
            source = str(resolve(cfg["source_dir"]) / image_id)
            reconstruction = references[image_id]
            with torch.no_grad():
                inversion_lpips = float(
                    lpips_model(
                        lpips_tensor(source, device, resolution),
                        lpips_tensor(reconstruction, device, resolution),
                    ).item()
                )
            inversion_rows.append(
                {
                    "image_id": image_id,
                    "inversion_lpips": inversion_lpips,
                    "inversion_arcface": identity.similarity(source, reconstruction),
                }
            )
        pd.DataFrame(inversion_rows).to_csv(
            result_dir / f"{args.output_prefix}_inversion.csv", index=False
        )

    rows = []
    for record in table.to_dict("records"):
        reference = references[record["image_id"]]
        edited = str(resolve(record["path"]))
        attribute, sign = CONCEPTS[record["concept"]]
        target = names.index(attribute)
        non_target = [index for index in range(len(names)) if index != target]
        row = dict(record)
        for classifier in CLASSIFIERS:
            before_logits = predictions[reference][f"{classifier}_logits"]
            after_logits = predictions[edited][f"{classifier}_logits"]
            before_probs = predictions[reference][f"{classifier}_probs"]
            after_probs = predictions[edited][f"{classifier}_probs"]
            probability_change = after_probs - before_probs
            logit_change = after_logits - before_logits
            before_target = before_probs[target] if sign > 0 else 1 - before_probs[target]
            after_target = after_probs[target] if sign > 0 else 1 - after_probs[target]
            row.update(
                {
                    f"{classifier}_target_change": float(sign * probability_change[target]),
                    f"{classifier}_target_logit_change": float(sign * logit_change[target]),
                    f"{classifier}_mean_offtarget_logit_change": float(
                        np.abs(logit_change[non_target]).mean()
                    ),
                    f"{classifier}_eligible_negative": int(before_target < 0.5),
                    f"{classifier}_negative_to_positive": int(
                        before_target < 0.5 and after_target >= 0.5
                    ),
                }
            )
        if args.with_preservation:
            with torch.no_grad():
                row["lpips"] = float(
                    lpips_model(
                        lpips_tensor(reference, device, resolution),
                        lpips_tensor(edited, device, resolution),
                    ).item()
                )
            row["arcface"] = identity.similarity(reference, edited)
        rows.append(row)

    per_image = pd.DataFrame(rows)
    per_image.to_csv(result_dir / f"{args.output_prefix}_per_image.csv", index=False)
    grouping = ["concept", "method", "topk", "strength"]
    summary_rows = []
    for keys, group in per_image.groupby(grouping, dropna=False):
        row = dict(zip(grouping, keys))
        target_changes, success_rates, eprs = [], [], []
        for classifier in CLASSIFIERS:
            target_changes.append(float(group[f"{classifier}_target_change"].mean()))
            eligible = group[group[f"{classifier}_eligible_negative"] == 1]
            success_rates.append(
                float(eligible[f"{classifier}_negative_to_positive"].mean())
                if len(eligible)
                else float("nan")
            )
            numerator = float(group[f"{classifier}_target_logit_change"].abs().mean())
            denominator = float(
                group[f"{classifier}_mean_offtarget_logit_change"].mean()
            )
            eprs.append(numerator / max(denominator, 1e-8))
            row[f"{classifier}_target_change"] = target_changes[-1]
            row[f"{classifier}_success_rate"] = success_rates[-1]
            row[f"{classifier}_epr"] = eprs[-1]
        row["target_change_4clf"] = float(np.mean(target_changes))
        row["positive_classifier_fraction"] = float(
            np.mean(np.asarray(target_changes) > 0)
        )
        row["success_rate_4clf"] = float(np.nanmean(success_rates))
        row["epr_4clf"] = float(np.mean(eprs))
        row["n"] = int(group.image_id.nunique())
        if args.with_preservation:
            row["lpips"] = float(group.lpips.mean())
            row["arcface"] = float(group.arcface.mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(grouping)
    summary.to_csv(result_dir / f"{args.output_prefix}_summary.csv", index=False)

    metadata = {
        "edit_reference": "Euler inversion reconstruction",
        "inversion_reference": "original image",
        "epr": "mean across ResNet18, MobileNetV2, VGG16, ViT-B/16; each backbone uses aggregate |target logit delta| / mean aggregate |off-target logit delta|",
        "target_change": "mean signed target probability change across four classifiers",
        "success": "negative-to-positive rate, averaged across four classifiers",
        "manifests": args.manifests,
        "rows": len(per_image),
    }
    (result_dir / f"{args.output_prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
