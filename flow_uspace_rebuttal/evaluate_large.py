from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer

import lpips


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_image(path: Path) -> torch.Tensor:
    return TF.pil_to_tensor(Image.open(path).convert("RGB")).float() / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="flow_uspace_rebuttal/config_large.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--manifest-names", nargs="+")
    parser.add_argument("--output-prefix", default="metrics")
    args = parser.parse_args()
    with resolve(args.config).open() as file:
        cfg = yaml.safe_load(file)
    result_dir = resolve(cfg["output_dir"])

    manifests = (
        [result_dir / name for name in args.manifest_names]
        if args.manifest_names
        else sorted(result_dir.glob("manifest*.csv"))
    )
    if not manifests:
        raise FileNotFoundError(f"No manifests under {result_dir}")
    table = pd.concat([pd.read_csv(path) for path in manifests], ignore_index=True)
    table = table.drop_duplicates(["image_id", "concept", "method", "topk", "strength", "path"])

    device = torch.device("cuda")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    for model in (clip, perceptual):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    concepts = sorted(table.concept.unique())
    prompt_pairs = [cfg["alignment"]["concepts"][concept] for concept in concepts]
    prompts = [prompt for pair in prompt_pairs for prompt in (pair["source"], pair["target"])]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        text = F.normalize(
            clip.get_text_features(**{key: value.to(device) for key, value in tokens.items()}),
            dim=-1,
        ).reshape(len(concepts), 2, -1)
    text_by_concept = {concept: text[index] for index, concept in enumerate(concepts)}
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None]

    @torch.no_grad()
    def clip_features(images):
        images = F.interpolate(images, (224, 224), mode="bicubic", align_corners=False)
        return F.normalize(clip.get_image_features(pixel_values=(images - mean) / std), dim=-1)

    metric_rows = []
    records = table.to_dict("records")
    for begin in range(0, len(records), args.batch_size):
        chunk = records[begin : begin + args.batch_size]
        references = torch.stack(
            [load_image(result_dir / "inversion" / row["image_id"]) for row in chunk]
        ).to(device)
        edits = torch.stack([load_image(resolve(row["path"])) for row in chunk]).to(device)
        with torch.no_grad():
            ref_features = clip_features(references)
            edit_features = clip_features(edits)
            image_directions = F.normalize(edit_features - ref_features, dim=-1)
            lpips_values = perceptual(references * 2 - 1, edits * 2 - 1).flatten()
            for index, row in enumerate(chunk):
                pair = text_by_concept[row["concept"]]
                text_direction = F.normalize(pair[1] - pair[0], dim=-1)
                metric_rows.append(
                    {
                        **row,
                        "directional_clip": float(image_directions[index] @ text_direction),
                        "target_clip_change": float(
                            edit_features[index] @ pair[1] - ref_features[index] @ pair[1]
                        ),
                        "lpips": float(lpips_values[index]),
                    }
                )
        print(f"\rmetrics {min(begin + args.batch_size, len(records))}/{len(records)}", end="")
    print()

    per_image = pd.DataFrame(metric_rows)
    per_image.to_csv(result_dir / f"{args.output_prefix}_per_image.csv", index=False)
    grouping = ["concept", "method", "topk", "strength"]
    metrics = ["directional_clip", "target_clip_change", "lpips"]
    summary = per_image.groupby(grouping, dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(result_dir / f"{args.output_prefix}_summary.csv", index=False)

    # A compact table for selecting matched-strength pairs. Matching itself is
    # performed per concept after inspecting the complete, non-cherry-picked sweep.
    compact = (
        per_image.groupby(grouping, dropna=False)[metrics]
        .mean()
        .reset_index()
        .sort_values(["concept", "method", "topk", "strength"])
    )
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
