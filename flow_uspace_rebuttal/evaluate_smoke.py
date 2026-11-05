from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from insightface.app import FaceAnalysis
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]


def image_tensor(path: Path, device: torch.device):
    return TF.pil_to_tensor(Image.open(path).convert("RGB")).float().to(device) / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/flow_uspace_smoke")
    args = parser.parse_args()
    result_dir = ROOT / args.result_dir
    manifest = pd.read_csv(result_dir / "manifest.csv")
    device = torch.device("cuda")

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    for parameter in clip.parameters():
        parameter.requires_grad_(False)
    tokens = tokenizer(
        [
            "a portrait of a person with a neutral expression",
            "a portrait of a smiling person",
        ],
        padding=True,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        text = F.normalize(clip.get_text_features(**tokens), dim=-1)
        text_direction = F.normalize(text[1] - text[0], dim=-1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None]

    def clip_features(image):
        x = F.interpolate(image, (224, 224), mode="bicubic", align_corners=False)
        return F.normalize(clip.get_image_features(pixel_values=(x - mean) / std), dim=-1)

    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
    embedding_cache = {}

    def embedding(path: Path):
        key = str(path)
        if key not in embedding_cache:
            faces = face.get(cv2.imread(key))
            embedding_cache[key] = None if not faces else faces[0].normed_embedding
        return embedding_cache[key]

    rows = []
    for record in manifest.to_dict("records"):
        image_id = record["image_id"]
        reference_path = result_dir / "inversion" / image_id
        edited_path = ROOT / record["path"]
        reference = image_tensor(reference_path, device)[None]
        edited = image_tensor(edited_path, device)[None]
        with torch.no_grad():
            ref_feat = clip_features(reference)
            edit_feat = clip_features(edited)
            image_direction = F.normalize(edit_feat - ref_feat, dim=-1)
            directional_clip = (image_direction * text_direction[None]).sum().item()
            target_clip = (edit_feat * text[1:2]).sum().item()
            lpips_value = perceptual(reference * 2 - 1, edited * 2 - 1).item()
        ref_emb, edit_emb = embedding(reference_path), embedding(edited_path)
        identity = (
            float(np.dot(ref_emb, edit_emb))
            if ref_emb is not None and edit_emb is not None
            else float("nan")
        )
        rows.append(
            {
                **record,
                "directional_clip": directional_clip,
                "target_clip": target_clip,
                "lpips": lpips_value,
                "arcface": identity,
            }
        )
    per_image = pd.DataFrame(rows)
    per_image.to_csv(result_dir / "metrics_per_image.csv", index=False)
    summary = (
        per_image.groupby(["method", "topk", "strength"], dropna=False)[
            ["directional_clip", "target_clip", "lpips", "arcface"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(result_dir / "metrics_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
