from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import clip
import cv2
import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm

from .utils import append_event, load_config, test_ids, write_json


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
LABEL_PATH = REPO_DIR.parent / "CelebA-HQ" / "list_attr_celeba_shift.txt"
CLASSIFIERS = {
    "resnet18": REPO_DIR
    / "celeba_classifier"
    / "resnet_model"
    / "resnet18_multiplicative_pretrain_",
    "mobilenetv2": REPO_DIR
    / "celeba_classifier"
    / "MobileNetV2_model"
    / "MobileNetV2_multiplicative_pretrain_",
    "vgg16": REPO_DIR
    / "celeba_classifier"
    / "vgg16_model"
    / "vgg16_multiplicative_pretrain_",
    "vit_b_16": REPO_DIR
    / "celeba_classifier"
    / "vit_model"
    / "vit_multiplicative_pretrain_",
}


def attribute_names() -> list[str]:
    with LABEL_PATH.open("r", encoding="utf-8") as handle:
        handle.readline()
        return handle.readline().split()


def make_classifier(name: str, device: torch.device) -> nn.Module:
    if name == "resnet18":
        backbone = tv_models.resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, 40)

        class Wrapped(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.resnet = model

            def forward(self, x):
                return self.resnet(x)

        model = Wrapped(backbone)
    elif name == "mobilenetv2":
        backbone = tv_models.mobilenet_v2(weights=None)
        backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, 40)

        class Wrapped(nn.Module):
            def __init__(self, network):
                super().__init__()
                self.mobilenet = network

            def forward(self, x):
                return self.mobilenet(x)

        model = Wrapped(backbone)
    elif name == "vgg16":
        backbone = tv_models.vgg16(weights=None)
        backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, 40)

        class Wrapped(nn.Module):
            def __init__(self, network):
                super().__init__()
                self.vgg = network

            def forward(self, x):
                return self.vgg(x)

        model = Wrapped(backbone)
    elif name == "vit_b_16":
        backbone = tv_models.vit_b_16(weights=None)
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, 40)

        class Wrapped(nn.Module):
            def __init__(self, network):
                super().__init__()
                self.vit = network

            def forward(self, x):
                return self.vit(x)

        model = Wrapped(backbone)
    else:
        raise ValueError(name)
    state = torch.load(CLASSIFIERS[name], map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def classifier_tensor(path: str | Path, classifier: str) -> torch.Tensor:
    # Match the classifier training datasets in celeba_classifier/celeba.py.
    # PIL takes (width, height): CNNs used HxW=274x224, ViT used 224x224.
    size = (224, 224) if classifier == "vit_b_16" else (224, 274)
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


@torch.no_grad()
def classify_paths(
    paths: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {path: {} for path in paths}
    for name in CLASSIFIERS:
        model = make_classifier(name, device)
        for start in tqdm(range(0, len(paths), batch_size), desc=f"classify/{name}"):
            batch_paths = paths[start : start + batch_size]
            batch = torch.stack(
                [classifier_tensor(path, name) for path in batch_paths]
            ).to(device)
            logits = model(batch).float()
            probs = logits.sigmoid()
            for index, path in enumerate(batch_paths):
                output[path][f"{name}_logits"] = logits[index].cpu().numpy()
                output[path][f"{name}_probs"] = probs[index].cpu().numpy()
        del model
        torch.cuda.empty_cache()
    return output


def lpips_tensor(path: str | Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


class IdentityEvaluator:
    def __init__(self, device: torch.device):
        device_id = int(device.index or 0) if device.type == "cuda" else 0
        providers = (
            [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
            if device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(Path.home() / ".insightface"),
            providers=providers,
        )
        self.app.prepare(
            ctx_id=device_id if device.type == "cuda" else -1,
            det_size=(640, 640),
            det_thresh=0.3,
        )
        self.cache: dict[str, np.ndarray | None] = {}

    def embedding(self, path: str) -> np.ndarray | None:
        if path not in self.cache:
            image = cv2.imread(path)
            faces = self.app.get(image) if image is not None else []
            if not faces:
                self.cache[path] = None
            else:
                face = max(faces, key=lambda item: float(item.bbox[2] - item.bbox[0]) * float(item.bbox[3] - item.bbox[1]))
                value = np.asarray(face.embedding, dtype=np.float32)
                self.cache[path] = value / max(np.linalg.norm(value), 1e-8)
        return self.cache[path]

    def similarity(self, first: str, second: str) -> float:
        one = self.embedding(first)
        two = self.embedding(second)
        if one is None or two is None:
            return float("nan")
        return float(np.dot(one, two))


class AugClipEvaluator:
    def __init__(self, device: torch.device, cfg: dict):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        self.features: dict[str, torch.Tensor] = {}
        source = cfg["neutral_prompt"]
        tokens = [source] + [cfg["concepts"][name]["target_prompt"] for name in cfg["concepts"]]
        with torch.no_grad():
            text = F.normalize(self.model.encode_text(clip.tokenize(tokens).to(device)).float(), dim=-1)
        self.directions = {
            name: F.normalize(text[index + 1] - text[0], dim=-1)
            for index, name in enumerate(cfg["concepts"])
        }

    @torch.no_grad()
    def feature(self, path: str) -> torch.Tensor:
        if path not in self.features:
            image = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
            self.features[path] = F.normalize(self.model.encode_image(image).float(), dim=-1).cpu()
        return self.features[path]

    def directional_score(self, reference: str, edited: str, concept: str) -> float:
        direction = self.feature(edited) - self.feature(reference)
        direction = F.normalize(direction, dim=-1)
        return float((direction.to(self.device) * self.directions[concept]).sum())


def unique_paths(cfg: dict, edit_records: list[dict]) -> list[str]:
    values = set()
    for image_id in test_ids(cfg):
        values.add(str(Path(cfg["data_dir"]) / image_id))
        values.add(str(Path(cfg["result_dir"]) / "reconstruction" / image_id))
    values.update(str(record["path"]) for record in edit_records)
    missing = [path for path in values if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} evaluation images; first: {missing[0]}")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--result-dir",
        help="Optional result directory override (useful for isolated ablations)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--skip-arcface", action="store_true")
    parser.add_argument("--skip-augclip", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.result_dir:
        cfg["result_dir"] = args.result_dir
    if args.test_count is not None:
        cfg["test_count"] = args.test_count
    result_dir = Path(cfg["result_dir"])
    with (result_dir / "edit_manifest.json").open("r", encoding="utf-8") as handle:
        edit_records = json.load(handle)
    # A resumed manifest can contain repeats; one row per exact output path is enough.
    edit_records = list({str(row["path"]): row for row in edit_records}.values())
    paths = unique_paths(cfg, edit_records)
    device = torch.device(args.device)
    started = time.time()
    predictions = classify_paths(paths, device, args.batch_size)
    names = attribute_names()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    identity = None if args.skip_arcface else IdentityEvaluator(device)
    augclip = None if args.skip_augclip else AugClipEvaluator(device, cfg)

    inversion_rows = []
    for image_id in tqdm(test_ids(cfg), desc="inversion metrics"):
        source = str(Path(cfg["data_dir"]) / image_id)
        reconstruction = str(result_dir / "reconstruction" / image_id)
        with torch.no_grad():
            reconstruction_lpips = float(
                lpips_model(lpips_tensor(source, device), lpips_tensor(reconstruction, device)).item()
            )
        inversion_rows.append(
            {
                "image_id": image_id,
                "source_path": source,
                "reconstruction_path": reconstruction,
                "inversion_lpips": reconstruction_lpips,
                "inversion_arcface": (
                    identity.similarity(source, reconstruction) if identity is not None else float("nan")
                ),
            }
        )
    pd.DataFrame(inversion_rows).to_csv(result_dir / "inversion_metrics.csv", index=False)

    rows = []
    for record in tqdm(edit_records, desc="edit metrics"):
        image_id = record["image_id"]
        reference = str(result_dir / "reconstruction" / image_id)
        edited = str(record["path"])
        concept = record["concept"]
        concept_cfg = cfg["concepts"][concept]
        target_attribute = concept_cfg.get("attribute", concept)
        target_sign = float(concept_cfg.get("target_sign", 1.0))
        target_idx = names.index(target_attribute)
        non_target = [index for index in range(len(names)) if index != target_idx]
        row = dict(record)
        for classifier in CLASSIFIERS:
            before_logits = predictions[reference][f"{classifier}_logits"]
            after_logits = predictions[edited][f"{classifier}_logits"]
            before_probs = predictions[reference][f"{classifier}_probs"]
            after_probs = predictions[edited][f"{classifier}_probs"]
            probability_change = after_probs - before_probs
            logit_change = after_logits - before_logits
            signed_target_probability_change = (
                target_sign * probability_change[target_idx]
            )
            signed_target_logit_change = target_sign * logit_change[target_idx]
            before_target = (
                before_probs[target_idx]
                if target_sign > 0
                else 1.0 - before_probs[target_idx]
            )
            after_target = (
                after_probs[target_idx]
                if target_sign > 0
                else 1.0 - after_probs[target_idx]
            )
            prefix = classifier
            row.update(
                {
                    f"{prefix}_target_before": float(before_target),
                    f"{prefix}_target_after": float(after_target),
                    f"{prefix}_target_change": float(signed_target_probability_change),
                    f"{prefix}_target_logit_change": float(signed_target_logit_change),
                    f"{prefix}_mean_offtarget_change": float(
                        np.abs(probability_change[non_target]).mean()
                    ),
                    f"{prefix}_max_offtarget_change": float(
                        np.abs(probability_change[non_target]).max()
                    ),
                    f"{prefix}_mean_offtarget_logit_change": float(
                        np.abs(logit_change[non_target]).mean()
                    ),
                    f"{prefix}_target_success": int(after_target >= 0.5),
                    f"{prefix}_eligible_negative": int(before_target < 0.5),
                    f"{prefix}_negative_to_positive": int(
                        before_target < 0.5 and after_target >= 0.5
                    ),
                }
            )
        with torch.no_grad():
            row["lpips"] = float(
                lpips_model(lpips_tensor(reference, device), lpips_tensor(edited, device)).item()
            )
        row["arcface"] = (
            identity.similarity(reference, edited) if identity is not None else float("nan")
        )
        row["augclip_directional"] = (
            augclip.directional_score(reference, edited, concept)
            if augclip is not None and float(record["strength"]) != 0
            else float("nan")
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(result_dir / "per_image_metrics.csv", index=False)
    metadata = {
        "reference_for_edit_metrics": "SDXL inversion reconstruction",
        "reference_for_inversion_metrics": "original CelebA-HQ image",
        "attribute_names": names,
        "attribute_classifiers": {name: str(path) for name, path in CLASSIFIERS.items()},
        "arcface": "InsightFace buffalo_l, SCRFD det_thresh=0.3; NaN means no detected face",
        "lpips": "official lpips 0.1.4 AlexNet",
        "augclip": (
            "not computed (--skip-augclip)"
            if augclip is None
            else "OpenAI CLIP ViT-B/32 cosine of image and text edit directions"
        ),
        "epr_definition_for_analysis": (
            "paper Eq. 12-14: aggregate mean absolute target classifier-logit change divided "
            "by aggregate mean absolute non-target classifier-logit change, epsilon=1e-8; "
            "per-image ratios are not averaged"
        ),
        "rows": len(frame),
        "seconds": time.time() - started,
    }
    write_json(metadata, result_dir / "evaluation_metadata.json")
    append_event(cfg, "evaluation_complete", **metadata)
    print(f"[saved] {result_dir / 'per_image_metrics.csv'} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
