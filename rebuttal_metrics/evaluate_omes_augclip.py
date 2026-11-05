#!/usr/bin/env python3
"""Evaluate image-level OMES (OS/MES) and official-formula AugCLIP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import SGDClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ID_PATTERN = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inactive-threshold", type=float, default=0.05)
    parser.add_argument("--omes-pooling", choices=("avg", "max"), default="avg")
    parser.add_argument(
        "--concepts",
        nargs="+",
        help="Optional concept subset, preserving the config's original order.",
    )
    parser.add_argument("--skip-augclip", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def image_id(path: Path) -> str:
    matches = ID_PATTERN.findall(path.stem)
    if not matches:
        raise ValueError(f"No numeric image ID in {path}")
    return matches[0].lstrip("0") or "0"


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = image_id(path)
            if key in indexed:
                raise ValueError(f"Duplicate image ID {key} in {directory}")
            indexed[key] = path
    return indexed


def load_celeba_descriptions(attribute_file: Path) -> dict[str, list[str]]:
    with attribute_file.open() as handle:
        count = int(handle.readline().strip())
        attributes = handle.readline().strip().split()
        descriptions: dict[str, list[str]] = {}
        for line in handle:
            fields = line.strip().split()
            if not fields:
                continue
            key = image_id(Path(fields[0]))
            values = [int(value) for value in fields[1:]]
            positive = [
                f"The person has {name.replace('_', ' ').lower()}"
                for name, value in zip(attributes, values)
                if value > 0
            ]
            descriptions[key] = positive
    if len(descriptions) != count:
        raise ValueError(
            f"Expected {count} CelebA rows in {attribute_file}, got {len(descriptions)}"
        )
    return descriptions


def load_source_descriptions(
    spec: dict[str, Any], repo_root: Path
) -> dict[str, list[str]]:
    kind = spec["type"]
    if kind == "celeba_attributes":
        return load_celeba_descriptions(
            resolve_path(repo_root, spec["attribute_file"])
        )
    if kind == "json":
        with resolve_path(repo_root, spec["path"]).open() as handle:
            raw = json.load(handle)
        return {str(key).lstrip("0") or "0": value for key, value in raw.items()}
    if kind == "constant":
        return {"_default": spec["descriptions"]}
    raise ValueError(f"Unsupported source description type: {kind}")


def build_pairs(config: dict[str, Any], repo_root: Path) -> tuple[
    dict[str, dict[str, list[tuple[str, Path, Path]]]], dict[str, Any]
]:
    source_dir = resolve_path(repo_root, config["source_dir"])
    source_images = index_images(source_dir)
    pairs: dict[str, dict[str, list[tuple[str, Path, Path]]]] = defaultdict(dict)
    audit: dict[str, Any] = {
        "dataset": config["dataset"],
        "source_dir": str(source_dir),
        "source_count": len(source_images),
        "methods": {},
    }
    for method, concept_dirs in config["methods"].items():
        audit["methods"][method] = {}
        for concept in config["concepts"]:
            directory_spec = concept_dirs.get(concept)
            if directory_spec is None:
                audit["methods"][method][concept] = {
                    "status": "missing_config",
                    "pairs": 0,
                }
                continue
            if isinstance(directory_spec, str):
                directory = resolve_path(repo_root, directory_spec)
                method_source_dir = source_dir
                method_source_images = source_images
            else:
                directory = resolve_path(repo_root, directory_spec["edited_dir"])
                method_source_dir = resolve_path(
                    repo_root, directory_spec.get("source_dir", config["source_dir"])
                )
                method_source_images = index_images(method_source_dir)
            edited = index_images(directory)
            common = sorted(set(method_source_images) & set(edited), key=int)
            pairs[method][concept] = [
                (key, method_source_images[key], edited[key]) for key in common
            ]
            audit["methods"][method][concept] = {
                "status": "ok" if common else "missing_or_empty",
                "directory": str(directory),
                "source_directory": str(method_source_dir),
                "edited_count": len(edited),
                "pairs": len(common),
                "missing_source_count": len(set(edited) - set(method_source_images)),
                "missing_edit_count": len(set(method_source_images) - set(edited)),
                "missing_source_ids_sample": sorted(
                    set(edited) - set(method_source_images), key=int
                )[:20],
                "missing_edit_ids_sample": sorted(
                    set(method_source_images) - set(edited), key=int
                )[:20],
            }
    return pairs, audit


class ClipEncoder:
    def __init__(self, model_name: str, device: str, batch_size: int):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.image_cache: dict[str, np.ndarray] = {}
        self.text_cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def encode_images(self, paths: list[Path]) -> np.ndarray:
        missing = [path for path in paths if str(path) not in self.image_cache]
        for start in range(0, len(missing), self.batch_size):
            batch_paths = missing[start : start + self.batch_size]
            batch = torch.stack(
                [
                    self.preprocess(Image.open(path).convert("RGB"))
                    for path in batch_paths
                ]
            ).to(self.device)
            features = self.model.encode_image(batch).float().cpu().numpy()
            for path, feature in zip(batch_paths, features):
                self.image_cache[str(path)] = feature
        return np.stack([self.image_cache[str(path)] for path in paths])

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        missing = [text for text in texts if text not in self.text_cache]
        for start in range(0, len(missing), self.batch_size):
            batch_texts = missing[start : start + self.batch_size]
            tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
            features = self.model.encode_text(tokens).float().cpu().numpy()
            for text, feature in zip(batch_texts, features):
                self.text_cache[text] = feature
        return np.stack([self.text_cache[text] for text in texts])


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def relation_matrix(
    concept_features: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[list[str], np.ndarray]:
    concepts = list(concept_features)
    columns = []
    for concept in concepts:
        source, edited = concept_features[concept]
        if source.shape != edited.shape:
            raise ValueError(f"Shape mismatch for {concept}: {source.shape}, {edited.shape}")
        source_centered = source - source.mean(axis=0, keepdims=True)
        edited_centered = edited - edited.mean(axis=0, keepdims=True)
        numerator = np.sum(source_centered * edited_centered, axis=0)
        denominator = np.sqrt(
            np.sum(source_centered**2, axis=0)
            * np.sum(edited_centered**2, axis=0)
        )
        correlation = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        columns.append(1.0 - np.abs(np.clip(correlation, -1.0, 1.0)))
    return concepts, np.stack(columns, axis=1)


def pool_weighted(values: np.ndarray, association: np.ndarray, pooling: str) -> float:
    weighted = values * association
    if pooling == "max":
        return float(np.max(weighted))
    denominator = float(np.sum(association))
    return float(np.sum(weighted * association) / denominator) if denominator else 0.0


def os_mes(
    relation: np.ndarray, factor_index: int, pooling: str
) -> tuple[float, float]:
    dimensions, factors = relation.shape
    ideal_factor = np.zeros(factors)
    ideal_factor[factor_index] = 1.0
    os_per_dimension = 1.0 - np.mean(
        np.abs(relation - ideal_factor[None, :]), axis=1
    )

    mes_per_dimension = np.empty(dimensions)
    factor_column = relation[:, factor_index]
    for dimension in range(dimensions):
        ideal_dimension = np.zeros(dimensions)
        ideal_dimension[dimension] = 1.0
        mes_per_dimension[dimension] = 1.0 - np.mean(
            np.abs(factor_column - ideal_dimension)
        )
    association = relation[:, factor_index]
    return (
        pool_weighted(os_per_dimension, association, pooling),
        pool_weighted(mes_per_dimension, association, pooling),
    )


def compute_omes(
    concept_features: dict[str, tuple[np.ndarray, np.ndarray]],
    representation_for_activity: np.ndarray,
    inactive_threshold: float,
    pooling: str,
) -> dict[str, Any]:
    concepts, relation = relation_matrix(concept_features)
    inactive = np.std(representation_for_activity, axis=0, ddof=1) < inactive_threshold
    active_relation = relation[~inactive]
    if not len(active_relation):
        raise ValueError("All representation dimensions were classified as inactive")
    factors: dict[str, dict[str, float]] = {}
    for index, concept in enumerate(concepts):
        os_score, mes_score = os_mes(active_relation, index, pooling)
        factors[concept] = {
            "OS": os_score,
            "MES": mes_score,
            "OMES": 0.5 * (os_score + mes_score),
        }
    return {
        "alpha": 0.5,
        "pooling": pooling,
        "inactive_threshold": inactive_threshold,
        "dimensions_total": int(relation.shape[0]),
        "dimensions_active": int((~inactive).sum()),
        "OS": float(np.mean([value["OS"] for value in factors.values()])),
        "MES": float(np.mean([value["MES"] for value in factors.values()])),
        "OMES": float(np.mean([value["OMES"] for value in factors.values()])),
        "factors": factors,
        "factor_order": concepts,
        "relation_matrix": active_relation.tolist(),
        "inactive_dimensions": np.flatnonzero(inactive).tolist(),
    }


def relative_weights(
    source_distribution: np.ndarray, target_distribution: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_distribution = normalize(source_distribution)
    target_distribution = normalize(target_distribution)
    source_source = source_distribution @ source_distribution.T
    target_target = target_distribution @ target_distribution.T
    source_target = source_distribution @ target_distribution.T
    source_weight = source_source.mean(axis=-1) - source_target.mean(axis=-1)
    target_weight = target_target.mean(axis=-1) - source_target.T.mean(axis=-1)

    def minmax(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min())
        return (values - values.min()) / span if span > 1e-12 else np.ones_like(values)

    return minmax(source_weight), minmax(target_weight)


def augclip_ideal(
    source_image: np.ndarray,
    source_descriptions: np.ndarray,
    source_text: np.ndarray,
    target_text: np.ndarray,
    target_descriptions: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    source_image = normalize(source_image[None, :])[0]
    source_text = normalize(source_text[None, :])[0]
    target_text = normalize(target_text[None, :])[0]
    source_descriptions = normalize(source_descriptions)
    target_descriptions = normalize(target_descriptions)
    source_distribution = normalize(
        np.concatenate([source_image[None, :], source_descriptions], axis=0)
    )
    target_distribution = normalize(
        np.concatenate([target_text[None, :], target_descriptions], axis=0)
    )
    source_weight, target_weight = relative_weights(
        source_distribution - source_text[None, :],
        target_distribution - source_text[None, :],
    )
    features = np.concatenate([source_distribution, target_distribution], axis=0)
    labels = np.array(
        [1] * len(source_distribution) + [2] * len(target_distribution)
    )
    weights = 100.0 * np.concatenate([source_weight, target_weight])
    classifier = SGDClassifier(
        penalty="l2",
        loss="hinge",
        random_state=seed,
        max_iter=1000,
        tol=1e-3,
    )
    classifier.fit(features, labels, sample_weight=weights)
    normal = classifier.coef_.reshape(-1)
    bias = float(classifier.intercept_[0])
    norm_squared = float(normal @ normal)
    if norm_squared <= 1e-12:
        raise ValueError("Degenerate AugCLIP separating hyperplane")
    modification = -((source_image @ normal + bias) / norm_squared) * normal
    ideal = source_image + modification
    error = float(np.mean(classifier.predict(features) != labels))
    return ideal, error


def compute_augclip_rows(
    config: dict[str, Any],
    pairs: dict[str, dict[str, list[tuple[str, Path, Path]]]],
    source_descriptions: dict[str, list[str]],
    encoder: ClipEncoder,
    seed: int,
) -> list[dict[str, Any]]:
    source_text_feature = encoder.encode_texts([config["source_text"]])[0]
    all_rows: list[dict[str, Any]] = []
    ideals: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    for method, concepts in pairs.items():
        for concept, concept_pairs in concepts.items():
            if not concept_pairs:
                continue
            target = config["concepts"][concept]
            target_text_feature = encoder.encode_texts([target["target_text"]])[0]
            target_description_features = encoder.encode_texts(
                target["target_descriptions"]
            )
            source_paths = [source for _, source, _ in concept_pairs]
            edited_paths = [edited for _, _, edited in concept_pairs]
            source_features = encoder.encode_images(source_paths)
            edited_features = normalize(encoder.encode_images(edited_paths))
            for index, (key, source_path, edited_path) in enumerate(concept_pairs):
                cache_key = (key, concept)
                if cache_key not in ideals:
                    descriptions = source_descriptions.get(
                        key, source_descriptions.get("_default")
                    )
                    if not descriptions:
                        raise ValueError(f"No source descriptions for image ID {key}")
                    source_description_features = encoder.encode_texts(descriptions)
                    ideals[cache_key] = augclip_ideal(
                        source_features[index],
                        source_description_features,
                        source_text_feature,
                        target_text_feature,
                        target_description_features,
                        seed,
                    )
                ideal, classification_error = ideals[cache_key]
                # This dot product matches the authors' released notebook.
                score = float(ideal @ edited_features[index])
                all_rows.append(
                    {
                        "dataset": config["dataset"],
                        "method": method,
                        "concept": concept,
                        "image_id": key,
                        "source_path": str(source_path),
                        "edited_path": str(edited_path),
                        "AugCLIP": score,
                        "svc_training_error": classification_error,
                    }
                )
    return all_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_augclip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["concept"])].append(float(row["AugCLIP"]))
    summary = []
    for (method, concept), scores in sorted(grouped.items()):
        values = np.asarray(scores)
        summary.append(
            {
                "method": method,
                "concept": concept,
                "n": len(values),
                "AugCLIP_mean": float(values.mean()),
                "AugCLIP_std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                "AugCLIP_sem": (
                    float(values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else math.nan
                ),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    with args.config.open() as handle:
        config = json.load(handle)
    if args.concepts:
        requested = set(args.concepts)
        unknown = requested - set(config["concepts"])
        if unknown:
            raise ValueError(f"Unknown concepts: {sorted(unknown)}")
        config["concepts"] = {
            name: spec
            for name, spec in config["concepts"].items()
            if name in requested
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs, audit = build_pairs(config, repo_root)
    with (args.output_dir / "pairing_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return

    encoder = ClipEncoder(args.clip_model, args.device, args.batch_size)
    if not args.skip_augclip:
        source_descriptions = load_source_descriptions(
            config["source_descriptions"], repo_root
        )
        augclip_rows = compute_augclip_rows(
            config, pairs, source_descriptions, encoder, args.seed
        )
        write_csv(args.output_dir / "augclip_per_image.csv", augclip_rows)
        augclip_summary = summarize_augclip(augclip_rows)
        write_csv(args.output_dir / "augclip_summary.csv", augclip_summary)

    omes_summary = []
    for method, concepts in pairs.items():
        complete = {
            concept: concept_pairs
            for concept, concept_pairs in concepts.items()
            if concept_pairs
        }
        if len(complete) != len(config["concepts"]):
            continue
        concept_features = {}
        activity_features = []
        for concept, concept_pairs in complete.items():
            source = encoder.encode_images([pair[1] for pair in concept_pairs])
            edited = encoder.encode_images([pair[2] for pair in concept_pairs])
            concept_features[concept] = (source, edited)
            activity_features.append(source)
        omes = compute_omes(
            concept_features,
            np.concatenate(activity_features, axis=0),
            args.inactive_threshold,
            args.omes_pooling,
        )
        with (args.output_dir / f"omes_{method}.json").open("w") as handle:
            json.dump(omes, handle, indent=2)
        omes_summary.append(
            {
                "method": method,
                "n_concepts": len(complete),
                "n_pairs_per_concept_min": min(map(len, complete.values())),
                "active_dimensions": omes["dimensions_active"],
                "OS": omes["OS"],
                "MES": omes["MES"],
                "OMES": omes["OMES"],
            }
        )
    write_csv(args.output_dir / "omes_summary.csv", omes_summary)
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
