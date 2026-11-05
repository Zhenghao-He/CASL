from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis


ROOT = Path(__file__).resolve().parents[1]

# Strongest positive target-CLIP matches within 15% relative tolerance after
# the complete 8-image sweep. These are fixed before ArcFace is computed.
PAIRS = {
    "smiling": {"dense": (None, 4.0), "casl": (16.0, -0.1)},
    "beard": {"dense": (None, 2.0), "casl": (1.0, -4.0)},
    "no_beard": {"dense": (None, -4.0), "casl": (8.0, -0.5)},
    "young": {"dense": (None, -1.0), "casl": (8.0, -0.5)},
    "blond_hair": {"dense": (None, 4.0), "casl": (8.0, 0.1)},
}

PAIRS32 = {
    "smiling": {"dense": (None, 4.0), "casl": (16.0, -0.5)},
    "no_beard": {"dense": (None, -4.0), "casl": (8.0, -0.1)},
    "blond_hair": {"dense": (None, 4.0), "casl": (8.0, 4.0)},
}

PAIRS32_INTERP = {
    "smiling": {"dense": (None, 4.0), "casl": (16.0, -0.43)},
    "no_beard": {"dense": (None, -4.0), "casl": (8.0, -0.075)},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/flow_uspace_large")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--profile",
        choices=["validation8", "fixed32", "interp32"],
        default="validation8",
    )
    args = parser.parse_args()
    result_dir = ROOT / args.result_dir
    pairs = {
        "validation8": PAIRS,
        "fixed32": PAIRS32,
        "interp32": PAIRS32_INTERP,
    }[args.profile]
    metrics_name = {
        "validation8": "metrics_per_image.csv",
        "fixed32": "metrics32_curve_per_image.csv",
        "interp32": "metrics32_interp_per_image.csv",
    }[args.profile]
    metrics = pd.read_csv(result_dir / metrics_name)

    face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
    cache = {}

    def embedding(path):
        key = str(path)
        if key not in cache:
            detections = face.get(cv2.imread(key))
            cache[key] = None if not detections else detections[0].normed_embedding
        return cache[key]

    rows = []
    rng = np.random.default_rng(42)
    for concept, methods in pairs.items():
        selected = {}
        for method, (topk, strength) in methods.items():
            subset = metrics[
                (metrics.concept == concept)
                & (metrics.method == method)
                & (metrics.strength == strength)
            ]
            if topk is not None:
                subset = subset[subset.topk == topk]
            subset = subset.sort_values("image_id").copy()
            identities = []
            for record in subset.to_dict("records"):
                reference = embedding(result_dir / "inversion" / record["image_id"])
                edited = embedding(ROOT / record["path"])
                identities.append(
                    np.nan
                    if reference is None or edited is None
                    else float(np.dot(reference, edited))
                )
            subset["arcface"] = identities
            selected[method] = subset

        dense = selected["dense"].set_index("image_id")
        casl = selected["casl"].set_index("image_id")
        common = dense.index.intersection(casl.index)
        lpips_delta = (casl.loc[common, "lpips"] - dense.loc[common, "lpips"]).to_numpy()
        arcface_delta = (
            casl.loc[common, "arcface"] - dense.loc[common, "arcface"]
        ).to_numpy()

        def interval(values):
            values = values[np.isfinite(values)]
            samples = rng.choice(values, (args.bootstrap, len(values)), replace=True).mean(1)
            return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]

        rows.append(
            {
                "concept": concept,
                "n": len(common),
                "dense_topk": methods["dense"][0],
                "dense_strength": methods["dense"][1],
                "casl_topk": methods["casl"][0],
                "casl_strength": methods["casl"][1],
                "dense_target_clip_change": float(dense.target_clip_change.mean()),
                "casl_target_clip_change": float(casl.target_clip_change.mean()),
                "dense_lpips": float(dense.lpips.mean()),
                "casl_lpips": float(casl.lpips.mean()),
                "lpips_delta_casl_minus_dense": float(np.mean(lpips_delta)),
                "lpips_delta_95ci": interval(lpips_delta),
                "dense_arcface": float(dense.arcface.mean()),
                "casl_arcface": float(casl.arcface.mean()),
                "arcface_delta_casl_minus_dense": float(np.nanmean(arcface_delta)),
                "arcface_delta_95ci": interval(arcface_delta),
            }
        )
    output = result_dir / f"matched_target_identity_{args.profile}.json"
    output.write_text(json.dumps(rows, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
