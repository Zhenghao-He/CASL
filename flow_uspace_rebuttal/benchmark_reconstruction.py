"""Choose the Euler step count using reconstruction only (no edit metrics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
import yaml

from run import decode, encode, euler, load_backbone, load_path_batch, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="flow_uspace_rebuttal/config_large.yaml")
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with resolve(args.config).open() as handle:
        cfg = yaml.safe_load(handle)
    with resolve(args.ids_file).open() as handle:
        ids = [line.strip() for line in handle if line.strip()][: args.count]
    paths = [resolve(cfg["source_dir"]) / image_id for image_id in ids]
    device = torch.device("cuda")
    model, vae = load_backbone(cfg, device)
    images = load_path_batch(cfg, paths).to(device)
    metric = lpips.LPIPS(net="alex").to(device).eval()
    scale = float(cfg["scale_factor"])
    results = []
    with torch.no_grad():
        z0 = encode(vae, images, scale)
        for steps in args.steps:
            noise = euler(model, z0, 0, 1, steps)
            reconstruction = decode(vae, euler(model, noise, 1, 0, steps), scale)
            values = metric(images, reconstruction * 2 - 1).flatten()
            results.append(
                {
                    "steps": steps,
                    "n": len(ids),
                    "lpips_mean": float(values.mean()),
                    "lpips_std": float(values.std()),
                }
            )
            print(results[-1])
    resolve(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
