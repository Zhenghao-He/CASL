#!/usr/bin/env python3
"""Paired-image bootstrap for CASL minus Asyrp AugCLIP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    casl = data[data.method == "CASL-Steer"][
        ["concept", "image_id", "AugCLIP"]
    ].rename(columns={"AugCLIP": "casl"})
    asyrp = data[data.method == "Asyrp"][
        ["concept", "image_id", "AugCLIP"]
    ].rename(columns={"AugCLIP": "asyrp"})
    paired = casl.merge(asyrp, on=["concept", "image_id"], validate="one_to_one")
    paired["delta"] = paired.casl - paired.asyrp
    rng = np.random.default_rng(args.seed)

    rows = []
    groups: list[tuple[str, np.ndarray]] = [
        (concept, group.delta.to_numpy())
        for concept, group in paired.groupby("concept", sort=True)
    ]
    macro = paired.pivot(index="image_id", columns="concept", values="delta")
    groups.append(("Macro_common", macro.mean(axis=1).dropna().to_numpy()))
    for concept, values in groups:
        samples = values[
            rng.integers(0, len(values), size=(args.samples, len(values)))
        ].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "concept": concept,
                "n": len(values),
                "CASL_minus_Asyrp": values.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_samples": args.samples,
                "seed": args.seed,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
