#!/usr/bin/env python3
"""Reweight completed OS/MES results without recomputing image embeddings."""

import argparse
import csv
from pathlib import Path


DATASETS = {
    "CelebA-HQ": "celeba_hq_table2",
    "LSUN-Church": "lsun_church_table12",
    "AFHQ-Dog": "afhq_dog_table13",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "omes_alpha_sensitivity.csv",
    )
    return parser.parse_args()


def read_summary(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    output_rows = []
    for dataset, directory in DATASETS.items():
        rows = read_summary(args.results_root / directory / "omes_summary.csv")
        for row in rows:
            if row["method"] == "Reconstruction":
                continue
            os_score = float(row["OS"])
            mes_score = float(row["MES"])
            for alpha_index in range(11):
                alpha = alpha_index / 10
                output_rows.append(
                    {
                        "dataset": dataset,
                        "method": row["method"],
                        "alpha_os_weight": f"{alpha:.1f}",
                        "mes_weight": f"{1 - alpha:.1f}",
                        "OS": f"{os_score:.9f}",
                        "MES": f"{mes_score:.9f}",
                        "OMES_alpha": f"{alpha * os_score + (1 - alpha) * mes_score:.9f}",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
