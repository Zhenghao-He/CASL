from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analyze import aggregate
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-dir")
    parser.add_argument("--test-count", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.result_dir:
        cfg["result_dir"] = args.result_dir
    if args.test_count is not None:
        cfg["test_count"] = args.test_count
    result_dir = Path(cfg["result_dir"])
    frame = pd.read_csv(result_dir / "per_image_metrics.csv")
    summary = aggregate(frame, int(cfg["seed"]))
    output = result_dir / "sweep_summary_four_classifier.csv"
    summary.to_csv(output, index=False)
    print(
        f"[saved] {output} "
        f"({len(frame)} image/config rows, {len(summary)} configurations)"
    )


if __name__ == "__main__":
    main()
