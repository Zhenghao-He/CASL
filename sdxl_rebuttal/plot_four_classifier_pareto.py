from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    styles = {"casl": ("#0072B2", "o"), "casl_lowrank": ("#009E73", "o"), "asyrp_cnn": ("#D55E00", "s")}
    for cost, ylabel, name in [
        ("four_classifier_mean_offtarget_change", "Four-classifier mean off-target ↓", "target_leakage"),
        ("lpips", "LPIPS ↓", "target_lpips"),
        ("identity_cost", "1 − ArcFace ↓", "target_identity"),
    ]:
        figure, axis = plt.subplots(figsize=(5.2, 4.2))
        local = frame.copy()
        local["identity_cost"] = 1 - local["arcface"]
        for method, group in local.groupby("method"):
            color, marker = styles.get(method, (None, "o"))
            axis.plot(
                group["four_classifier_mean_target_change"],
                group[cost],
                marker=marker,
                color=color,
                label=method,
            )
            for _, row in group.iterrows():
                axis.annotate(
                    f"α={row['strength']:g}",
                    (row["four_classifier_mean_target_change"], row[cost]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_xlabel("Four-classifier mean signed target change ↑")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=180)
        figure.savefig(output / f"{name}.pdf")
        plt.close(figure)


if __name__ == "__main__":
    main()
