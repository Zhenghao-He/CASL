#!/usr/bin/env python3
"""Create compact, rebuttal-ready tables from the two alpha experiments."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONCEPT_ORDER = ["smiling", "young", "big_nose", "beards", "blond_hair"]
DISPLAY = {
    "smiling": "Smiling",
    "young": "Young",
    "big_nose": "Big Nose",
    "beards": "Beards",
    "blond_hair": "Blond Hair",
}
NATIVE_GRIDS = {
    "CASL": [0, 32, 64, 128, 192],
    "ASYRP": [0, 0.5, 1, 1.5, 2],
    "Boundary": [0, 5, 9, 15, 19],
}


def reviewer1(frame):
    lines = [
        "# Reviewer 1: CASL alpha sweep",
        "",
        "Eight fixed CelebA-HQ images per concept. EPR and attribute changes are "
        "averaged over ResNet-18, VGG-16, MobileNetV2, and ViT-B/16; LPIPS uses "
        "AlexNet; CLIP-Score uses ViT-L/14. All edit-induced quantities use the "
        "alpha=0 reconstruction as reference.",
        "",
        "| Concept | alpha | EPR ↑ | CLIP-Score ↑ | LPIPS ↓ | Target delta ↑ | Off-target delta ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for concept in CONCEPT_ORDER:
        rows = frame[frame.concept == concept].sort_values("alpha")
        for row in rows.itertuples():
            lines.append(
                f"| {DISPLAY[concept]} | {row.alpha:g} | {row.epr:.3f} | "
                f"{row.clip_score:.3f} | {row.lpips:.3f} | "
                f"{row.target_delta:.3f} | {row.nontarget_delta:.3f} |"
            )
    lines.extend(
        [
            "",
            "OS/MES/OMES characterize the selected SAE coordinates and therefore "
            "do not change when only the inference multiplier alpha changes. "
            "Their fixed top-1 values are:",
            "",
            "| Concept | OS ↑ | MES ↑ | OMES ↑ |",
            "|---|---:|---:|---:|",
        ]
    )
    for concept in CONCEPT_ORDER:
        row = frame[frame.concept == concept].iloc[0]
        lines.append(
            f"| {DISPLAY[concept]} | {row.OS:.3f} | {row.MES:.3f} | {row.OMES:.3f} |"
        )

    nonzero = frame[frame.alpha > 0].copy()
    selected = []
    for concept in CONCEPT_ORDER:
        candidates = nonzero[
            (nonzero.concept == concept) & (nonzero.lpips <= 0.4)
        ]
        if candidates.empty:
            candidates = nonzero[nonzero.concept == concept]
        selected.append(candidates.loc[candidates.clip_score.idxmax()])
    selected = pd.DataFrame(selected)
    lines.extend(
        [
            "",
            "A reproducible validation rule, rather than visual inspection, is to "
            "maximize CLIP-Score subject to LPIPS <= 0.40. It selects:",
            "",
            "| Concept | Selected alpha | EPR ↑ | CLIP-Score ↑ | LPIPS ↓ | Target delta ↑ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected.itertuples():
        lines.append(
            f"| {DISPLAY[row.concept]} | {row.alpha:g} | {row.epr:.3f} | "
            f"{row.clip_score:.3f} | {row.lpips:.3f} | {row.target_delta:.3f} |"
        )
    lines.extend(
        [
            "",
            "Important diagnostic: for Big Nose, absolute CLIP-Score increases "
            "while the independent classifier target delta is slightly negative. "
            "Thus CLIP alone is not a universally reliable selector; a preservation "
            "constraint plus an independent target check is safer.",
            "",
        ]
    )
    (RESULTS / "reviewer1_alpha_table.md").write_text("\n".join(lines))


def nearest_row(group, strength):
    index = (group.strength - strength).abs().idxmin()
    return group.loc[index]


def reviewer2(frame):
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for method, grid in NATIVE_GRIDS.items():
        for level, strength in zip(levels, grid):
            selected = []
            for concept in CONCEPT_ORDER:
                group = frame[
                    (frame.method == method) & (frame.concept == concept)
                ]
                if group.empty:
                    continue
                selected.append(nearest_row(group, strength))
            if len(selected) != len(CONCEPT_ORDER):
                continue
            selected = pd.DataFrame(selected)
            rows.append(
                {
                    "method": method,
                    "level": level,
                    "native_strength": strength,
                    "epr": selected.epr.mean(),
                    "target_delta": selected.target_delta.mean(),
                    "off_target": selected.nontarget_delta.mean(),
                }
            )
    compact = pd.DataFrame(rows)
    compact.to_csv(RESULTS / "epr_sweeps/macro_normalized.csv", index=False)

    lines = [
        "# Reviewer 2: native intensity sweeps",
        "",
        "Eight fixed CelebA-HQ images and five concepts. The normalized level is "
        "only an index over five sampled settings from each method's own native "
        "sweep; raw coefficients (and Boundary grid indices) are not assumed to "
        "be comparable across methods.",
        "",
        "| Method | Native strengths at normalized 0/.25/.5/.75/1 | EPR at normalized 0/.25/.5/.75/1 ↑ | Target delta at normalized 0/.25/.5/.75/1 ↑ |",
        "|---|---|---|---|",
    ]
    for method, grid in NATIVE_GRIDS.items():
        group = compact[compact.method == method].sort_values("level")
        if len(group) != 5:
            continue
        strengths = "/".join(f"{value:g}" for value in grid)
        eprs = "/".join(f"{value:.3f}" for value in group.epr)
        targets = "/".join(f"{value:.3f}" for value in group.target_delta)
        lines.append(f"| {method} | {strengths} | {eprs} | {targets} |")
    lines.extend(
        [
            "",
            "This table reports the full trajectories rather than each method's "
            "best point, so preservation caused merely by weak editing is visible. "
            "Per-concept values and off-target changes are retained in "
            "`results/epr_sweeps/summary.csv`.",
            "",
        ]
    )
    (RESULTS / "reviewer2_epr_table.md").write_text("\n".join(lines))


def main():
    reviewer1(pd.read_csv(RESULTS / "casl_alpha/summary.csv"))
    reviewer2(pd.read_csv(RESULTS / "epr_sweeps/summary.csv"))
    print("Wrote rebuttal tables")


if __name__ == "__main__":
    main()
