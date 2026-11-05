from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .utils import append_event, load_config, test_ids, write_json


GROUP = ["concept", "method", "range", "topk", "strength"]
METRICS = [
    "resnet18_target_change",
    "resnet18_mean_offtarget_change",
    "resnet18_max_offtarget_change",
    "resnet18_target_success",
    "mobilenetv2_target_change",
    "mobilenetv2_mean_offtarget_change",
    "mobilenetv2_max_offtarget_change",
    "mobilenetv2_target_success",
    "vgg16_target_change",
    "vgg16_mean_offtarget_change",
    "vgg16_max_offtarget_change",
    "vgg16_target_success",
    "vit_b_16_target_change",
    "vit_b_16_mean_offtarget_change",
    "vit_b_16_max_offtarget_change",
    "vit_b_16_target_success",
    "lpips",
    "arcface",
    "augclip_directional",
]


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for index in range(draws):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def aggregate(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(GROUP, dropna=False):
        row = dict(zip(GROUP, key))
        row["n"] = len(group)
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[metric] = float(finite.mean()) if len(finite) else float("nan")
            low, high = bootstrap_mean_ci(values, seed + len(rows) * 17)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        for classifier in ("resnet18", "mobilenetv2", "vgg16", "vit_b_16"):
            eligible = group[group[f"{classifier}_eligible_negative"] == 1]
            conversions = eligible[f"{classifier}_negative_to_positive"].to_numpy(dtype=float)
            metric = f"{classifier}_negative_to_positive_rate"
            row[f"{classifier}_eligible_negative_n"] = len(eligible)
            row[metric] = float(conversions.mean()) if len(conversions) else float("nan")
            low, high = bootstrap_mean_ci(
                conversions, seed + len(rows) * 17 + (1 if classifier == "resnet18" else 2)
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        # Paper Eq. 12–14: absolute classifier-logit target change divided
        # by the mean absolute non-target logit change, aggregated over the
        # image set. Signed probability change remains a separate edit-
        # direction metric and is intentionally not used in EPR.
        for classifier in ("resnet18", "mobilenetv2", "vgg16", "vit_b_16"):
            target_logit = group[f"{classifier}_target_logit_change"].abs().mean()
            off_target_logit = group[f"{classifier}_mean_offtarget_logit_change"].mean()
            name = "epr" if classifier == "resnet18" else f"{classifier}_epr"
            row[name] = float(target_logit / max(float(off_target_logit), 1e-8))
        row["four_classifier_mean_epr"] = float(
            np.mean(
                [
                    row["epr"],
                    row["mobilenetv2_epr"],
                    row["vgg16_epr"],
                    row["vit_b_16_epr"],
                ]
            )
        )
        target_changes = [
            row["resnet18_target_change"],
            row["mobilenetv2_target_change"],
            row["vgg16_target_change"],
            row["vit_b_16_target_change"],
        ]
        row["four_classifier_mean_target_change"] = float(np.mean(target_changes))
        row["four_classifier_positive_target_fraction"] = float(
            np.mean(np.asarray(target_changes) > 0)
        )
        row["four_classifier_mean_offtarget_change"] = float(
            np.mean(
                [
                    row["resnet18_mean_offtarget_change"],
                    row["mobilenetv2_mean_offtarget_change"],
                    row["vgg16_mean_offtarget_change"],
                    row["vit_b_16_mean_offtarget_change"],
                ]
            )
        )
        row["four_classifier_max_offtarget_change"] = float(
            np.mean(
                [
                    row["resnet18_max_offtarget_change"],
                    row["mobilenetv2_max_offtarget_change"],
                    row["vgg16_max_offtarget_change"],
                    row["vit_b_16_max_offtarget_change"],
                ]
            )
        )
        row["identity_cost"] = 1 - row["arcface"]
        rows.append(row)
    return pd.DataFrame(rows)


def macro_fixed_strength(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Macro-average concepts at each literal setting, bootstrapping image IDs.

    This is deliberately an alternate fixed-strength summary, not a replacement
    for the matched-target analysis.
    """
    rows = []
    keys = ["method", "range", "topk", "strength"]
    metrics = [
        "resnet18_target_change",
        "resnet18_mean_offtarget_change",
        "lpips",
        "arcface",
    ]
    for group_index, (key, group) in enumerate(
        frame.groupby(keys, dropna=False, sort=True)
    ):
        per_image = group.groupby("image_id")[metrics].mean(numeric_only=True)
        row = dict(zip(keys, key))
        row["concepts"] = int(group["concept"].nunique())
        row["images"] = int(per_image.shape[0])
        for metric_index, metric in enumerate(metrics):
            values = per_image[metric].to_numpy(float)
            row[metric] = float(np.nanmean(values))
            low, high = bootstrap_mean_ci(
                values, seed + group_index * 101 + metric_index
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_front(points: pd.DataFrame, cost: str) -> pd.DataFrame:
    points = points[np.isfinite(points["resnet18_target_change"]) & np.isfinite(points[cost])].copy()
    points = points.sort_values(["resnet18_target_change", cost], ascending=[True, True])
    kept = []
    for index, row in points.iterrows():
        target = row["resnet18_target_change"]
        nuisance = row[cost]
        dominated = (
            (points["resnet18_target_change"] >= target)
            & (points[cost] <= nuisance)
            & (
                (points["resnet18_target_change"] > target)
                | (points[cost] < nuisance)
            )
        ).any()
        if not dominated:
            kept.append(index)
    return points.loc[kept].sort_values("resnet18_target_change")


def interpolate_cost(front: pd.DataFrame, targets: np.ndarray, cost: str) -> np.ndarray:
    if front.empty:
        return np.full_like(targets, np.nan, dtype=float)
    x = front["resnet18_target_change"].to_numpy(float)
    y = front[cost].to_numpy(float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    # At a fixed target threshold, choose the lowest cost among points at least that strong.
    envelope = np.minimum.accumulate(y[::-1])[::-1]
    unique_x, unique_indices = np.unique(x, return_index=True)
    envelope = envelope[unique_indices]
    return np.interp(targets, unique_x, envelope, left=np.nan, right=np.nan)


def matched_target_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_pairs = []
    grid_rows = []
    for concept in sorted(summary["concept"].unique()):
        concept_frame = summary[summary["concept"] == concept]
        casl = concept_frame[concept_frame["method"] == "casl"]
        dense = concept_frame[concept_frame["method"] == "dense"]
        for _, casl_row in casl.iterrows():
            if dense.empty:
                continue
            distances = (dense["resnet18_target_change"] - casl_row["resnet18_target_change"]).abs()
            dense_row = dense.loc[distances.idxmin()]
            row = {
                "concept": concept,
                "casl_target": casl_row["resnet18_target_change"],
                "dense_target": dense_row["resnet18_target_change"],
                "target_mismatch": abs(
                    casl_row["resnet18_target_change"] - dense_row["resnet18_target_change"]
                ),
                "casl_range": casl_row["range"],
                "casl_topk": casl_row["topk"],
                "casl_strength": casl_row["strength"],
                "dense_range": dense_row["range"],
                "dense_strength": dense_row["strength"],
            }
            for metric in [
                "resnet18_mean_offtarget_change",
                "resnet18_max_offtarget_change",
                "lpips",
                "arcface",
                "epr",
            ]:
                row[f"casl_{metric}"] = casl_row[metric]
                row[f"dense_{metric}"] = dense_row[metric]
                if metric == "arcface":
                    row[f"casl_minus_dense_{metric}"] = casl_row[metric] - dense_row[metric]
                else:
                    row[f"casl_minus_dense_{metric}"] = casl_row[metric] - dense_row[metric]
            all_pairs.append(row)
        casl_front = pareto_front(casl, "resnet18_mean_offtarget_change")
        dense_front = pareto_front(dense, "resnet18_mean_offtarget_change")
        if casl_front.empty or dense_front.empty:
            continue
        lower = max(
            0.0,
            casl_front["resnet18_target_change"].min(),
            dense_front["resnet18_target_change"].min(),
        )
        upper = min(
            casl_front["resnet18_target_change"].max(),
            dense_front["resnet18_target_change"].max(),
        )
        if upper <= lower + 1e-6:
            continue
        targets = np.linspace(lower, upper, 5)
        for target in targets:
            row = {"concept": concept, "matched_target": target}
            for cost in [
                "resnet18_mean_offtarget_change",
                "lpips",
                "identity_cost",
            ]:
                for method, front in (("casl", casl_front), ("dense", dense_front)):
                    row[f"{method}_{cost}"] = interpolate_cost(
                        front, np.asarray([target]), cost
                    )[0]
            grid_rows.append(row)
    return pd.DataFrame(all_pairs), pd.DataFrame(grid_rows)


def fixed_budget(summary: pd.DataFrame) -> pd.DataFrame:
    budgets = [
        {"name": "strict", "leakage": 0.01, "lpips": 0.10, "arcface": 0.90},
        {"name": "moderate", "leakage": 0.02, "lpips": 0.20, "arcface": 0.80},
        {"name": "loose", "leakage": 0.05, "lpips": 0.35, "arcface": 0.65},
    ]
    rows = []
    for concept in sorted(summary["concept"].unique()):
        for method in sorted(summary["method"].unique()):
            subset = summary[
                (summary["concept"] == concept) & (summary["method"] == method)
            ]
            for budget in budgets:
                eligible = subset[
                    (subset["resnet18_mean_offtarget_change"] <= budget["leakage"])
                    & (subset["lpips"] <= budget["lpips"])
                    & (subset["arcface"] >= budget["arcface"])
                ]
                best = (
                    eligible.loc[eligible["resnet18_target_change"].idxmax()]
                    if len(eligible)
                    else None
                )
                rows.append(
                    {
                        "concept": concept,
                        "method": method,
                        "budget": budget["name"],
                        **{f"budget_{key}": value for key, value in budget.items() if key != "name"},
                        "eligible_configurations": len(eligible),
                        "best_target_change": (
                            float(best["resnet18_target_change"]) if best is not None else np.nan
                        ),
                        "best_range": best["range"] if best is not None else None,
                        "best_topk": best["topk"] if best is not None else None,
                        "best_strength": best["strength"] if best is not None else None,
                    }
                )
    return pd.DataFrame(rows)


def pareto_auc(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for concept in sorted(summary["concept"].unique()):
        concept_data = summary[summary["concept"] == concept]
        methods = {}
        for method in sorted(concept_data["method"].unique()):
            methods[method] = pareto_front(
                concept_data[concept_data["method"] == method],
                "resnet18_mean_offtarget_change",
            )
        if not methods:
            continue
        lower = max(max(0.0, front["resnet18_target_change"].min()) for front in methods.values() if len(front))
        upper = min(front["resnet18_target_change"].max() for front in methods.values() if len(front))
        if upper <= lower:
            continue
        grid = np.linspace(lower, upper, 101)
        for method, front in methods.items():
            leakage = interpolate_cost(front, grid, "resnet18_mean_offtarget_change")
            identity = interpolate_cost(front, grid, "identity_cost")
            scale = max(float(np.nanmax(leakage)), 1e-8)
            rows.append(
                {
                    "concept": concept,
                    "method": method,
                    "target_min": lower,
                    "target_max": upper,
                    "normalized_target_leakage_auc": float(
                        np.trapz(1 - leakage / scale, grid) / (upper - lower)
                    ),
                    "target_identity_auc": float(
                        np.trapz(1 - identity, grid) / (upper - lower)
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if len(result):
        macro = (
            result.groupby("method")[
                ["normalized_target_leakage_auc", "target_identity_auc"]
            ]
            .mean()
            .reset_index()
        )
        macro.insert(0, "concept", "MACRO")
        result = pd.concat([result, macro], ignore_index=True)
    return result


def make_plots(summary: pd.DataFrame, result_dir: Path) -> None:
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"casl": "#0072B2", "dense": "#D55E00"}
    for concept in sorted(summary["concept"].unique()):
        subset = summary[summary["concept"] == concept]
        for cost, label, filename in [
            ("resnet18_mean_offtarget_change", "Mean off-target change ↓", "target_leakage"),
            ("identity_cost", "1 − ArcFace identity ↓", "target_identity"),
            ("lpips", "LPIPS ↓", "target_lpips"),
        ]:
            fig, axis = plt.subplots(figsize=(5.2, 4.2))
            for method in sorted(subset["method"].unique()):
                points = subset[subset["method"] == method]
                axis.scatter(
                    points["resnet18_target_change"],
                    points[cost],
                    label=method,
                    alpha=0.65,
                    color=colors.get(method),
                )
                front = pareto_front(points, cost)
                axis.plot(
                    front["resnet18_target_change"],
                    front[cost],
                    color=colors.get(method),
                    linewidth=2,
                )
            axis.set_xlabel("Signed target-attribute probability change ↑")
            axis.set_ylabel(label)
            axis.set_title(concept)
            axis.grid(alpha=0.2)
            axis.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{concept}_{filename}.png", dpi=180)
            fig.savefig(plot_dir / f"{concept}_{filename}.pdf")
            plt.close(fig)


def nearest_row(summary: pd.DataFrame, concept: str, method: str) -> pd.Series | None:
    subset = summary[(summary["concept"] == concept) & (summary["method"] == method)]
    positive = subset[subset["resnet18_target_change"] > 0]
    if len(positive):
        subset = positive
    if not len(subset):
        return None
    # A deterministic middle-strength point, not a best-case selection.
    target = subset["resnet18_target_change"].median()
    return subset.loc[(subset["resnet18_target_change"] - target).abs().idxmin()]


def row_image_path(result_dir: Path, row: pd.Series, image_id: str) -> Path:
    topk = f"topk{int(row['topk'])}" if pd.notna(row["topk"]) else "dense"
    strength = "a" + f"{float(row['strength']):.5g}".replace("-", "m").replace(".", "p")
    bounds = row["range"]
    if isinstance(bounds, str):
        bounds = json.loads(bounds.replace("'", '"'))
    range_name = f"steps{int(bounds[0]):02d}-{int(bounds[1]):02d}"
    return (
        result_dir
        / "edits"
        / row["concept"]
        / row["method"]
        / range_name
        / topk
        / strength
        / image_id
    )


def qualitative_grids(cfg: dict, summary: pd.DataFrame) -> None:
    result_dir = Path(cfg["result_dir"])
    output_dir = result_dir / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = test_ids(cfg)[: min(4, int(cfg["test_count"]))]
    for concept in sorted(summary["concept"].unique()):
        casl_row = nearest_row(summary, concept, "casl")
        dense_row = nearest_row(summary, concept, "dense")
        external_row = nearest_row(summary, concept, "concept_slider")
        if casl_row is None or dense_row is None:
            continue
        columns = [
            "source",
            "inversion",
            "dense",
            "Concept Slider" if external_row is not None else "external (N/A)",
            "CASL",
        ]
        cell = 256
        header = 34
        canvas = Image.new("RGB", (cell * len(columns), header + cell * len(ids)), "white")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(columns):
            draw.text((column * cell + 8, 10), label, fill="black")
        for row_index, image_id in enumerate(ids):
            paths = [
                Path(cfg["data_dir"]) / image_id,
                result_dir / "reconstruction" / image_id,
                row_image_path(result_dir, dense_row, image_id),
                (
                    row_image_path(result_dir, external_row, image_id)
                    if external_row is not None
                    else None
                ),
                row_image_path(result_dir, casl_row, image_id),
            ]
            for column, path in enumerate(paths):
                top = header + row_index * cell
                if path is None or not path.is_file():
                    tile = Image.new("RGB", (cell, cell), (235, 235, 235))
                    tile_draw = ImageDraw.Draw(tile)
                    tile_draw.text((75, 118), "not included", fill=(80, 80, 80))
                else:
                    tile = Image.open(path).convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
                canvas.paste(tile, (column * cell, top))
                if column == 0:
                    draw.text((8, top + 8), image_id, fill="white", stroke_width=2, stroke_fill="black")
        canvas.save(output_dir / f"{concept}_grid.png")


def write_report(
    cfg: dict,
    summary: pd.DataFrame,
    matched_grid: pd.DataFrame,
    auc: pd.DataFrame,
    inversion: pd.DataFrame,
) -> None:
    result_dir = Path(cfg["result_dir"])
    concepts = ", ".join(sorted(summary["concept"].unique()))
    methods = ", ".join(sorted(summary["method"].unique()))
    inv_lpips = inversion["inversion_lpips"].mean()
    inv_arc = inversion["inversion_arcface"].mean()
    lines = [
        f"# CASL–SDXL {cfg['stage']} report",
        "",
        "## Completed setup",
        "",
        f"- Backbone: `{cfg['model_id']}` base model only; frozen throughout.",
        f"- Concepts completed: {concepts}. Methods completed: {methods}.",
        f"- Images: {cfg['test_count']} fixed CelebA-HQ IDs; 50-step DDIM inversion with the same neutral prompt for reconstruction and editing.",
        f"- Primary space: SDXL U-Net mid-block. Resolution: {cfg['resolution']}×{cfg['resolution']}.",
        "- Dense comparator label: **Asyrp-style matched dense H-space** (not an exact Asyrp reproduction).",
        "- All edit-induced metrics use the inversion reconstruction as reference; only inversion error uses the source image.",
        "",
        "## Inversion",
        "",
        f"Mean inversion LPIPS: **{inv_lpips:.4f}**. Mean ArcFace similarity: **{inv_arc:.4f}**.",
        "",
        "## Matched-target and Pareto results",
        "",
    ]
    if len(matched_grid):
        macro = matched_grid.groupby(lambda _: True).mean(numeric_only=True).iloc[0]
        lines.extend(
            [
                "Across the common target-strength support of each concept (five-point interpolation on each method's Pareto envelope):",
                "",
                f"- CASL mean off-target change: {macro.get('casl_resnet18_mean_offtarget_change', np.nan):.4f}; dense: {macro.get('dense_resnet18_mean_offtarget_change', np.nan):.4f}.",
                f"- CASL LPIPS: {macro.get('casl_lpips', np.nan):.4f}; dense: {macro.get('dense_lpips', np.nan):.4f}.",
                f"- CASL ArcFace: {1-macro.get('casl_identity_cost', np.nan):.4f}; dense: {1-macro.get('dense_identity_cost', np.nan):.4f}.",
                "",
            ]
        )
        for concept, group in matched_grid.groupby("concept"):
            values = group.mean(numeric_only=True)
            lines.append(
                f"- {concept}: matched-target mean leakage CASL "
                f"{values['casl_resnet18_mean_offtarget_change']:.4f} vs dense "
                f"{values['dense_resnet18_mean_offtarget_change']:.4f}; LPIPS "
                f"{values['casl_lpips']:.4f} vs {values['dense_lpips']:.4f}; ArcFace "
                f"{1-values['casl_identity_cost']:.4f} vs "
                f"{1-values['dense_identity_cost']:.4f}."
            )
        lines.append("")
    else:
        lines.append("No non-degenerate common target-strength interval was available in this stage.\n")
    lines.extend(
        [
            "The complete sweep, including zero/failed/weak settings, is retained in `sweep_summary.csv`; the comparison above is not a single arbitrarily selected strength.",
            "",
            "## Baseline feasibility and failed attempts",
            "",
            (
                "- Official Concept Slider SDXL weights are included for Smiling and Age/Young "
                "on exactly the same inversion latents, neutral prompt, IDs, and 50-step schedule; "
                "no official Eyeglasses SDXL weight was available."
                if "concept_slider" in summary["method"].unique()
                else "- Concept Slider was not completed in this stage."
            ),
            "- MasaCtrl was excluded: its official SDXL code path is shared-noise synthesis, while its official real-image path targets SD1.4; porting it would require a substantial unsupported reimplementation.",
            "- The dense control is the controlled causal comparator; it shares the alignment data, CLIP loss, preservation loss, optimizer budget, layer, timestep candidates, and norm-matched strength sweep.",
            "",
            "## Limitations and claim scope",
            "",
            "This experiment tests a large-scale text-conditioned latent diffusion U-Net whose backbone was not fine-tuned on FFHQ/CelebA. It gives a positive end-to-end result for Smiling, a weaker mixed result for Young, and a failed Eyeglasses edit; it therefore does not establish concept-universal superiority. The SAE and semantic map do use target-domain images, so this is not fully zero-shot domain generalization. It provides no empirical evidence for DiTs, SD3, FLUX, flow matching, or rectified flow.",
            "",
            "## Rebuttal-ready wording",
            "",
        ]
    )
    if len(matched_grid):
        macro = matched_grid.mean(numeric_only=True)
        casl_leak = macro.get("casl_resnet18_mean_offtarget_change", np.nan)
        dense_leak = macro.get("dense_resnet18_mean_offtarget_change", np.nan)
        casl_id = 1 - macro.get("casl_identity_cost", np.nan)
        dense_id = 1 - macro.get("dense_identity_cost", np.nan)
        paragraph = (
            f"We added an end-to-end experiment on the frozen SDXL-base text-conditioned latent U-Net "
            f"({cfg['test_count']} fixed real CelebA-HQ images; {concepts}; 50-step DDIM inversion). "
            f"At matched target strength over the common Pareto support, CASL obtained mean off-target "
            f"attribute change {casl_leak:.4f} versus {dense_leak:.4f} for a norm- and budget-matched "
            f"dense H-space intervention, with ArcFace identity {casl_id:.4f} versus {dense_id:.4f}. "
            "CASL produced a clear Smiling edit, but the aggregate comparison was mixed: dense was better "
            "on Young and neither method produced a meaningful Eyeglasses edit. Thus the result demonstrates "
            "an end-to-end SDXL transfer path rather than universal specificity superiority. The backbone was "
            "not fine-tuned on the face datasets, but CASL still learns its SAE/alignment on target-domain "
            "images, and we do not claim empirical transfer to DiTs or rectified-flow models."
        )
    else:
        paragraph = (
            "The completed run did not provide a non-degenerate matched-target interval, so it does not "
            "yet support a comparative specificity claim. It verifies the SDXL execution path only."
        )
    lines.append(paragraph)
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "- `run_manifest.json` records the exact config and software versions.",
            "- `per_image_metrics.csv` contains every image/strength/configuration result.",
            "- `matched_target_grid.csv`, `fixed_budget_summary.csv`, and `pareto_auc.csv` contain alternate fair summaries.",
            "- `events.jsonl` records completed stages and wall-clock costs.",
        ]
    )
    (result_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (result_dir / "REBUTTAL_PARAGRAPH.md").write_text(paragraph + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result_dir = Path(cfg["result_dir"])
    frame = pd.read_csv(result_dir / "per_image_metrics.csv")
    # Preserve list-valued timestep ranges through CSV round trips.
    frame["range"] = frame["range"].astype(str)
    summary = aggregate(frame, int(cfg["seed"]))
    summary.to_csv(result_dir / "sweep_summary.csv", index=False)
    macro = macro_fixed_strength(frame, int(cfg["seed"]))
    macro.to_csv(result_dir / "macro_fixed_strength_summary.csv", index=False)
    matched_all, matched_grid = matched_target_tables(summary)
    matched_all.to_csv(result_dir / "matched_target_all.csv", index=False)
    matched_grid.to_csv(result_dir / "matched_target_grid.csv", index=False)
    fixed = fixed_budget(summary)
    fixed.to_csv(result_dir / "fixed_budget_summary.csv", index=False)
    auc = pareto_auc(summary)
    auc.to_csv(result_dir / "pareto_auc.csv", index=False)
    make_plots(summary, result_dir)
    qualitative_grids(cfg, summary)
    inversion = pd.read_csv(result_dir / "inversion_metrics.csv")
    write_report(cfg, summary, matched_grid, auc, inversion)
    append_event(
        cfg,
        "analysis_complete",
        configurations=len(summary),
        matched_pairs=len(matched_all),
        matched_grid_rows=len(matched_grid),
    )
    print(f"[saved] {result_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
