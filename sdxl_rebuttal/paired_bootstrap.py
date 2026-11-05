from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CLASSIFIERS = ("resnet18", "mobilenetv2", "vgg16", "vit_b_16")


def subset(
    frame: pd.DataFrame, method: str, strength: float, topk: int | None
) -> pd.DataFrame:
    selected = frame[
        (frame["method"] == method) & np.isclose(frame["strength"], strength)
    ]
    if topk is not None:
        selected = selected[selected["topk"] == topk]
    return selected.set_index("image_id").sort_index()


def derived(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["target"] = frame[
        [f"{name}_target_change" for name in CLASSIFIERS]
    ].mean(axis=1)
    result["offtarget"] = frame[
        [f"{name}_mean_offtarget_change" for name in CLASSIFIERS]
    ].mean(axis=1)
    result["lpips"] = frame["lpips"]
    result["arcface"] = frame["arcface"]
    return result


def epr(frame: pd.DataFrame) -> float:
    values = []
    for name in CLASSIFIERS:
        target = frame[f"{name}_target_logit_change"].abs().mean()
        off = frame[f"{name}_mean_offtarget_logit_change"].mean()
        values.append(float(target / max(float(off), 1e-8)))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-image", required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--casl-method", default="casl")
    parser.add_argument("--casl-topk", type=int, required=True)
    parser.add_argument("--casl-strength", type=float, required=True)
    parser.add_argument("--baseline-strength", type=float, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_rows = pd.read_csv(args.per_image)
    all_rows = all_rows[all_rows["concept"] == args.concept]
    casl_raw = subset(
        all_rows, args.casl_method, args.casl_strength, args.casl_topk
    )
    baseline_raw = subset(all_rows, "asyrp_cnn", args.baseline_strength, None)
    common = casl_raw.index.intersection(baseline_raw.index)
    casl_raw, baseline_raw = casl_raw.loc[common], baseline_raw.loc[common]
    casl, baseline = derived(casl_raw), derived(baseline_raw)
    rng = np.random.default_rng(args.seed)
    metrics = ("target", "offtarget", "lpips", "arcface")
    samples = {metric: np.empty(args.draws) for metric in metrics}
    epr_samples = np.empty(args.draws)
    for draw in range(args.draws):
        indices = rng.integers(0, len(common), len(common))
        for metric in metrics:
            samples[metric][draw] = (
                casl[metric].iloc[indices] - baseline[metric].iloc[indices]
            ).mean()
        epr_samples[draw] = epr(casl_raw.iloc[indices]) - epr(
            baseline_raw.iloc[indices]
        )

    row: dict[str, float | int | str] = {
        "concept": args.concept,
        "n": len(common),
        "casl_method": args.casl_method,
        "casl_topk": args.casl_topk,
        "casl_strength": args.casl_strength,
        "baseline_strength": args.baseline_strength,
    }
    for metric in metrics:
        row[f"casl_{metric}"] = float(casl[metric].mean())
        row[f"baseline_{metric}"] = float(baseline[metric].mean())
        row[f"casl_minus_baseline_{metric}"] = float(
            casl[metric].mean() - baseline[metric].mean()
        )
        low, high = np.quantile(samples[metric], [0.025, 0.975])
        row[f"casl_minus_baseline_{metric}_ci_low"] = float(low)
        row[f"casl_minus_baseline_{metric}_ci_high"] = float(high)
    row["casl_epr"] = epr(casl_raw)
    row["baseline_epr"] = epr(baseline_raw)
    row["casl_minus_baseline_epr"] = row["casl_epr"] - row["baseline_epr"]
    low, high = np.quantile(epr_samples, [0.025, 0.975])
    row["casl_minus_baseline_epr_ci_low"] = float(low)
    row["casl_minus_baseline_epr_ci_high"] = float(high)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
