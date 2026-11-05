#!/usr/bin/env python3
"""
Build concept-vs-concept latent overlap heatmaps for:
1. raw top-100 activation-ranked latent indices
2. probe-selected top-k latent indices
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


REPO_ROOT = Path("/p/realai/zhenghao/LoRA_SAE")
DEFAULT_TOP100_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_concept_latents"
DEFAULT_PROBE_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_linear_probe"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_overlap_summary"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text(x: float, y: float, text: str, size: int) -> str:
    return f"0 0 0 rg BT /F1 {size} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm ({_pdf_escape(text)}) Tj ET"


def _pdf_rect(x: float, y: float, w: float, h: float, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"{r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f"


def _pdf_line(x1: float, y1: float, x2: float, y2: float, width: float = 1.0) -> str:
    return f"0 0 0 RG {width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S"


def _write_single_page_pdf(ops: list[str], width: int, height: int, output_pdf: Path) -> None:
    content = "\n".join(ops).encode("latin-1")
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
        f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode("latin-1")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    pdf.extend((f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n").encode("latin-1"))
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.write_bytes(pdf)


def load_top100_sets(top100_dir: Path) -> dict[str, set[int]]:
    concept_sets: dict[str, set[int]] = {}
    for path in sorted(top100_dir.glob("*_top100_indices.pt")):
        concept = path.name.replace("_top100_indices.pt", "")
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            indices = payload.flatten().tolist()
        elif isinstance(payload, (list, tuple)):
            indices = list(payload)
        else:
            raise TypeError(f"Unsupported top100 format: {path}")
        concept_sets[concept] = {int(v) for v in indices}
    if not concept_sets:
        raise FileNotFoundError(f"No *_top100_indices.pt files found in {top100_dir}")
    return concept_sets


def load_probe_sets(probe_dir: Path, top_k: int) -> dict[str, set[int]]:
    concept_sets: dict[str, set[int]] = {}
    for path in sorted(probe_dir.glob("*_linear_probe.json")):
        payload = json.loads(path.read_text())
        concept = payload["concept"]
        ranked = payload.get("top_ranked_latents", [])
        if not ranked:
            continue
        subset = ranked[: min(top_k, len(ranked))]
        concept_sets[concept] = {int(item["latent_index"]) for item in subset}
    if not concept_sets:
        raise FileNotFoundError(f"No *_linear_probe.json files found in {probe_dir}")
    return concept_sets


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def compute_matrix(concept_sets: dict[str, set[int]]) -> tuple[list[str], list[list[float]], list[list[int]]]:
    concepts = sorted(concept_sets)
    ratio_matrix: list[list[float]] = []
    count_matrix: list[list[int]] = []
    for row_c in concepts:
        ratio_row: list[float] = []
        count_row: list[int] = []
        for col_c in concepts:
            overlap_count = len(concept_sets[row_c] & concept_sets[col_c])
            ratio_row.append(jaccard(concept_sets[row_c], concept_sets[col_c]))
            count_row.append(overlap_count)
        ratio_matrix.append(ratio_row)
        count_matrix.append(count_row)
    return concepts, ratio_matrix, count_matrix


def save_csv(concepts: list[str], ratio_matrix: list[list[float]], count_matrix: list[list[int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["concept"] + concepts)
        for concept, row in zip(concepts, ratio_matrix):
            writer.writerow([concept] + [f"{value:.6f}" for value in row])

    counts_csv = output_csv.with_name(output_csv.stem + "_counts.csv")
    with counts_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["concept"] + concepts)
        for concept, row in zip(concepts, count_matrix):
            writer.writerow([concept] + row)


def _heatmap_color(value: float) -> tuple[float, float, float]:
    # White -> teal-green
    low = (1.0, 1.0, 1.0)
    high = (0.16, 0.56, 0.52)
    return tuple(low[i] + (high[i] - low[i]) * value for i in range(3))


def plot_heatmap_pdf(concepts: list[str], ratio_matrix: list[list[float]], output_pdf: Path, title: str) -> None:
    n = len(concepts)
    width = max(860, 250 + n * 100)
    height = max(860, 250 + n * 100)
    margin_left = 155
    margin_right = 90
    margin_top = 105
    margin_bottom = 145
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    cell_w = chart_w / n
    cell_h = chart_h / n
    x0 = margin_left
    y0 = margin_bottom

    ops: list[str] = []
    ops.append(_pdf_text(width / 2 - 210, height - 52, title, 28))

    # cells
    for i, row_name in enumerate(concepts):
        for j, col_name in enumerate(concepts):
            value = ratio_matrix[i][j]
            x = x0 + j * cell_w
            y = y0 + (n - 1 - i) * cell_h
            ops.append(_pdf_rect(x, y, cell_w, cell_h, _heatmap_color(value)))
            ops.append(_pdf_line(x, y, x + cell_w, y, 0.4))
            ops.append(_pdf_line(x, y, x, y + cell_h, 0.4))
            ops.append(_pdf_text(x + cell_w * 0.16, y + cell_h * 0.38, f"{value:.2f}", 16))

    # outer border
    ops.append(_pdf_line(x0, y0, x0 + chart_w, y0, 1.0))
    ops.append(_pdf_line(x0, y0, x0, y0 + chart_h, 1.0))
    ops.append(_pdf_line(x0 + chart_w, y0, x0 + chart_w, y0 + chart_h, 1.0))
    ops.append(_pdf_line(x0, y0 + chart_h, x0 + chart_w, y0 + chart_h, 1.0))

    # row labels
    for i, concept in enumerate(concepts):
        y = y0 + (n - 1 - i) * cell_h + cell_h * 0.36
        ops.append(_pdf_text(x0 - 112, y, concept, 19))

    # column labels (rotated)
    for j, concept in enumerate(concepts):
        x = x0 + j * cell_w + cell_w * 0.30
        escaped = _pdf_escape(concept)
        ops.append(f"0 0 0 rg BT /F1 19 Tf 0.9063 0.4226 -0.4226 0.9063 {x:.1f} {y0 - 82:.1f} Tm ({escaped}) Tj ET")

    # simple legend
    legend_x = width - 55
    legend_y = y0
    legend_h = chart_h
    steps = 10
    step_h = legend_h / steps
    for s in range(steps):
        value = s / (steps - 1)
        y = legend_y + s * step_h
        ops.append(_pdf_rect(legend_x, y, 18, step_h, _heatmap_color(value)))
    ops.append(_pdf_line(legend_x, legend_y, legend_x + 18, legend_y, 0.6))
    ops.append(_pdf_line(legend_x, legend_y + legend_h, legend_x + 18, legend_y + legend_h, 0.6))
    ops.append(_pdf_text(legend_x - 10, legend_y - 22, "0.0", 15))
    ops.append(_pdf_text(legend_x - 10, legend_y + legend_h + 6, "1.0", 15))
    ops.append(_pdf_text(legend_x - 20, legend_y + legend_h + 32, "Jaccard", 17))

    _write_single_page_pdf(ops, width, height, output_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize latent overlap heatmaps.")
    parser.add_argument("--top100-dir", type=Path, default=DEFAULT_TOP100_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-top-k", type=int, default=10, help="Use top-k probe-ranked latents per concept.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_sets = load_top100_sets(args.top100_dir)
    probe_sets = load_probe_sets(args.probe_dir, args.probe_top_k)

    raw_concepts, raw_ratio, raw_count = compute_matrix(raw_sets)
    probe_concepts, probe_ratio, probe_count = compute_matrix(probe_sets)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = args.output_dir / "raw_top100_overlap_jaccard.csv"
    raw_pdf = args.output_dir / "raw_top100_overlap_jaccard.pdf"
    save_csv(raw_concepts, raw_ratio, raw_count, raw_csv)
    plot_heatmap_pdf(raw_concepts, raw_ratio, raw_pdf, "Raw Top-100 Latent Overlap (Jaccard)")

    probe_csv = args.output_dir / f"probe_top{args.probe_top_k}_overlap_jaccard.csv"
    probe_pdf = args.output_dir / f"probe_top{args.probe_top_k}_overlap_jaccard.pdf"
    save_csv(probe_concepts, probe_ratio, probe_count, probe_csv)
    plot_heatmap_pdf(probe_concepts, probe_ratio, probe_pdf, f"Probe Top-{args.probe_top_k} Latent Overlap (Jaccard)")

    print(f"Saved raw overlap CSV to: {raw_csv}")
    print(f"Saved raw overlap figure to: {raw_pdf}")
    print(f"Saved probe overlap CSV to: {probe_csv}")
    print(f"Saved probe overlap figure to: {probe_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
