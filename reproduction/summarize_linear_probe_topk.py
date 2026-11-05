#!/usr/bin/env python3
"""
Aggregate top-k linear-probe sweep results into a CSV and a PDF line chart.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path("/p/realai/zhenghao/LoRA_SAE")
DEFAULT_INPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_linear_probe_sweep"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_linear_probe_topk_summary"
TARGET_KS = [1, 2, 4, 8, 16, 32, 64, 100]
PALETTE = [
    (0.235, 0.431, 0.443),
    (0.851, 0.682, 0.380),
    (0.686, 0.286, 0.310),
    (0.353, 0.518, 0.235),
    (0.349, 0.412, 0.694),
    (0.600, 0.400, 0.200),
    (0.500, 0.300, 0.500),
    (0.200, 0.600, 0.600),
]


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text(x: float, y: float, text: str, size: int) -> str:
    escaped = _pdf_escape(text)
    return f"0 0 0 rg BT /F1 {size} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm ({escaped}) Tj ET"


def _pdf_line(x1: float, y1: float, x2: float, y2: float, width: float = 1.0, rgb: tuple[float, float, float] = (0, 0, 0)) -> str:
    r, g, b = rgb
    return f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S"


def _pdf_rect(x: float, y: float, w: float, h: float, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"{r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f"


def _pdf_circle(x: float, y: float, r: float, rgb: tuple[float, float, float]) -> str:
    # Approximate a small circle with Bezier curves.
    c = 0.5522847498 * r
    rr, gg, bb = rgb
    return (
        f"{rr:.3f} {gg:.3f} {bb:.3f} rg "
        f"{x+r:.1f} {y:.1f} m "
        f"{x+r:.1f} {y+c:.1f} {x+c:.1f} {y+r:.1f} {x:.1f} {y+r:.1f} c "
        f"{x-c:.1f} {y+r:.1f} {x-r:.1f} {y+c:.1f} {x-r:.1f} {y:.1f} c "
        f"{x-r:.1f} {y-c:.1f} {x-c:.1f} {y-r:.1f} {x:.1f} {y-r:.1f} c "
        f"{x+c:.1f} {y-r:.1f} {x+r:.1f} {y-c:.1f} {x+r:.1f} {y:.1f} c f"
    )


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


def load_reports(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("*_linear_probe_top*.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "concept": payload["concept"],
                "top_k_features": int(payload["top_k_features"]),
                "accuracy": float(payload["accuracy"]),
                "auc": float(payload["auc"]),
                "source_file": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No *_linear_probe_top*.json files found in {input_dir}")
    return rows


def save_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["concept", "top_k_features", "accuracy", "auc", "source_file"])
        writer.writeheader()
        writer.writerows(rows)


def plot_pdf(rows: list[dict], output_pdf: Path) -> None:
    grouped: dict[str, dict[int, float]] = {}
    for row in rows:
        grouped.setdefault(row["concept"], {})[row["top_k_features"]] = row["auc"]

    concepts = sorted(grouped)
    width = 1100
    height = 680
    margin_left = 105
    margin_right = 220
    margin_top = 85
    margin_bottom = 140
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    x0 = margin_left
    y0 = margin_bottom

    ops: list[str] = []
    ops.append(_pdf_text(width / 2 - 160, height - 48, "Top-k Features vs AUC", 24))
    ops.append(_pdf_line(x0, y0, x0, y0 + chart_h, 1.2))
    ops.append(_pdf_line(x0, y0, x0 + chart_w, y0, 1.2))
    ops.append(_pdf_text(28, y0 + chart_h / 2, "AUC", 20))
    ops.append(_pdf_text(x0 + chart_w / 2 - 75, 35, "Number of Top-k Features", 18))

    # Y ticks 0..1
    for tick in range(6):
        value = tick / 5
        y = y0 + value * chart_h
        ops.append(_pdf_line(x0 - 6, y, x0, y, 1.0))
        ops.append(_pdf_text(x0 - 42, y - 5, f"{value:.1f}", 16))
        if tick > 0:
            ops.append(_pdf_line(x0, y, x0 + chart_w, y, 0.8, (0.87, 0.87, 0.87)))

    # X ticks at categorical positions
    step = chart_w / (len(TARGET_KS) - 1)
    x_positions = {k: x0 + i * step for i, k in enumerate(TARGET_KS)}
    for k, x in x_positions.items():
        ops.append(_pdf_line(x, y0, x, y0 - 6, 1.0))
        label = str(k)
        offset = 6 * len(label)
        ops.append(_pdf_text(x - offset / 2, y0 - 28, label, 16))

    # Draw concept lines
    for idx, concept in enumerate(concepts):
        color = PALETTE[idx % len(PALETTE)]
        concept_points = [(k, grouped[concept][k]) for k in TARGET_KS if k in grouped[concept]]
        for (k1, auc1), (k2, auc2) in zip(concept_points[:-1], concept_points[1:]):
            ops.append(_pdf_line(x_positions[k1], y0 + auc1 * chart_h, x_positions[k2], y0 + auc2 * chart_h, 2.0, color))
        for k, auc in concept_points:
            ops.append(_pdf_circle(x_positions[k], y0 + auc * chart_h, 3.8, color))

    # Legend
    legend_x = width - 185
    legend_y = height - 95
    for idx, concept in enumerate(concepts):
        color = PALETTE[idx % len(PALETTE)]
        y = legend_y - idx * 24
        ops.append(_pdf_rect(legend_x, y, 14, 14, color))
        ops.append(_pdf_text(legend_x + 22, y + 1, concept, 14))

    _write_single_page_pdf(ops, width, height, output_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize top-k linear probe sweep results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_reports(args.input_dir)
    rows = sorted(rows, key=lambda item: (item["concept"], item["top_k_features"]))

    output_csv = args.output_dir / "linear_probe_topk_summary.csv"
    output_pdf = args.output_dir / "linear_probe_topk_auc.pdf"
    save_csv(rows, output_csv)
    plot_pdf(rows, output_pdf)

    print(f"Loaded reports: {len(rows)}")
    print(f"Saved CSV to: {output_csv}")
    print(f"Saved figure to: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
