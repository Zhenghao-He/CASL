#!/usr/bin/env python3
"""
Aggregate concept linear-probe reports into a CSV and a PDF bar chart.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path("/p/realai/zhenghao/LoRA_SAE")
DEFAULT_INPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_linear_probe"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_linear_probe_summary"


def load_reports(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("*_linear_probe.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "concept": payload["concept"],
                "accuracy": float(payload["accuracy"]),
                "auc": float(payload["auc"]),
                "num_samples": int(payload["num_samples"]),
                "num_positive": int(payload["num_positive"]),
                "num_negative": int(payload["num_negative"]),
                "feature_dim": int(payload["feature_shape"][1]),
                "source_file": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No *_linear_probe.json files found in {input_dir}")
    return rows


def save_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "concept",
        "accuracy",
        "auc",
        "num_samples",
        "num_positive",
        "num_negative",
        "feature_dim",
        "source_file",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text(x: float, y: float, text: str, size: int) -> str:
    escaped = _pdf_escape(text)
    return f"0 0 0 rg BT /F1 {size} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm ({escaped}) Tj ET"


def _pdf_line(x1: float, y1: float, x2: float, y2: float, width: float = 1.0) -> str:
    return f"{width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S"


def _pdf_rect(x: float, y: float, w: float, h: float, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"{r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f"


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
    objects.append(
        f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream"
    )

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("latin-1")
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.write_bytes(pdf)


def plot_metrics_pdf(rows: list[dict], output_pdf: Path) -> None:
    concepts = [row["concept"] for row in rows]
    acc = [row["accuracy"] for row in rows]
    auc = [row["auc"] for row in rows]
    width = 1080
    height = 660
    margin_left = 100
    margin_right = 30
    margin_top = 80
    margin_bottom = 190
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    n = max(1, len(concepts))
    group_w = chart_w / n
    bar_w = group_w * 0.28

    ops: list[str] = []

    # Axes
    x0 = margin_left
    y0 = margin_bottom
    chart_top = y0 + chart_h
    ops.append(_pdf_text(width / 2 - 225, height - 48, "DiT SAE Linear Probe Performance by Concept", 24))
    ops.append(_pdf_line(x0, y0, x0, chart_top, 1.2))
    ops.append(_pdf_line(x0, y0, x0 + chart_w, y0, 1.2))
    ops.append(_pdf_text(28, y0 + chart_h / 2, "Score", 20))

    # Y ticks
    for tick in range(6):
        value = tick / 5
        y = y0 + value * chart_h
        ops.append(_pdf_line(x0 - 6, y, x0, y, 1.0))
        ops.append(_pdf_text(x0 - 42, y - 5, f"{value:.1f}", 16))
        if tick > 0:
            ops.append("0.867 0.867 0.867 RG")
            ops.append(_pdf_line(x0, y, x0 + chart_w, y, 0.8))
            ops.append("0 0 0 RG")

    # Bars
    for i, concept in enumerate(concepts):
        group_x = x0 + i * group_w
        acc_h = acc[i] * chart_h
        auc_h = auc[i] * chart_h
        acc_x = group_x + group_w * 0.18
        auc_x = acc_x + bar_w + group_w * 0.08
        acc_y = y0
        auc_y = y0
        ops.append(_pdf_rect(acc_x, acc_y, bar_w, acc_h, (60 / 255, 110 / 255, 113 / 255)))
        ops.append(_pdf_rect(auc_x, auc_y, bar_w, auc_h, (217 / 255, 174 / 255, 97 / 255)))
        center_x = group_x + group_w / 2
        # PDF text rotation: 0.9063 0.4226 -0.4226 0.9063
        escaped = _pdf_escape(concept)
        ops.append(
            f"0 0 0 rg BT /F1 16 Tf 0.9063 0.4226 -0.4226 0.9063 {center_x - 22:.1f} {y0 - 62:.1f} Tm ({escaped}) Tj ET"
        )

    # Legend
    legend_x = width - 210
    legend_y = height - 92
    ops.append(_pdf_rect(legend_x, legend_y, 16, 16, (60 / 255, 110 / 255, 113 / 255)))
    ops.append(_pdf_text(legend_x + 24, legend_y + 2, "Accuracy", 18))
    ops.append(_pdf_rect(legend_x, legend_y - 26, 16, 16, (217 / 255, 174 / 255, 97 / 255)))
    ops.append(_pdf_text(legend_x + 24, legend_y - 24, "AUC", 18))

    _write_single_page_pdf(ops, width, height, output_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize linear probe JSON reports.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_reports(args.input_dir)
    rows = sorted(rows, key=lambda item: item["concept"])

    output_csv = args.output_dir / "linear_probe_summary.csv"
    output_pdf = args.output_dir / "linear_probe_accuracy_auc.pdf"

    save_csv(rows, output_csv)
    plot_metrics_pdf(rows, output_pdf)

    best_auc = max(rows, key=lambda item: item["auc"])
    best_acc = max(rows, key=lambda item: item["accuracy"])

    print(f"Loaded reports: {len(rows)}")
    print(f"Saved CSV to: {output_csv}")
    print(f"Saved figure to: {output_pdf}")
    print(f"Best AUC: {best_auc['concept']} ({best_auc['auc']:.4f})")
    print(f"Best Accuracy: {best_acc['concept']} ({best_acc['accuracy']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
