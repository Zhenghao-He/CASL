#!/usr/bin/env python3
"""
Plot SAE training loss curves from a saved history JSON into a PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path("/p/realai/zhenghao/LoRA_SAE")
DEFAULT_HISTORY_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_sae"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_sae_curves"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text(x: float, y: float, text: str, size: int) -> str:
    return f"0 0 0 rg BT /F1 {size} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm ({_pdf_escape(text)}) Tj ET"


def _pdf_line(x1: float, y1: float, x2: float, y2: float, width: float = 1.0, rgb: tuple[float, float, float] = (0, 0, 0)) -> str:
    r, g, b = rgb
    return f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S"


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


def load_history(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty history list in {path}")
    return data


def plot_history_pdf(history: list[dict], output_pdf: Path, title: str) -> None:
    epochs = [int(item["epoch"]) for item in history]
    loss = [float(item["loss"]) for item in history]
    recon = [float(item["recon"]) for item in history]

    width = 1100
    height = 680
    margin_left = 110
    margin_right = 70
    margin_top = 90
    margin_bottom = 140
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    x0 = margin_left
    y0 = margin_bottom

    max_epoch = max(epochs)
    min_epoch = min(epochs)
    max_y = max(max(loss), max(recon))
    min_y = min(min(loss), min(recon))
    if max_y == min_y:
        max_y = min_y + 1.0
    y_pad = (max_y - min_y) * 0.05
    y_min = max(0.0, min_y - y_pad)
    y_max = max_y + y_pad

    def map_x(epoch: int) -> float:
        if max_epoch == min_epoch:
            return x0 + chart_w / 2
        return x0 + (epoch - min_epoch) / (max_epoch - min_epoch) * chart_w

    def map_y(value: float) -> float:
        return y0 + (value - y_min) / (y_max - y_min) * chart_h

    ops: list[str] = []
    ops.append(_pdf_text(width / 2 - 180, height - 50, title, 24))
    ops.append(_pdf_line(x0, y0, x0, y0 + chart_h, 1.2))
    ops.append(_pdf_line(x0, y0, x0 + chart_w, y0, 1.2))
    ops.append(_pdf_text(35, y0 + chart_h / 2, "Loss", 20))
    ops.append(_pdf_text(x0 + chart_w / 2 - 35, 38, "Epoch", 20))

    # Grid and ticks
    for tick in range(6):
        t = tick / 5
        y = y0 + t * chart_h
        val = y_min + t * (y_max - y_min)
        ops.append(_pdf_line(x0 - 6, y, x0, y, 1.0))
        ops.append(_pdf_text(x0 - 70, y - 5, f"{val:.2f}", 15))
        if tick > 0:
            ops.append(_pdf_line(x0, y, x0 + chart_w, y, 0.8, (0.87, 0.87, 0.87)))

    for tick in range(6):
        t = tick / 5
        x = x0 + t * chart_w
        ep = round(min_epoch + t * (max_epoch - min_epoch))
        ops.append(_pdf_line(x, y0, x, y0 - 6, 1.0))
        ops.append(_pdf_text(x - 14, y0 - 30, str(ep), 15))

    # Loss curve
    for i in range(len(history) - 1):
        ops.append(_pdf_line(map_x(epochs[i]), map_y(loss[i]), map_x(epochs[i + 1]), map_y(loss[i + 1]), 2.0, (0.85, 0.68, 0.38)))
        ops.append(_pdf_line(map_x(epochs[i]), map_y(recon[i]), map_x(epochs[i + 1]), map_y(recon[i + 1]), 2.0, (0.24, 0.43, 0.44)))

    # Legend
    legend_x = width - 190
    legend_y = height - 100
    ops.append(_pdf_line(legend_x, legend_y + 6, legend_x + 20, legend_y + 6, 3.0, (0.85, 0.68, 0.38)))
    ops.append(_pdf_text(legend_x + 28, legend_y, "Total loss", 17))
    ops.append(_pdf_line(legend_x, legend_y - 24 + 6, legend_x + 20, legend_y - 24 + 6, 3.0, (0.24, 0.43, 0.44)))
    ops.append(_pdf_text(legend_x + 28, legend_y - 24, "Recon loss", 17))

    _write_single_page_pdf(ops, width, height, output_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAE training curve from history JSON.")
    parser.add_argument("--history-file", type=Path, default=None)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-epoch", type=int, default=0, help="If > 0, only plot epochs <= max-epoch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.history_file is not None:
        history_file = args.history_file
    else:
        candidates = sorted(args.history_dir.glob("*_history.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No *_history.json found in {args.history_dir}. Existing older runs did not save per-epoch history."
            )
        history_file = candidates[-1]

    history = load_history(history_file)
    if args.max_epoch > 0:
        history = [item for item in history if int(item["epoch"]) <= args.max_epoch]
        if not history:
            raise ValueError(f"No history entries found with epoch <= {args.max_epoch}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{args.max_epoch}" if args.max_epoch > 0 else ""
    output_pdf = args.output_dir / f"{history_file.stem}{suffix}.pdf"
    title = f"SAE Training Convergence: {history_file.stem}{suffix}"
    plot_history_pdf(history, output_pdf, title)

    print(f"Loaded history: {history_file}")
    print(f"Epochs: {len(history)}")
    print(f"Saved figure to: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
