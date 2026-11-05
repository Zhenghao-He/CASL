from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .utils import load_config, test_ids


def strength_slug(value: float) -> str:
    return "a" + f"{value:.5g}".replace("-", "m").replace(".", "p")


def edit_path(
    root: Path, concept: str, method: str, topk: int | None, strength: float, image_id: str
) -> Path:
    space = "dense" if topk is None else f"topk{topk}"
    return (
        root
        / "edits"
        / concept
        / method
        / "steps05-25"
        / space
        / strength_slug(strength)
        / image_id
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--casl-method", default="casl")
    parser.add_argument("--casl-topk", type=int, required=True)
    parser.add_argument("--casl-strength", type=float, required=True)
    parser.add_argument("--baseline-strength", type=float, required=True)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(args.result_dir)
    ids = test_ids({**cfg, "test_count": args.images})
    labels = ["source", "inversion", "Asyrp-style", "external (N/A)", "CASL"]
    tile, header = 256, 36
    canvas = Image.new("RGB", (tile * len(labels), header + tile * len(ids)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * tile + 8, 11), label, fill="black")

    for row, image_id in enumerate(ids):
        paths: list[Path | None] = [
            Path(cfg["data_dir"]) / image_id,
            root / "reconstruction" / image_id,
            edit_path(root, args.concept, "asyrp_cnn", None, args.baseline_strength, image_id),
            None,
            edit_path(
                root,
                args.concept,
                args.casl_method,
                args.casl_topk,
                args.casl_strength,
                image_id,
            ),
        ]
        for column, path in enumerate(paths):
            if path is None:
                image = Image.new("RGB", (tile, tile), (232, 232, 232))
                ImageDraw.Draw(image).text((75, 120), "not included", fill=(70, 70, 70))
            else:
                image = Image.open(path).convert("RGB").resize(
                    (tile, tile), Image.Resampling.LANCZOS
                )
            canvas.paste(image, (column * tile, header + row * tile))
        draw.text(
            (8, header + row * tile + 8),
            image_id,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
