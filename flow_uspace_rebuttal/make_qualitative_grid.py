from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/flow_uspace_large"
PAIRS = {
    "smiling": {"dense": (None, 4.0), "casl": (16.0, -0.43)},
    "no_beard": {"dense": (None, -4.0), "casl": (8.0, -0.075)},
    "blond_hair": {"dense": (None, 4.0), "casl": (8.0, 4.0)},
}


def main():
    metrics = pd.concat(
        [
            pd.read_csv(RESULT / "metrics32_interp_per_image.csv"),
            pd.read_csv(RESULT / "metrics32_curve_per_image.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates(["concept", "method", "topk", "strength", "image_id"])
    tile, label = 256, 28
    columns = ["source", "inversion", "dense", "casl"]
    rows = []
    for concept, methods in PAIRS.items():
        dense = metrics[
            (metrics.concept == concept)
            & (metrics.method == "dense")
            & (metrics.strength == methods["dense"][1])
        ].sort_values("image_id")
        casl = metrics[
            (metrics.concept == concept)
            & (metrics.method == "casl")
            & (metrics.topk == methods["casl"][0])
            & (metrics.strength == methods["casl"][1])
        ].set_index("image_id")
        for record in dense.head(4).to_dict("records"):
            image_id = record["image_id"]
            if image_id not in casl.index:
                continue
            rows.append(
                (
                    concept,
                    [
                        RESULT / "source" / image_id,
                        RESULT / "inversion" / image_id,
                        ROOT / record["path"],
                        ROOT / casl.loc[image_id, "path"],
                    ],
                )
            )
    canvas = Image.new("RGB", (len(columns) * tile, len(rows) * (tile + label) + label), "white")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(columns):
        draw.text((column * tile + 8, 7), name, fill="black")
    for row, (concept, paths) in enumerate(rows):
        y = label + row * (tile + label)
        for column, path in enumerate(paths):
            image = Image.open(path).convert("RGB").resize((tile, tile))
            canvas.paste(image, (column * tile, y))
        draw.text((8, y + tile + 6), concept, fill="black")
    canvas.save(RESULT / "qualitative_matched32.jpg", quality=92)


if __name__ == "__main__":
    main()
