"""Create fixed CelebA-HQ test manifests whose target attribute is absent."""

from __future__ import annotations

import argparse
from pathlib import Path


TARGETS = {
    "smiling": ("Smiling", -1),
    "young": ("Young", -1),
    "big_nose": ("Big_Nose", -1),
    "beard": ("No_Beard", 1),
    "blond_hair": ("Blond_Hair", -1),
    "eyeglasses": ("Eyeglasses", -1),
    "male": ("Male", -1),
    "mustache": ("Mustache", -1),
    "no_beard": ("No_Beard", -1),
    "bangs": ("Bangs", -1),
    "heavy_makeup": ("Heavy_Makeup", -1),
    "chubby": ("Chubby", -1),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attributes", type=Path, default=Path("../CelebA-HQ/list_attr_celeba_shift.txt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("flow_uspace_rebuttal/manifests")
    )
    parser.add_argument("--concepts", nargs="+", default=list(TARGETS))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--minimum-id", type=int, default=10000)
    args = parser.parse_args()

    with args.attributes.open() as handle:
        expected = int(handle.readline())
        names = handle.readline().split()
        rows = []
        for line in handle:
            fields = line.split()
            rows.append((fields[0], [int(value) for value in fields[1:]]))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for concept in args.concepts:
        attribute, absent_value = TARGETS[concept]
        index = names.index(attribute)
        selected = [
            image_id
            for image_id, values in rows
            if int(Path(image_id).stem) >= args.minimum_id and values[index] == absent_value
        ][: args.count]
        if len(selected) != args.count:
            raise ValueError(f"{concept}: only {len(selected)} eligible images")
        path = args.output_dir / f"{concept}_negative32.txt"
        path.write_text("\n".join(selected) + "\n")
        print(concept, attribute, absent_value, path, selected[0], selected[-1])


if __name__ == "__main__":
    main()
