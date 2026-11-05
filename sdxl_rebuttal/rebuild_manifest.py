from __future__ import annotations

import argparse
import re
from pathlib import Path

from .utils import load_config, test_ids, write_json


RANGE_RE = re.compile(r"steps(\d+)-(\d+)$")


def parse_strength(slug: str) -> float:
    if not slug.startswith("a"):
        raise ValueError(f"Invalid strength directory: {slug}")
    return float(slug[1:].replace("m", "-").replace("p", "."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--result-dir",
        help="Optional result directory override (useful for isolated ablations)",
    )
    parser.add_argument("--concepts", help="Optional comma-separated concept subset")
    parser.add_argument("--test-count", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.result_dir:
        cfg["result_dir"] = args.result_dir
    if args.test_count is not None:
        cfg["test_count"] = args.test_count
    result_dir = Path(cfg["result_dir"])
    requested = set(args.concepts.split(",")) if args.concepts else None
    expected_ids = set(test_ids(cfg))
    records: list[dict] = []
    for path in sorted((result_dir / "edits").glob("*/*/steps*-*/*/a*/*.jpg")):
        relative = path.relative_to(result_dir / "edits")
        concept, method, range_slug, topk_slug, strength_slug, image_id = relative.parts
        if requested is not None and concept not in requested:
            continue
        if image_id not in expected_ids:
            continue
        match = RANGE_RE.fullmatch(range_slug)
        if match is None:
            raise ValueError(f"Invalid range directory: {range_slug}")
        strength = parse_strength(strength_slug)
        records.append(
            {
                "image_id": image_id,
                "concept": concept,
                "method": method,
                "range": [int(match.group(1)), int(match.group(2))],
                "topk": None if topk_slug == "dense" else int(topk_slug.removeprefix("topk")),
                "strength": strength,
                "path": str(path.resolve()),
                "status": "recovered_existing",
            }
        )
    output = result_dir / "edit_manifest.json"
    write_json(records, output)
    print(f"[saved] {output} ({len(records)} rows)")


if __name__ == "__main__":
    main()
