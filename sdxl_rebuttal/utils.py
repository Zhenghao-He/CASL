from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image


PACKAGE_DIR = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    for key in ("test_manifest", "result_dir"):
        value = Path(cfg[key])
        if not value.is_absolute():
            cfg[key] = str((PACKAGE_DIR / value).resolve())
    cfg["work_dir"] = str(Path(cfg["work_dir"]).resolve())
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    Path(cfg["work_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["result_dir"]).mkdir(parents=True, exist_ok=True)


def test_ids(cfg: dict[str, Any]) -> list[str]:
    with Path(cfg["test_manifest"]).open("r", encoding="utf-8") as handle:
        values = [line.strip() for line in handle if line.strip()]
    return values[: int(cfg["test_count"])]


def train_ids(cfg: dict[str, Any], limit: int | None = None) -> list[str]:
    start = int(cfg["train_ids"]["start"])
    count = int(cfg["train_ids"]["count"] if limit is None else limit)
    return [f"{index:05d}.jpg" for index in range(start, start + count)]


def image_path(cfg: dict[str, Any], image_id: str) -> Path:
    path = Path(cfg["data_dir"]) / image_id
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_square_image(path: str | Path, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_pil(image_m11: torch.Tensor) -> Image.Image:
    image = image_m11.detach().float().cpu().clamp(-1, 1)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected batch size one, got {tuple(image.shape)}")
        image = image[0]
    array = ((image.permute(1, 2, 0) + 1.0) * 127.5).round()
    return Image.fromarray(array.clamp(0, 255).byte().numpy())


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def append_event(cfg: dict[str, Any], event: str, **payload: Any) -> None:
    record = {"time": time.time(), "event": event, **payload}
    path = Path(cfg["result_dir"]) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(part) for part in value.split(",") if part.strip()]


def parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]
