from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics.data.utils import check_det_dataset

from comp4330sef.config import REPO_ROOT
from comp4330sef.data.yolo_to_coco import labels_dir_for, names_to_dict, yolo_split_to_coco

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class EvalSet:
    name: str
    images: list[Path]
    image_ids: list[int]
    gt: dict[str, Any]
    source: str

    def __post_init__(self) -> None:
        if len(self.images) != len(self.image_ids):
            raise ValueError("images and image_ids must be aligned")
        if not self.images:
            raise ValueError(f"{self.name}: empty evaluation set")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def from_yolo_yaml(
    data_name: str = "coco8.yaml",
    split: str = "val",
    name: str | None = None,
) -> EvalSet:
    data = check_det_dataset(data_name)
    names = names_to_dict(data["names"])

    entry = data.get(split)
    if entry is None:
        raise SystemExit(f"{data_name}: no {split!r} split")
    entries = entry if isinstance(entry, (list, tuple)) else [entry]

    images: list[Path] = []
    for item in entries:
        path = Path(item)
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            images.append(path)
    if not images:
        raise SystemExit(f"{data_name}: no images for split {split!r}")

    gt = yolo_split_to_coco(images, labels_dir_for(images[0].parent), names, split=split)
    return EvalSet(
        name=name or f"{Path(data_name).stem}_{split}",
        images=sorted(images),
        image_ids=[int(record["id"]) for record in gt["images"]],
        gt=gt,
        source=str(data_name),
    )


def from_coco(
    gt_path: str | Path,
    images_root: str | Path | None = None,
    name: str | None = None,
) -> EvalSet:
    path = _resolve(gt_path)
    gt = json.loads(path.read_text(encoding="utf-8"))

    missing = {"images", "annotations", "categories"} - set(gt)
    if missing:
        raise SystemExit(f"{path}: not a COCO annotation file (missing {sorted(missing)})")

    root = _resolve(images_root) if images_root else path.parent
    records = sorted(gt["images"], key=lambda record: int(record["id"]))

    images: list[Path] = []
    for record in records:
        candidate = root / record["file_name"]
        if not candidate.exists():
            raise SystemExit(
                f"missing image {candidate} — pass --images with the root that holds file_name"
            )
        images.append(candidate)

    return EvalSet(
        name=name or path.stem,
        images=images,
        image_ids=[int(record["id"]) for record in records],
        gt=gt,
        source=str(path),
    )