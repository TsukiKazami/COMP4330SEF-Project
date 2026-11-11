"""YOLO-format labels -> COCO annotation json"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from comp4330sef.eval.results import image_id_from_stem


def names_to_dict(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def labels_dir_for(images_dir: Path) -> Path:
    parts = list(Path(images_dir).parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return Path(images_dir).parent


def yolo_split_to_coco(
    images: Sequence[Path],
    labels_dir: Path,
    names: Mapping[int, str] | Sequence[str],
    *,
    split: str = "val",
) -> dict[str, Any]:
    name_map = names_to_dict(names)
    categories = [
        {"id": index + 1, "name": name_map[index], "supercategory": "coco"}
        for index in sorted(name_map)
    ]

    image_records: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_path in sorted(images):
        image_id = image_id_from_stem(image_path)
        if image_id is None:
            raise ValueError(
                f"{image_path.name}: numeric stem required so that image ids are stable"
            )
        with Image.open(image_path) as handle:
            width, height = handle.size

        image_records.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(width),
                "height": int(height),
            }
        )

        label_path = Path(labels_dir) / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_index = int(parts[0])
            cx, cy, norm_w, norm_h = (float(value) for value in parts[1:5])
            box_w = norm_w * width
            box_h = norm_h * height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_index + 1,
                    "bbox": [
                        round((cx - norm_w / 2) * width, 2),
                        round((cy - norm_h / 2) * height, 2),
                        round(box_w, 2),
                        round(box_h, 2),
                    ],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "info": {"description": f"{split} split", "split": split},
        "licenses": [],
        "images": image_records,
        "annotations": annotations,
        "categories": categories,
    }