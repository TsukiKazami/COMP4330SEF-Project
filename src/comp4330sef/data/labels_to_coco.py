from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def load_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 80:
        raise SystemExit(f"{path}: expected 80 class names, got {len(names)}")
    return names


def parse_label_file(path: Path, width: int, height: int, classes: int) -> tuple[list[list[float]], list[str]]:
    boxes: list[list[float]] = []
    problems: list[str] = []
    if not path.exists():
        return boxes, problems
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 5:
            problems.append(f"{path.name}:{number} has {len(tokens)} tokens")
            continue
        try:
            label = int(tokens[0])
            cx, cy, bw, bh = (float(value) for value in tokens[1:])
        except ValueError:
            problems.append(f"{path.name}:{number} is not numeric")
            continue
        if not 0 <= label < classes:
            problems.append(f"{path.name}:{number} class {label} out of range")
            continue
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            problems.append(f"{path.name}:{number} degenerate box after clipping")
            continue
        boxes.append([label, round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)])
    return boxes, problems


def build(selection: pd.DataFrame, images_root: Path, labels_dir: Path, names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    image_records: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    problems: list[str] = []
    annotation_id = 1
    empty: list[int] = []

    for row in selection.itertuples(index=False):
        image_id = int(row.image_id)
        path = images_root / str(row.file_name)
        if not path.exists():
            raise SystemExit(f"image missing: {path}")
        with Image.open(path) as handle:
            width, height = handle.size
        image_records.append(
            {"id": image_id, "file_name": str(row.file_name), "width": int(width), "height": int(height)}
        )
        boxes, issues = parse_label_file(labels_dir / f"{image_id:04d}.txt", width, height, len(names))
        problems.extend(issues)
        if not boxes:
            empty.append(image_id)
        for label, x, y, w, h in boxes:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label + 1,
                    "bbox": [x, y, w, h],
                    "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    categories = [
        {"id": index + 1, "name": name, "supercategory": "coco"} for index, name in enumerate(names)
    ]
    coco = {
        "info": {"description": "Hong Kong street-scene test set", "version": "1.0"},
        "licenses": [],
        "images": image_records,
        "annotations": annotations,
        "categories": categories,
    }
    counts = {int(a["category_id"]): 0 for a in annotations}
    for annotation in annotations:
        counts[int(annotation["category_id"])] += 1
    stats = {
        "images": len(image_records),
        "annotations": len(annotations),
        "empty_images": empty,
        "problems": problems,
        "classes_present": len(counts),
        "per_class": {
            names[category_id - 1]: count for category_id, count in sorted(counts.items())
        },
        "duplicate_annotation_ids": len(annotations) - len({a["id"] for a in annotations}),
    }
    return coco, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert YOLO labels of the frozen test set to COCO json")
    parser.add_argument("--set", default="data/annotations/test_set.csv")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--labels", default="data/labels")
    parser.add_argument("--classes", default="configs/annot_labels.txt")
    parser.add_argument("--out", default="data/annotations/hk_test_coco.json")
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    names = load_names(resolve(args.classes))
    selection = pd.read_csv(resolve(args.set))
    if selection["image_id"].duplicated().any():
        raise SystemExit("duplicate image_id in the selection")

    coco, stats = build(selection, resolve(args.images), resolve(args.labels), names)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    print(f"images      : {stats['images']}")
    print(f"annotations : {stats['annotations']}")
    print(f"classes     : {stats['classes_present']} of 80 present")
    print(f"empty images: {len(stats['empty_images'])} {stats['empty_images'][:10]}")
    print(f"problems    : {len(stats['problems'])}")
    for problem in stats["problems"][:10]:
        print(f"  {problem}")
    print("per class   :")
    for name, count in stats["per_class"].items():
        print(f"  {name:<18}{count:>5}")
    print(f"\nwrote {out}")

    if stats["problems"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
