from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "valid", "test")

SOURCE_TO_COCO = {
    "car": 3,
    "taxi": 3,
    "motorcycle": 4,
    "bus": 6,
    "mini bus": 6,
    "truck": 8,
}

MODEL_CLASSES = (3, 4, 6, 8)
CLASS_NAMES = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
MIN_BOX_SIDE = 0.5


def group_of(file_name: str) -> str:
    """Scene group: video uuid for frames, camera id for stills, stem otherwise."""
    stem = file_name.split(".rf.")[0]
    head = stem.split("_")[0]
    prefix, separator, rest = head.partition("-")
    if separator and len(prefix) == 8 and all(c in "0123456789abcdef" for c in prefix.lower()):
        return f"{prefix}-{rest[:6]}"
    if head[:2] in ("SK", "RI") and head[2:6].isdigit():
        return head[:6]
    return head


def load_source(root: Path) -> dict:
    payload = {}
    for split in SPLITS:
        path = root / split / "_annotations.coco.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        payload[split] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def collect(payload: dict, root: Path):
    """Keep images that still carry at least one valid vehicle class."""
    records = []
    dropped_images = 0
    dropped_boxes = 0
    for split, data in payload.items():
        names = {int(c["id"]): str(c["name"]) for c in data["categories"]}
        by_image = collections.defaultdict(list)
        for ann in data["annotations"]:
            by_image[int(ann["image_id"])].append(ann)
        for image in data["images"]:
            kept = []
            for ann in by_image.get(int(image["id"]), []):
                coco_id = SOURCE_TO_COCO.get(names[int(ann["category_id"])])
                if coco_id is None:
                    dropped_boxes += 1
                    continue
                kept.append((coco_id, ann["bbox"]))
            if not kept:
                dropped_images += 1
                continue
            source_path = root / split / image["file_name"]
            if not source_path.exists():
                raise SystemExit(f"missing image {source_path}")
            records.append(
                {
                    "file_name": image["file_name"],
                    "source_path": source_path,
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "annotations": kept,
                }
            )
    return records, dropped_images, dropped_boxes


def split_groups(records: list, ratios) -> tuple:
    """Deterministic greedy split: largest groups first, into the greatest deficit."""
    sizes = collections.Counter(group_of(rec["file_name"]) for rec in records)
    total = len(records)
    targets = {split: ratios[index] * total for index, split in enumerate(SPLITS)}
    filled = {split: 0 for split in SPLITS}
    assignment = {}
    for group, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        split = max(SPLITS, key=lambda name: targets[name] - filled[name])
        assignment[group] = split
        filled[split] += size
    return assignment, sizes


def clip_bbox(bbox, width: int, height: int):
    x, y, w, h = (float(value) for value in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        return None
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def build_split(records: list, split: str, images_root: Path, labels_root: Path, write: bool) -> dict:
    image_dir = images_root / split
    label_dir = labels_root / split
    if write:
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    per_class = collections.Counter()
    dropped_boxes = 0
    empty_images = 0
    annotation_id = 1

    for image_id, record in enumerate(sorted(records, key=lambda item: item["file_name"]), start=1):
        width, height = record["width"], record["height"]
        images.append(
            {
                "id": image_id,
                "file_name": record["file_name"],
                "width": width,
                "height": height,
            }
        )
        lines = []
        for coco_id, bbox in record["annotations"]:
            clipped = clip_bbox(bbox, width, height)
            if clipped is None:
                dropped_boxes += 1
                continue
            x, y, w, h = clipped
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": coco_id,
                    "bbox": clipped,
                    "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            per_class[coco_id] += 1
            lines.append(
                "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    MODEL_CLASSES.index(coco_id),
                    (x + w / 2) / width,
                    (y + h / 2) / height,
                    w / width,
                    h / height,
                )
            )

        if not lines:
            empty_images += 1
        if write:
            shutil.copy2(record["source_path"], image_dir / record["file_name"])
            label_path = label_dir / (Path(record["file_name"]).stem + ".txt")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "images": images,
        "annotations": annotations,
        "per_class": per_class,
        "dropped_boxes": dropped_boxes,
        "empty_images": empty_images,
    }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_coco(path: Path, images: list, annotations: list) -> None:
    document = {
        "info": {
            "description": "Hong Kong street scenes, 4 COCO classes",
            "version": "1.0",
            "source": "Roboflow hong kong vehicle detection v3 (COCO JSON), MIT",
        },
        "licenses": [
            {
                "id": 1,
                "name": "MIT",
                "url": "https://universe.roboflow.com/roy-b2g9i/hong-kong-vehicle-detection-8zkg0",
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": coco_id, "name": CLASS_NAMES[coco_id], "supercategory": "vehicle"}
            for coco_id in MODEL_CLASSES
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def write_yaml(path: Path, out_root: Path) -> None:
    lines = [
        "# Generated by scripts/build_hk_dataset.py; do not edit by hand.",
        "# check_det_dataset resolves a relative 'path' against DATASETS_DIR,",
        "# so the absolute path below is intentional (same as datasets/coco8.yaml).",
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        "names:",
    ]
    for index, coco_id in enumerate(MODEL_CLASSES):
        lines.append(f"  {index}: {CLASS_NAMES[coco_id]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the HK evaluation set")
    parser.add_argument("--src", default="datasets/hk_vehicle", help="Roboflow export root")
    parser.add_argument("--out", default="data/hk", help="image and label root")
    parser.add_argument("--annotations", default="data/annotations", help="COCO output directory")
    parser.add_argument("--config", default="configs/hk4.yaml", help="Ultralytics dataset yaml")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.10, 0.20))
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite a non-empty output tree")
    args = parser.parse_args()

    source_root = Path(args.src)
    out_root = Path(args.out)
    annotation_dir = Path(args.annotations)
    config_path = Path(args.config)
    ratios = list(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1, got {sum(ratios)}")

    payload = load_source(source_root)
    records, dropped_images, dropped_boxes = collect(payload, source_root)
    if not records:
        raise SystemExit("no usable image left")

    counts = collections.Counter(record["file_name"] for record in records)
    duplicates = [name for name, count in counts.items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate file names across splits: {duplicates[:5]}")

    assignment, group_sizes = split_groups(records, ratios)
    by_split = {split: [] for split in SPLITS}
    for record in records:
        by_split[assignment[group_of(record["file_name"])]].append(record)

    multi = sum(1 for size in group_sizes.values() if size > 1)
    print(f"source        : {source_root.resolve()}")
    print(f"kept          : {len(records)} images / {sum(len(r['annotations']) for r in records)} boxes")
    print(f"dropped       : {dropped_images} images / {dropped_boxes} boxes with a corrupted class")
    print(f"groups        : {len(group_sizes)} ({multi} multi-image, largest {max(group_sizes.values())})")
    print(f"split rule    : deterministic greedy by group, ratios {ratios}")

    if out_root.exists() and any(out_root.iterdir()) and not args.dry_run and not args.force:
        raise SystemExit(f"{out_root} is not empty; pass --force to overwrite")

    results = {}
    for split in SPLITS:
        built = build_split(
            by_split[split], split, out_root / "images", out_root / "labels", write=not args.dry_run
        )
        results[split] = built
        if built["empty_images"]:
            raise SystemExit(f"{split}: {built['empty_images']} images lost all boxes during clipping")
        per_class = {CLASS_NAMES[cid]: built["per_class"].get(cid, 0) for cid in MODEL_CLASSES}
        dropped = f", dropped={built['dropped_boxes']}" if built["dropped_boxes"] else ""
        print(
            f"{split:<13} : {len(built['images']):>3} images, {len(built['annotations']):>5} boxes "
            f"({', '.join(f'{name}={count}' for name, count in per_class.items())}){dropped}"
        )

    total_images = sum(len(results[split]["images"]) for split in SPLITS)
    total_boxes = sum(len(results[split]["annotations"]) for split in SPLITS)
    print(f"total         : {total_images} images, {total_boxes} boxes")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_root.resolve().as_posix(),
        "source_export": "Roboflow hong kong vehicle detection v3, COCO JSON, MIT",
        "class_map": SOURCE_TO_COCO,
        "model_classes": list(MODEL_CLASSES),
        "split_rule": "deterministic greedy by scene group",
        "ratios": {split: ratios[index] for index, split in enumerate(SPLITS)},
        "dropped": {"images": dropped_images, "annotations": dropped_boxes},
        "splits": {},
    }
    for split in SPLITS:
        path = annotation_dir / f"hk_{split}_coco.json"
        write_coco(path, results[split]["images"], results[split]["annotations"])
        manifest["splits"][split] = {
            "images": len(results[split]["images"]),
            "annotations": len(results[split]["annotations"]),
            "per_class": {CLASS_NAMES[cid]: results[split]["per_class"].get(cid, 0) for cid in MODEL_CLASSES},
            "sha256": sha256_of(path),
        }
    write_yaml(config_path, out_root)

    manifest_path = annotation_dir / "hk_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nwrote         : {out_root}/images, {out_root}/labels")
    print(f"                {annotation_dir}/hk_{{train,valid,test}}_coco.json")
    print(f"                {manifest_path}")
    print(f"                {config_path}")
    print(f"test sha256   : {manifest['splits']['test']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
