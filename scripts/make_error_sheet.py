"""Pick 15 error instances, crop them, and write the manual annotation form.

RQ3 asks for a small manual review of the dominant error cases. This script
does the mechanical part: it selects the largest instances of each error type
(round-robin over the three checkpoints), crops them with context, draws the
detection box and the ground-truth boxes, and writes a CSV with two empty
columns for the human label (`occlusion_level`, `text_density`).

    uv run python scripts\\make_error_sheet.py
    uv run python scripts\\make_error_sheet.py --per-type 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path.cwd()
ORDER = ("Miss", "Cls", "Bkg", "Loc", "Dupe")
MODELS = ("yolov8n", "yolov8s", "yolov8m")
THUMB = 320


def _select(frame: pd.DataFrame, per_type: int) -> list[pd.Series]:
    picked: list[pd.Series] = []
    for error_type in ORDER:
        block = frame[frame["error_type"] == error_type]
        if block.empty:
            continue
        block = block.sort_values("area", ascending=False)
        per_model = {name: part.reset_index(drop=True) for name, part in block.groupby("model")}
        chosen = 0
        while chosen < per_type and any(len(part) for part in per_model.values()):
            for model in MODELS:
                if chosen >= per_type:
                    break
                part = per_model.get(model)
                if part is None or part.empty:
                    continue
                picked.append(part.iloc[0])
                per_model[model] = part.iloc[1:]
                chosen += 1
    return picked


def _crop(image: Image.Image, row: pd.Series) -> tuple[Image.Image, float, float]:
    width, height = image.size
    x, y = float(row["bbox_x"]), float(row["bbox_y"])
    w, h = float(row["bbox_w"]), float(row["bbox_h"])
    pad = 0.6 * max(w, h, 24.0)
    left = int(max(0, x - pad))
    top = int(max(0, y - pad))
    right = int(min(width, x + w + pad))
    bottom = int(min(height, y + h + pad))
    if right - left < 8 or bottom - top < 8:
        right, bottom = min(width, left + 8), min(height, top + 8)
    return image.crop((left, top, right, bottom)), float(left), float(top)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", default="results/tables/rq3_error_taxonomy.csv")
    parser.add_argument("--gt", default="data/annotations/hk_test_coco.json")
    parser.add_argument("--images", default="data/hk/images/test")
    parser.add_argument("--out-csv", default="data/annotations/rq3_manual.csv")
    parser.add_argument("--crops", default="results/figures/rq3_crops")
    parser.add_argument("--per-type", type=int, default=3)
    args = parser.parse_args()

    frame = pd.read_csv(args.taxonomy)
    gt = json.loads((ROOT / args.gt).read_text(encoding="utf-8"))
    names = {int(category["id"]): str(category["name"]) for category in gt["categories"]}
    gt_by_image: dict[int, list[dict]] = {}
    for annotation in gt["annotations"]:
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    selected = _select(frame, args.per_type)
    crop_dir = ROOT / args.crops
    crop_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    tiles: list[tuple[Image.Image, str]] = []
    for index, row in enumerate(selected, start=1):
        image_path = ROOT / args.images / str(row["file_name"])
        if not image_path.exists():
            print(f"[skip] missing image {image_path}")
            continue
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            crop, offset_x, offset_y = _crop(image, row)

        draw = ImageDraw.Draw(crop)
        # Ground truth in green, the instance under review in red (blue = a miss).
        for annotation in gt_by_image.get(int(row["image_id"]), []):
            x, y, w, h = (float(value) for value in annotation["bbox"])
            draw.rectangle(
                [x - offset_x, y - offset_y, x + w - offset_x, y + h - offset_y],
                outline=(0, 200, 0),
                width=2,
            )
        colour = (0, 80, 255) if row["error_type"] == "Miss" else (230, 0, 0)
        draw.rectangle(
            [
                float(row["bbox_x"]) - offset_x,
                float(row["bbox_y"]) - offset_y,
                float(row["bbox_x"]) + float(row["bbox_w"]) - offset_x,
                float(row["bbox_y"]) + float(row["bbox_h"]) - offset_y,
            ],
            outline=colour,
            width=3,
        )

        scale = THUMB / max(crop.size)
        if scale < 1:
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        caption = f"{index:02d} {row['error_type']} {row['model']} {names.get(int(row['category_id']), '?')}"
        crop.save(crop_dir / f"err_{index:02d}_{row['error_type'].lower()}.png")

        tile = Image.new("RGB", (THUMB, THUMB), (240, 240, 240))
        tile.paste(crop.convert("RGB"), (0, 0))
        tiles.append((tile, caption))
        records.append(
            {
                "sample_id": index,
                "error_type": row["error_type"],
                "model": row["model"],
                "image_id": int(row["image_id"]),
                "file_name": row["file_name"],
                "category": names.get(int(row["category_id"]), "?"),
                "nearest_gt_category": names.get(int(row["nearest_gt_category"]), "")
                if pd.notna(row.get("nearest_gt_category"))
                else "",
                "nearest_iou": row.get("nearest_iou"),
                "score": row.get("score"),
                "scale_bucket": row["scale_bucket"],
                "crop": str(Path(args.crops) / f"err_{index:02d}_{row['error_type'].lower()}.png"),
                "occlusion_level": "",
                "text_density": "",
                "note": "",
            }
        )

    if tiles:
        columns = 5
        rows = (len(tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * THUMB, rows * (THUMB + 18)), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for position, (tile, caption) in enumerate(tiles):
            column, row_index = position % columns, position // columns
            sheet.paste(tile, (column * THUMB, row_index * (THUMB + 18)))
            draw.text((column * THUMB + 4, row_index * (THUMB + 18) + THUMB + 2), caption, fill=(0, 0, 0))
        sheet_path = ROOT / args.crops / "error_sheet.png"
        sheet.save(sheet_path)
        print(f"contact sheet: {sheet_path}")

    out = ROOT / args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"selected {len(records)} instances")
    print(f"form      : {out}")
    print("fill `occlusion_level` (none / partial / heavy) and `text_density` (low / medium / high)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
