from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

COCO_NUM_CLASSES = 80


def category_id(ultralytics_index: int) -> int:
    return int(ultralytics_index) + 1


def image_id_from_stem(path: str | Path) -> int | None:
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else None


def detections_to_coco(
    results: Sequence[Any],
    image_ids: Sequence[int],
    class_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    if len(results) != len(image_ids):
        raise ValueError(f"got {len(results)} results but {len(image_ids)} image ids")

    records: list[dict[str, Any]] = []
    for result, image_id in zip(results, image_ids):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy().astype(float)
        scores = boxes.conf.cpu().numpy().astype(float)
        classes = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
            index = int(cls)
            if class_ids is None:
                category = category_id(index)
            else:
                if not 0 <= index < len(class_ids):
                    raise ValueError(
                        f"class index {index} outside class_ids with {len(class_ids)} entries"
                    )
                category = int(class_ids[index])
            records.append(
                {
                    "image_id": int(image_id),
                    "category_id": category,
                    "bbox": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(max(float(x2 - x1), 0.0), 2),
                        round(max(float(y2 - y1), 0.0), 2),
                    ],
                    "score": round(float(score), 5),
                }
            )
    return records


def validate_records(
    records: Iterable[dict[str, Any]], num_classes: int = COCO_NUM_CLASSES
) -> dict[str, int]:
    count = 0
    problems = 0
    for record in records:
        count += 1
        x, y, width, height = record["bbox"]
        ok = (
            1 <= record["category_id"] <= num_classes
            and 0.0 <= record["score"] <= 1.0
            and width > 0
            and height > 0
            and x >= 0
            and y >= 0
        )
        problems += 0 if ok else 1
    return {"records": count, "problems": problems}