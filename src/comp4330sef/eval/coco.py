from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_KEYS = (
    ("mAP_50_95", 0),
    ("mAP_50", 1),
    ("mAP_75", 2),
    ("AP_small", 3),
    ("AP_medium", 4),
    ("AP_large", 5),
    ("AR_1", 6),
    ("AR_10", 7),
    ("AR_100", 8),
)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def as_records(detections: Sequence[dict[str, Any]] | str | Path) -> list[dict[str, Any]]:
    if isinstance(detections, (str, Path)):
        return list(load_json(detections))
    return list(detections)


def _clean(value: float) -> float | None:
    return None if math.isnan(float(value)) else round(float(value), 4)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def evaluate_coco(
    gt_path: str | Path,
    detections: Sequence[dict[str, Any]] | str | Path,
) -> dict[str, Any]:
    records = as_records(detections)

    with quiet():
        coco_gt = COCO(str(gt_path))

    gt_counts: dict[int, int] = {}
    for annotation in coco_gt.dataset.get("annotations", []):
        category_id = int(annotation["category_id"])
        gt_counts[category_id] = gt_counts.get(category_id, 0) + 1
    category_names = {
        int(category["id"]): str(category["name"])
        for category in coco_gt.dataset.get("categories", [])
    }

    if not records:
        return {
            "ap": {name: None for name, _ in STAT_KEYS},
            "per_class": [],
            "num_detections": 0,
        }

    with quiet():
        coco_dt = coco_gt.loadRes(records)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        "ap": {name: _clean(evaluator.stats[index]) for name, index in STAT_KEYS},
        "per_class": _per_class(evaluator, category_names, gt_counts),
        "num_detections": len(records),
    }


def _per_class(
    evaluator: COCOeval,
    category_names: dict[int, str],
    gt_counts: dict[int, int],
) -> list[dict[str, Any]]:
    precision = evaluator.eval.get("precision")
    if precision is None:
        return []

    iou_50 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.5)))
    rows: list[dict[str, Any]] = []

    for position, raw_category_id in enumerate(evaluator.params.catIds):
        category_id = int(raw_category_id)
        if gt_counts.get(category_id, 0) == 0:
            continue
        all_iou = precision[:, :, position, 0, -1]
        at_50 = precision[iou_50, :, position, 0, -1]
        rows.append(
            {
                "category_id": category_id,
                "name": category_names.get(category_id, str(category_id)),
                "AP_50_95": _clean(all_iou[all_iou > -1].mean()) if (all_iou > -1).any() else None,
                "AP_50": _clean(at_50[at_50 > -1].mean()) if (at_50 > -1).any() else None,
                "gt_instances": gt_counts.get(category_id, 0),
            }
        )

    rows.sort(key=lambda row: (-(row["AP_50_95"] or 0.0), row["name"]))
    return rows