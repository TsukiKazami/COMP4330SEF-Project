"""Instance-level error taxonomy for RQ3.

TIDE gives the *dAP* of each error type but not how many instances of each kind
occur, and RQ3 needs counts to cross-tabulate errors against scene attributes.
This module produces that instance table.

Each error is counted exactly once:

    Dupe  a detection that matches a ground truth already taken by a
          higher-scoring detection
    Cls   a detection that matches a ground truth of another class (IoU >= 0.5)
    Loc   a detection that overlaps a ground truth of its own class,
          0.1 <= IoU < 0.5
    Bkg   any other detection (no ground-truth overlap >= 0.1)
    Miss  a ground truth that no detection of any class overlaps at IoU >= 0.1

Loc/Cls/Dupe detections are the ones that cost their ground truth, so the
corresponding unmatched ground truth is not counted a second time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from comp4330sef.eval.matching import iou_matrix, xywh_to_xyxy

# COCO size buckets, in pixels of the 640x640 evaluation image.
SMALL_AREA = 32 ** 2
LARGE_AREA = 96 ** 2
BUCKETS = ("small", "medium", "large")
ERROR_TYPES = ("Miss", "Loc", "Cls", "Dupe", "Bkg")


def scale_bucket(area: float) -> str:
    if area < SMALL_AREA:
        return "small"
    if area < LARGE_AREA:
        return "medium"
    return "large"


def classify(
    gt: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    model: str,
    conf: float = 0.25,
    match_iou: float = 0.5,
    bg_iou: float = 0.1,
    max_dets: int = 100,
) -> list[dict[str, Any]]:
    """Return one row per error instance at the operating confidence ``conf``.

    TIDE integrates over the whole confidence range, so its dAP counts every
    low-score box. An instance table does not want that: at the 0.001 floor more
    than 85% of the rows are background noise. The taxonomy therefore runs at
    the operating point, which is also what the proposal asks for ("all
    misdetections under the optimal parameter configuration").

    ``max_dets`` mirrors TIDE's COCO loader for the surviving boxes.
    """
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in gt.get("annotations", []):
        gt_by_image[int(annotation["image_id"])].append(annotation)

    det_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if float(record["score"]) < conf:
            continue
        det_by_image[int(record["image_id"])].append(record)

    file_name = {int(image["id"]): str(image["file_name"]) for image in gt.get("images", [])}
    rows: list[dict[str, Any]] = []

    for image_id, truths in gt_by_image.items():
        detections = sorted(det_by_image.get(image_id, []), key=lambda item: -float(item["score"]))
        detections = detections[:max_dets]
        if not truths:
            continue

        gt_boxes = np.asarray([xywh_to_xyxy(item["bbox"]) for item in truths], dtype=float)
        gt_class = np.asarray([int(item["category_id"]) for item in truths])

        if detections:
            det_boxes = np.asarray([xywh_to_xyxy(item["bbox"]) for item in detections], dtype=float)
            det_class = np.asarray([int(item["category_id"]) for item in detections])
            ious = iou_matrix(det_boxes, gt_boxes)
        else:
            ious = np.zeros((0, len(truths)), dtype=float)
            det_class = np.zeros(0, dtype=int)

        used_gt = np.zeros(len(truths), dtype=bool)
        matched = np.zeros(len(detections), dtype=bool)

        for index in range(len(detections)):
            candidates = np.where(~used_gt)[0]
            same = candidates[gt_class[candidates] == det_class[index]]
            if len(same) == 0:
                continue
            best = int(same[np.argmax(ious[index, same])])
            if ious[index, best] >= match_iou:
                used_gt[best] = True
                matched[index] = True

        for index, detection in enumerate(detections):
            if matched[index]:
                continue
            same = np.where(gt_class == det_class[index])[0]
            other = np.where(gt_class != det_class[index])[0]
            same_max = float(ious[index, same].max()) if len(same) else 0.0
            other_max = float(ious[index, other].max()) if len(other) else 0.0

            if same_max >= match_iou:
                error_type = "Dupe"
                nearest = int(same[np.argmax(ious[index, same])])
            elif other_max >= match_iou:
                error_type = "Cls"
                nearest = int(other[np.argmax(ious[index, other])])
            elif same_max >= bg_iou:
                error_type = "Loc"
                nearest = int(same[np.argmax(ious[index, same])])
            else:
                error_type = "Bkg"
                nearest = int(np.argmax(ious[index])) if len(truths) else None

            x, y, w, h = (float(value) for value in detection["bbox"])
            rows.append(
                {
                    "model": model,
                    "image_id": int(image_id),
                    "file_name": file_name.get(image_id),
                    "category_id": int(detection["category_id"]),
                    "error_type": error_type,
                    "side": "detection",
                    "score": round(float(detection["score"]), 4),
                    "nearest_gt_category": None if nearest is None else int(gt_class[nearest]),
                    "nearest_iou": round(float(ious[index, nearest]), 4) if nearest is not None else None,
                    "bbox_x": round(x, 2),
                    "bbox_y": round(y, 2),
                    "bbox_w": round(w, 2),
                    "bbox_h": round(h, 2),
                    "area": round(w * h, 2),
                    "scale_bucket": scale_bucket(w * h),
                }
            )

        overlapped = ious.max(axis=0) if len(detections) else np.zeros(len(truths), dtype=float)
        for index, truth in enumerate(truths):
            if used_gt[index] or overlapped[index] >= bg_iou:
                continue
            x, y, w, h = (float(value) for value in truth["bbox"])
            rows.append(
                {
                    "model": model,
                    "image_id": int(image_id),
                    "file_name": file_name.get(image_id),
                    "category_id": int(truth["category_id"]),
                    "error_type": "Miss",
                    "side": "ground_truth",
                    "score": None,
                    "bbox_x": round(x, 2),
                    "bbox_y": round(y, 2),
                    "bbox_w": round(w, 2),
                    "bbox_h": round(h, 2),
                    "area": round(w * h, 2),
                    "scale_bucket": scale_bucket(w * h),
                }
            )

    return rows


def chi_square(frame: pd.DataFrame, factor: str = "scale_bucket") -> pd.DataFrame:
    """Test whether the error-type mix depends on ``factor``, pooled and per model."""
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    groups += [(str(name), block) for name, block in frame.groupby("model")]

    for name, block in groups:
        table = pd.crosstab(block["error_type"], block[factor])
        if table.shape[0] < 2 or table.shape[1] < 2:
            rows.append(
                {
                    "group": name,
                    "factor": factor,
                    "n": int(table.values.sum()),
                    "chi2": None,
                    "dof": None,
                    "p_value": None,
                    "cramers_v": None,
                    "min_expected": None,
                    "note": "fewer than two error types or two levels",
                }
            )
            continue
        chi2, p_value, dof, expected = chi2_contingency(table)
        total = int(table.values.sum())
        cramers_v = math.sqrt(chi2 / (total * (min(table.shape) - 1))) if total else None
        rows.append(
            {
                "group": name,
                "factor": factor,
                "n": total,
                "chi2": round(float(chi2), 4),
                "dof": int(dof),
                "p_value": float(p_value),
                "cramers_v": round(float(cramers_v), 4) if cramers_v is not None else None,
                "min_expected": round(float(expected.min()), 2),
                "note": "expected count < 5" if (expected < 5).any() else "",
            }
        )
    return pd.DataFrame(rows)
