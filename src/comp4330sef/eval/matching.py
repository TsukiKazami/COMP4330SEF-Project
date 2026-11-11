from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box[:4])
    return np.array([x, y, x + width, y + height], dtype=float)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def build_gt_index(gt: dict[str, Any]) -> dict[int, dict[int, list[list[float]]]]:
    index: dict[int, dict[int, list[list[float]]]] = {
        int(image["id"]): {} for image in gt.get("images", [])
    }
    for annotation in gt.get("annotations", []):
        if annotation.get("iscrowd"):
            continue
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        index.setdefault(image_id, {}).setdefault(category_id, []).append(
            xywh_to_xyxy(annotation["bbox"]).tolist()
        )
    return index


def _match_image(
    gt_by_category: dict[int, list[list[float]]],
    detections_by_category: dict[int, list[dict[str, Any]]],
    iou_thr: float,
) -> tuple[int, int, int]:
    tp = 0
    fp = 0
    total_gt = 0

    for category_id in sorted(set(gt_by_category) | set(detections_by_category)):
        gt_boxes = gt_by_category.get(category_id, [])
        detections = detections_by_category.get(category_id, [])
        total_gt += len(gt_boxes)

        if not detections:
            continue
        if not gt_boxes:
            fp += len(detections)
            continue

        gt_xyxy = np.asarray(gt_boxes, dtype=float)
        matched = np.zeros(len(gt_xyxy), dtype=bool)
        for detection in sorted(detections, key=lambda item: -float(item["score"])):
            ious = iou_matrix(xywh_to_xyxy(detection["bbox"])[None, :], gt_xyxy)[0]
            ious[matched] = -1.0
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                matched[best] = True
                tp += 1
            else:
                fp += 1

    return tp, fp, total_gt - tp


def counts_at_conf(
    gt_index: dict[int, dict[int, list[list[float]]]],
    detections: Iterable[dict[str, Any]],
    *,
    conf: float = 0.25,
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = {}
    kept = 0
    for detection in detections:
        if float(detection["score"]) < conf:
            continue
        kept += 1
        grouped.setdefault(int(detection["image_id"]), {}).setdefault(
            int(detection["category_id"]), []
        ).append(detection)

    tp = fp = fn = 0
    for image_id in sorted(set(gt_index) | set(grouped)):
        image_tp, image_fp, image_fn = _match_image(
            gt_index.get(image_id, {}), grouped.get(image_id, {}), iou_thr
        )
        tp += image_tp
        fp += image_fp
        fn += image_fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "conf": conf,
        "iou": iou_thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "detections_kept": kept,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }