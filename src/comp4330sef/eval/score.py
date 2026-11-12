from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from comp4330sef.eval.coco import as_records, evaluate_coco, load_json
from comp4330sef.eval.matching import build_gt_index, counts_at_conf


def restrict_to_categories(
    records: Sequence[dict[str, Any]], categories: set[int]
) -> list[dict[str, Any]]:
    return [record for record in records if int(record["category_id"]) in categories]


def score_run(
    gt_path: str | Path,
    detections: Sequence[dict[str, Any]] | str | Path,
    *,
    operating_conf: float = 0.25,
    match_iou: float = 0.5,
    restrict_to_gt_classes: bool = True,
) -> dict[str, Any]:
    gt = load_json(gt_path)
    records = as_records(detections)

    ap_result = evaluate_coco(gt_path, records)
    gt_categories = {int(category["id"]) for category in gt.get("categories", [])}
    evaluated = (
        restrict_to_categories(records, gt_categories)
        if restrict_to_gt_classes
        else list(records)
    )
    operating_point = counts_at_conf(
        build_gt_index(gt), evaluated, conf=operating_conf, iou_thr=match_iou
    )
    gt_instances = len(gt.get("annotations", []))

    return {
        "ap": ap_result["ap"],
        "operating_point": operating_point,
        "per_class": ap_result["per_class"],
        "counts": {
            "images": len(gt.get("images", [])),
            "gt_instances": gt_instances,
            "classes": len(gt.get("categories", [])),
            "detections": len(records),
            "detections_evaluated": len(evaluated),
        },
        "operating_point_filter": "gt_categories" if restrict_to_gt_classes else "all_classes",
        "consistency": {
            "tp_plus_fp_equals_kept": operating_point["tp"] + operating_point["fp"]
            == operating_point["detections_kept"],
            "tp_plus_fn_equals_gt": operating_point["tp"] + operating_point["fn"] == gt_instances,
        },
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_result(result: dict[str, Any], top: int = 10) -> str:
    ap = result["ap"]
    operating = result["operating_point"]
    counts = result["counts"]

    lines = [
        f"GT       : {counts['images']} images, {counts['gt_instances']} instances, "
        f"{counts['classes']} classes",
        f"dets     : {counts['detections']} records "
        f"({counts.get('detections_evaluated', counts['detections'])} kept for P/R/F1)",
        "AP       : "
        + "  ".join(
            f"{name}={_fmt(ap.get(name))}" for name in ("mAP_50_95", "mAP_50", "mAP_75")
        ),
        f"operating: conf={operating['conf']} iou={operating['iou']}  "
        f"TP={operating['tp']} FP={operating['fp']} FN={operating['fn']}  "
        f"P={operating['precision']:.4f} R={operating['recall']:.4f} F1={operating['f1']:.4f}",
    ]

    rows = result.get("per_class") or []
    if rows:
        lines.append("per class (worst first):")
        ordered = sorted(rows, key=lambda row: row["AP_50_95"] if row["AP_50_95"] is not None else -1.0)
        for row in ordered[:top]:
            lines.append(
                f"  {row['name']:<18} AP50-95={_fmt(row['AP_50_95']):<7} "
                f"AP50={_fmt(row['AP_50']):<7} gt={row['gt_instances']}"
            )
    return "\n".join(lines)