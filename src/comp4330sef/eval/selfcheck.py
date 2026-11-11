from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from comp4330sef.config import REPO_ROOT, load_models_config, rel_path, resolve_model
from comp4330sef.data.yolo_to_coco import labels_dir_for, names_to_dict, yolo_split_to_coco
from comp4330sef.eval.results import detections_to_coco, image_id_from_stem
from comp4330sef.eval.runcard import new_run_id, precision_check, write_runcard
from comp4330sef.eval.score import format_result, score_run

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TOLERANCE_AP50_95 = 0.03
TOLERANCE_AP50 = 0.10
MIN_SANE_AP50_95 = 0.20

CHECKS = (
    ("mAP_50_95", "metrics/mAP50-95(B)", TOLERANCE_AP50_95),
    ("mAP_50", "metrics/mAP50(B)", TOLERANCE_AP50),
)


def _split_images(data: dict[str, Any], split: str) -> list[Path]:
    entry = data.get(split)
    if entry is None:
        raise SystemExit(f"dataset has no {split!r} split")
    entries = entry if isinstance(entry, (list, tuple)) else [entry]

    files: list[Path] = []
    for item in entries:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            files.append(path)
    if not files:
        raise SystemExit(f"no images found for split {split!r}")
    return files


def _reference_metrics(
    weights: Path,
    data_name: str,
    split: str,
    cfg: Any,
    conf_floor: float,
) -> dict[str, float] | None:
    try:
        model = YOLO(str(weights))
        metrics = model.val(
            data=data_name,
            split=split,
            imgsz=cfg.imgsz,
            conf=conf_floor,
            iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
            batch=16,
            verbose=False,
            plots=False,
        )
    except Exception as exc:
        print(f"[warn] Ultralytics cross-check unavailable: {exc}")
        return None
    return {key: float(value) for key, value in metrics.results_dict.items()}


def run_selfcheck(config_path: str, model_name: str, split: str, reference: bool = True) -> int:
    cfg_dict = load_models_config(config_path)
    cfg = resolve_model(cfg_dict, model_name)
    eval_cfg = cfg_dict.get("eval") or {}
    conf_floor = float(eval_cfg.get("conf_floor", 0.001))
    match_iou = float(eval_cfg.get("iou_match", 0.5))

    dummy_cfg = cfg_dict.get("dummy") or {}
    data_name = dummy_cfg.get("data", "coco8.yaml")
    data = check_det_dataset(data_name)
    names = names_to_dict(data["names"])
    images = _split_images(data, split)
    labels_dir = labels_dir_for(images[0].parent)

    gt = yolo_split_to_coco(images, labels_dir, names, split=split)

    run_id = new_run_id(f"phase0_selfcheck_{cfg.name}")
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    gt_path = run_dir / "gt.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    model = YOLO(str(cfg.weights))
    precision = precision_check(model)
    if not precision["ok"]:
        raise SystemExit(f"expected fp32 weights, got {precision['actual']}")
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=cfg.imgsz,
        conf=conf_floor,
        iou=cfg.iou,
        max_det=cfg.max_det,
        batch=cfg.batch,
        device=cfg.device,
        verbose=False,
        save=False,
    )

    image_ids = []
    for path in images:
        image_id = image_id_from_stem(path)
        if image_id is None:
            raise SystemExit(f"{path.name}: numeric filename required")
        image_ids.append(image_id)

    records = detections_to_coco(results, image_ids)
    detections_path = run_dir / "detections.json"
    detections_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    scored = score_run(gt_path, records, operating_conf=cfg.conf, match_iou=match_iou)
    scored["dataset"] = {
        "name": data_name,
        "split": split,
        "images": len(images),
        "gt_instances": len(gt["annotations"]),
    }
    scored["config"] = cfg.to_dict()
    scored["precision"] = precision
    scored["eval"] = {"conf_floor": conf_floor, "iou_match": match_iou}

    reference_metrics = (
        _reference_metrics(cfg.weights, data_name, split, cfg, conf_floor) if reference else None
    )
    checks: list[dict[str, Any]] = []
    if reference_metrics:
        scored["reference"] = reference_metrics
        for ours_key, ref_key, tolerance in CHECKS:
            ours = scored["ap"].get(ours_key)
            theirs = reference_metrics.get(ref_key)
            if ours is None or theirs is None:
                continue
            delta = abs(float(ours) - float(theirs))
            checks.append(
                {
                    "metric": ours_key,
                    "ours": round(float(ours), 4),
                    "ultralytics": round(float(theirs), 4),
                    "delta": round(delta, 4),
                    "tolerance": tolerance,
                    "pass": delta <= tolerance,
                }
            )
    scored["checks"] = checks

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")

    print(format_result(scored))
    if checks:
        print("\ncross-check vs Ultralytics val:")
        for check in checks:
            status = "OK" if check["pass"] else "FAIL"
            print(
                f"  {check['metric']:<10} ours={check['ours']:.4f}  "
                f"ultralytics={check['ultralytics']:.4f}  "
                f"delta={check['delta']:.4f}  [{status}]"
            )

    runcard_path = write_runcard(
        run_id,
        config=cfg_dict,
        config_path=config_path,
        device=cfg.device,
        artifacts={
            "gt": rel_path(gt_path),
            "detections": rel_path(detections_path),
            "metrics": rel_path(metrics_path),
        },
        extra={
            "model": cfg.to_dict(),
            "eval": {"conf_floor": conf_floor, "iou_match": match_iou},
            "check_tolerance": {"mAP_50_95": TOLERANCE_AP50_95, "mAP_50": TOLERANCE_AP50},
            "precision": precision,
        },
    )

    print(f"\nartifacts: {rel_path(run_dir)}")
    print(f"runcard  : {rel_path(runcard_path)}")

    if not records:
        print("\nFAIL: no detections at all")
        return 1
    ap_50_95 = scored["ap"].get("mAP_50_95")
    if ap_50_95 is None or ap_50_95 < MIN_SANE_AP50_95:
        print(
            f"\nFAIL: mAP@50:95={ap_50_95} is far below the expected range for "
            "pretrained weights — suspect the GT/prediction category mapping"
        )
        return 1
    if checks and not all(check["pass"] for check in checks):
        print("\nFAIL: our AP disagrees with Ultralytics beyond tolerance")
        return 1
    print("\nOK: metric stack validated")
    return 0