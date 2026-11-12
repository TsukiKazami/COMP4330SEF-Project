"""RQ1: accuracy/speed scaling of yolov8n/s/m.

One run per model: an accuracy pass at the AP confidence floor, an operating
point (P/R/F1 at conf=0.25), a latency protocol at batch 1, and a runcard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

from comp4330sef.config import REPO_ROOT, load_models_config, rel_path, resolve_model
from comp4330sef.data.eval_set import EvalSet, from_coco, from_yolo_yaml
from comp4330sef.eval.results import detections_to_coco
from comp4330sef.eval.runcard import git_state, new_run_id, precision_check, write_runcard
from comp4330sef.eval.score import format_result, score_run
from comp4330sef.eval.timing import measure_latency

HK_NAMES = {"hk", "hk_test"}
TABLE_COLUMNS = (
    "model",
    "params",
    "mAP_50_95",
    "mAP_50",
    "precision",
    "recall",
    "f1",
    "latency_total_ms",
    "fps",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _table_paths(
    slug: str,
    table: str | None,
    per_class: str | None,
    marginal: str | None,
) -> tuple[Path, Path, Path]:
    suffix = "" if slug.lower() in HK_NAMES else f"_{slug}"
    tables = REPO_ROOT / "results" / "tables"
    return (
        _resolve(table) if table else tables / f"rq1_scaling{suffix}.csv",
        _resolve(per_class) if per_class else tables / f"rq1_per_class{suffix}.csv",
        _resolve(marginal) if marginal else tables / f"rq1_marginal{suffix}.csv",
    )


def _upsert(
    path: Path,
    rows: list[dict[str, Any]],
    keys: Sequence[str],
    sort_by: Sequence[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        if not existing.empty:
            frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=list(keys), keep="last")

    frame = frame.sort_values(list(sort_by), kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _compute_marginal(rows: list[dict[str, Any]]) -> pd.DataFrame:
    ordered = sorted(rows, key=lambda row: row["params"])
    records: list[dict[str, Any]] = []
    for previous, current in zip(ordered, ordered[1:]):
        if previous["mAP_50_95"] is None or current["mAP_50_95"] is None:
            continue
        delta_params = (current["params"] - previous["params"]) / 1e6
        delta_map = float(current["mAP_50_95"]) - float(previous["mAP_50_95"])
        records.append(
            {
                "dataset": current["dataset"],
                "from_model": previous["model"],
                "to_model": current["model"],
                "from_params": previous["params"],
                "to_params": current["params"],
                "delta_params_m": round(delta_params, 2),
                "delta_mAP_50_95": round(delta_map, 4),
                "delta_mAP_50": round(float(current["mAP_50"]) - float(previous["mAP_50"]), 4),
                "map_per_mparam": round(delta_map / delta_params, 4) if delta_params else None,
                "delta_latency_ms": round(
                    float(current["latency_total_ms"]) - float(previous["latency_total_ms"]), 2
                ),
            }
        )
    return pd.DataFrame(records)


def _inflection_note(marginal: pd.DataFrame) -> str:
    if marginal.empty or "map_per_mparam" not in marginal:
        return "not enough size steps to locate an inflection point"
    ratios = [value for value in marginal["map_per_mparam"].tolist() if value is not None]
    if len(ratios) < 2:
        return "a single size step: no inflection can be judged yet"
    first, last = float(ratios[0]), float(ratios[-1])
    if first == 0:
        return "the first step already gains nothing per parameter"
    ratio = last / first
    if ratio <= 0:
        return (
            f"marginal return turns zero or negative ({last:.4f} vs {first:.4f}): "
            "the larger checkpoint is not buying accuracy"
        )
    drop = (1 - ratio) * 100
    verdict = "diminishing returns" if drop > 25 else "still improving"
    return f"marginal return changes by {-drop:+.0f}% across the size range ({verdict})"


def _evaluate_model(
    cfg: Any,
    eval_set: EvalSet,
    *,
    cfg_dict: dict[str, Any],
    config_path: str,
    conf_floor: float,
    match_iou: float,
    restrict_classes: bool,
    acc_batch: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch.manual_seed(cfg.seed)

    run_id = new_run_id(f"rq1_{cfg.name}")
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    gt_path = run_dir / "gt.json"
    gt_path.write_text(json.dumps(eval_set.gt, indent=2), encoding="utf-8")

    model = YOLO(str(cfg.weights))
    precision = precision_check(model)
    if not precision["ok"]:
        raise SystemExit(f"{cfg.name}: expected fp32 weights, got {precision['actual']}")

    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    weight_mb = round(cfg.weights.stat().st_size / 1e6, 1)

    results = model.predict(
        source=[str(path) for path in eval_set.images],
        imgsz=cfg.imgsz,
        conf=conf_floor,
        iou=cfg.iou,
        max_det=cfg.max_det,
        batch=acc_batch,
        device=cfg.device,
        verbose=False,
        save=False,
    )
    if len(results) != len(eval_set.images):
        raise SystemExit(f"{cfg.name}: {len(results)} results for {len(eval_set.images)} images")

    records = detections_to_coco(results, eval_set.image_ids)
    detections_path = run_dir / "detections.json"
    detections_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    scored = score_run(
        gt_path,
        records,
        operating_conf=cfg.conf,
        match_iou=match_iou,
        restrict_to_gt_classes=restrict_classes,
    )

    timing_image = eval_set.images[0]
    image = cv2.imread(str(timing_image))
    if image is None:
        raise SystemExit(f"could not read timing image {timing_image}")
    timing = measure_latency(
        model,
        image,
        predict_kwargs={
            "imgsz": cfg.imgsz,
            "conf": cfg.conf,
            "iou": cfg.iou,
            "max_det": cfg.max_det,
            "batch": 1,
            "device": cfg.device,
            "verbose": False,
            "save": False,
        },
        warmup=warmup,
        iterations=iterations,
    )

    row = {
        "dataset": eval_set.name,
        "model": cfg.name,
        "params": parameters,
        "weight_mb": weight_mb,
        "imgsz": cfg.imgsz,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "conf_floor": conf_floor,
        "acc_batch": acc_batch,
        "mAP_50_95": scored["ap"]["mAP_50_95"],
        "mAP_50": scored["ap"]["mAP_50"],
        "mAP_75": scored["ap"]["mAP_75"],
        "AP_small": scored["ap"]["AP_small"],
        "AP_medium": scored["ap"]["AP_medium"],
        "AP_large": scored["ap"]["AP_large"],
        "AR_100": scored["ap"]["AR_100"],
        "precision": scored["operating_point"]["precision"],
        "recall": scored["operating_point"]["recall"],
        "f1": scored["operating_point"]["f1"],
        **timing,
        "images": len(eval_set.images),
        "gt_instances": scored["counts"]["gt_instances"],
        "detections": scored["counts"]["detections"],
        "detections_evaluated": scored["counts"]["detections_evaluated"],
        "run_id": run_id,
        "git_sha": git_state()["short_sha"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    payload = {
        "dataset": {
            "name": eval_set.name,
            "source": eval_set.source,
            "images": len(eval_set.images),
            "gt_instances": scored["counts"]["gt_instances"],
        },
        "model": {**cfg.to_dict(), "params": parameters, "weight_mb": weight_mb},
        "protocol": {
            "acc_batch": acc_batch,
            "conf_floor": conf_floor,
            "match_iou": match_iou,
            "operating_point_filter": scored["operating_point_filter"],
            "warmup": warmup,
            "iterations": iterations,
            "timing_image": rel_path(timing_image),
        },
        "ap": scored["ap"],
        "operating_point": scored["operating_point"],
        "timing": timing,
        "precision": precision,
        "consistency": scored.get("consistency"),
        "per_class": scored["per_class"],
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_runcard(
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
            "model": payload["model"],
            "dataset": payload["dataset"],
            "protocol": payload["protocol"],
            "timing": timing,
            "precision": precision,
        },
    )

    del results, model
    torch.cuda.empty_cache()

    return row, scored, payload


def run_rq1(
    *,
    config_path: str = "configs/models.yaml",
    models: Sequence[str] = ("yolov8n", "yolov8s", "yolov8m"),
    data: str = "datasets/coco8.yaml",
    split: str = "val",
    gt: str | None = None,
    images: str | None = None,
    tag: str | None = None,
    table: str | None = None,
    per_class_table: str | None = None,
    marginal_table: str | None = None,
    acc_batch: int = 8,
    warmup: int = 10,
    restrict_classes: bool = True,
    iterations: int = 100,
) -> int:
    cfg_dict = load_models_config(config_path)
    eval_cfg = cfg_dict.get("eval") or {}
    conf_floor = float(eval_cfg.get("conf_floor", 0.001))
    match_iou = float(eval_cfg.get("iou_match", 0.5))

    eval_set = from_coco(gt, images, name=tag) if gt else from_yolo_yaml(data, split, name=tag)
    scaling_path, per_class_path, marginal_path = _table_paths(
        eval_set.name, table, per_class_table, marginal_table
    )

    print(
        f"dataset  : {eval_set.name} — {len(eval_set.images)} images, "
        f"{len(eval_set.gt.get('annotations', []))} instances"
    )
    print(f"protocol : conf_floor={conf_floor}  acc_batch={acc_batch}  "
          f"warmup={warmup}  iters={iterations}")
    print(f"filter   : P/R/F1 on {'GT categories only' if restrict_classes else 'all classes'}")
    if eval_set.name.lower() not in HK_NAMES:
        print("note     : sample dataset — this validates the pipeline only; "
              "the RQ1 answer needs the Hong Kong test set (--gt ... --tag hk_test)")

    scaling_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for name in models:
        cfg = resolve_model(cfg_dict, name)
        print(f"\n=== {cfg.name} ===")
        row, scored, payload = _evaluate_model(
            cfg,
            eval_set,
            cfg_dict=cfg_dict,
            config_path=config_path,
            conf_floor=conf_floor,
            match_iou=match_iou,
            acc_batch=acc_batch,
            warmup=warmup,
            iterations=iterations,
            restrict_classes=restrict_classes,
        )
        print(format_result(scored))
        print(
            f"timing   : {row['latency_total_ms']:.2f} ms/frame "
            f"({row['fps']:.1f} FPS, p95={row['latency_total_p95_ms']:.2f} ms, "
            f"wall={row['wall_mean_ms']:.2f} ms)"
        )
        scaling_rows.append(row)
        for class_row in payload["per_class"]:
            per_class_rows.append({"dataset": eval_set.name, "model": cfg.name, **class_row})

    scaling = _upsert(
        scaling_path, scaling_rows, keys=("dataset", "model"), sort_by=("dataset", "params")
    )
    if per_class_rows:
        _upsert(
            per_class_path,
            per_class_rows,
            keys=("dataset", "model", "category_id"),
            sort_by=("dataset", "model", "category_id"),
        )

    marginal = _compute_marginal(scaling_rows)
    if not marginal.empty:
        _upsert(
            marginal_path,
            marginal.to_dict("records"),
            keys=("dataset", "from_model", "to_model"),
            sort_by=("dataset", "from_params"),
        )

    print("\nscaling table:")
    print(scaling[list(TABLE_COLUMNS)].to_string(index=False))

    if not marginal.empty:
        print("\nmarginal return per +1M parameters:")
        print(marginal.to_string(index=False))
        print(f"\ninflection: {_inflection_note(marginal)}")

    print(f"\ntables   : {rel_path(scaling_path)}")
    print(f"           {rel_path(per_class_path)}")
    if not marginal.empty:
        print(f"           {rel_path(marginal_path)}")
    return 0