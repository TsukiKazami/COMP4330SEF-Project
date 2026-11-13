from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import cv2
import pandas as pd
import torch
from scipy import stats
from ultralytics import YOLO

from comp4330sef.analysis.taxonomy import classify
from comp4330sef.config import REPO_ROOT, load_models_config, rel_path, resolve_model
from comp4330sef.data.eval_set import from_coco
from comp4330sef.eval.matching import build_gt_index, counts_at_conf
from comp4330sef.eval.results import detections_to_coco
from comp4330sef.eval.runcard import new_run_id, write_runcard
from comp4330sef.eval.score import score_run
from comp4330sef.eval.timing import measure_latency

HK_NAMES = {"hk", "hk_test"}
CLASS_IDS = (3, 4, 6, 8)
VARIANTS = ("baseline", "p2")
LABELS = {"baseline": "yolov8s", "p2": "yolov8s-p2"}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def dataset_yaml(source: str = "configs/hk4.yaml", out_dir: Path | None = None) -> Path:
    """Copy the dataset yaml with ``path`` pointing at this checkout."""
    text = _resolve(source).read_text(encoding="utf-8")
    root = (REPO_ROOT / "data" / "hk").resolve().as_posix()
    text = re.sub(r"^path:.*$", f"path: {root}", text, flags=re.MULTILINE)
    target = (out_dir or REPO_ROOT / "results" / "runs") / "hk4.local.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def build_model(variant: str) -> YOLO:
    if variant == "baseline":
        return YOLO(str(_resolve("weights/yolov8s.pt")))
    if variant == "p2":
        config = _resolve("configs/models/yolov8s-p2.yaml")
        if not config.exists():
            raise SystemExit(f"missing {rel_path(config)}")
        model = YOLO(str(config))
        model.load(str(_resolve("weights/yolov8s.pt")))
        return model
    raise SystemExit(f"unknown variant {variant!r}; choose from {VARIANTS}")


def train_variant(
    variant: str,
    *,
    data: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    device: int | str,
    workers: int = 2,
) -> Path:
    model = build_model(variant)
    print(f"\n=== training {LABELS[variant]} ({variant}) ===")
    model.train(
        data=str(data),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        seed=seed,
        deterministic=True,
        patience=0,
        device=device,
        workers=workers,
        project=str(REPO_ROOT / "results" / "runs"),
        name=f"rq5_{variant}",
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    best = REPO_ROOT / "results" / "runs" / f"rq5_{variant}" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training finished without {rel_path(best)}")
    return best


def per_image_f1(
    gt_index: dict[int, dict[int, list[list[float]]]],
    records: Sequence[dict[str, Any]],
    image_ids: Sequence[int],
    *,
    conf: float,
    iou: float,
) -> dict[int, float]:
    by_image: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_image.setdefault(int(record["image_id"]), []).append(record)
    scores: dict[int, float] = {}
    for image_id in image_ids:
        result = counts_at_conf(
            {image_id: gt_index.get(image_id, {})},
            by_image.get(int(image_id), []),
            conf=conf,
            iou_thr=iou,
        )
        scores[int(image_id)] = float(result["f1"])
    return scores


def evaluate_variant(
    variant: str,
    weights: Path,
    eval_set: Any,
    gt_path: Path,
    *,
    conf_floor: float,
    operating_conf: float,
    nms_iou: float,
    match_iou: float,
    acc_batch: int,
    device: int | str,
    timing_iters: int,
) -> dict[str, Any]:
    model = YOLO(str(weights))
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    results = model.predict(
        source=[str(path) for path in eval_set.images],
        imgsz=640,
        conf=conf_floor,
        iou=nms_iou,
        max_det=300,
        batch=acc_batch,
        device=device,
        verbose=False,
        save=False,
    )
    try:
        records = detections_to_coco(results, eval_set.image_ids, class_ids=CLASS_IDS)
    except ValueError as exc:
        raise SystemExit(
            f"{rel_path(weights)} does not predict the four HK classes ({exc}). "
        ) from exc
    scored = score_run(gt_path, records, operating_conf=operating_conf, match_iou=match_iou)

    timing: dict[str, Any] = {}
    if timing_iters > 0:
        image = cv2.imread(str(eval_set.images[0]))
        timing = measure_latency(
            model,
            image,
            predict_kwargs={
                "imgsz": 640,
                "conf": operating_conf,
                "iou": nms_iou,
                "max_det": 300,
                "batch": 1,
                "device": device,
                "verbose": False,
                "save": False,
            },
            warmup=10,
            iterations=timing_iters,
        )

    image_scores = per_image_f1(
        build_gt_index(eval_set.gt),
        records,
        eval_set.image_ids,
        conf=operating_conf,
        iou=match_iou,
    )
    rows = classify(eval_set.gt, records, model=LABELS[variant], conf=operating_conf)

    del model
    torch.cuda.empty_cache()
    return {
        "variant": variant,
        "label": LABELS[variant],
        "weights": rel_path(weights),
        "params": parameters,
        "records": records,
        "ap": scored["ap"],
        "operating_point": scored["operating_point"],
        "per_class": scored["per_class"],
        "counts": scored["counts"],
        "timing": timing,
        "image_f1": image_scores,
        "errors": rows,
    }


def _paired_table(baseline: dict[str, float], modified: dict[str, float]) -> dict[str, Any]:
    image_ids = sorted(set(baseline) & set(modified))
    left = [baseline[image_id] for image_id in image_ids]
    right = [modified[image_id] for image_id in image_ids]
    differences = [b - a for a, b in zip(left, right)]
    n = len(differences)
    mean_difference = sum(differences) / n if n else 0.0
    sd = statistics_sd(differences)
    t_value, p_value = stats.ttest_rel(right, left) if n > 1 else (float("nan"), float("nan"))
    se = sd / math.sqrt(n) if n and sd else 0.0
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    return {
        "images": n,
        "mean_f1_baseline": round(sum(left) / n, 4) if n else None,
        "mean_f1_modified": round(sum(right) / n, 4) if n else None,
        "mean_difference": round(mean_difference, 4),
        "sd_difference": round(sd, 4),
        "ci95_low": round(mean_difference - critical * se, 4) if n > 1 else None,
        "ci95_high": round(mean_difference + critical * se, 4) if n > 1 else None,
        "t_statistic": round(float(t_value), 4) if n > 1 else None,
        "p_value": float(p_value) if n > 1 else None,
        "cohens_d": round(mean_difference / sd, 4) if sd else None,
        "modified_wins": sum(1 for value in differences if value > 0),
        "baseline_wins": sum(1 for value in differences if value < 0),
        "ties": sum(1 for value in differences if value == 0),
    }


def statistics_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def run_rq5(
    *,
    train: bool = False,
    evaluate: bool = True,
    variants: Sequence[str] = VARIANTS,
    epochs: int = 30,
    batch: int = 8,
    imgsz: int = 640,
    seed: int = 0,
    device: int | str | None = None,
    workers: int = 2,
    gt: str = "data/annotations/hk_test_coco.json",
    images: str = "data/hk/images/test",
    tag: str = "hk_test",
    conf: float = 0.25,
    nms_iou: float = 0.3,
    match_iou: float = 0.5,
    timing_iters: int = 100,
    config: str = "configs/models.yaml",
) -> int:
    cfg_dict = load_models_config(config)
    eval_cfg = cfg_dict.get("eval") or {}
    conf_floor = float(eval_cfg.get("conf_floor", 0.001))
    device = resolve_model(cfg_dict, "yolov8s").device if device is None else device

    yaml_path = dataset_yaml()
    print(f"dataset yaml: {rel_path(yaml_path)}")

    weights: dict[str, Path] = {}
    if train:
        for variant in variants:
            weights[variant] = train_variant(
                variant,
                data=yaml_path,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                seed=seed,
                device=device,
                workers=workers,
            )
            print(f"best weights: {rel_path(weights[variant])}")
    else:
        for variant in variants:
            candidate = REPO_ROOT / "results" / "runs" / f"rq5_{variant}" / "weights" / "best.pt"
            if not candidate.exists():
                raise SystemExit(f"missing {rel_path(candidate)}; run with --train first")
            weights[variant] = candidate

    if not evaluate:
        return 0

    eval_set = from_coco(gt, images, name=tag)
    gt_path = _resolve(gt)
    print(
        f"\ndataset   : {eval_set.name} — {len(eval_set.images)} images, "
        f"{len(eval_set.gt.get('annotations', []))} instances"
    )
    print(f"operating : conf {conf}, NMS IoU {nms_iou}, matching IoU {match_iou}")

    run_id = new_run_id("rq5_eval")
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for variant in variants:
        print(f"\n=== evaluating {LABELS[variant]} ({variant}) ===")
        result = evaluate_variant(
            variant,
            weights[variant],
            eval_set,
            gt_path,
            conf_floor=conf_floor,
            operating_conf=conf,
            nms_iou=nms_iou,
            match_iou=match_iou,
            acc_batch=batch,
            device=device,
            timing_iters=timing_iters,
        )
        results[variant] = result
        det_path = run_dir / f"{variant}_detections.json"
        det_path.write_text(json.dumps(result["records"]), encoding="utf-8")
        print(
            f"  mAP@50:95={result['ap']['mAP_50_95']}  mAP@50={result['ap']['mAP_50']}  "
            f"P={result['operating_point']['precision']} R={result['operating_point']['recall']} "
            f"F1={result['operating_point']['f1']}"
            + (f"  {result['timing']['latency_total_ms']:.2f} ms" if result["timing"] else "")
        )

    ablation = pd.DataFrame(
        [
            {
                "variant": result["variant"],
                "label": result["label"],
                "params": result["params"],
                "mAP_50_95": result["ap"]["mAP_50_95"],
                "mAP_50": result["ap"]["mAP_50"],
                "precision": result["operating_point"]["precision"],
                "recall": result["operating_point"]["recall"],
                "f1": result["operating_point"]["f1"],
                "latency_ms": result["timing"].get("latency_total_ms"),
                "fps": result["timing"].get("fps"),
                "images": result["counts"]["images"],
                "detections": result["counts"]["detections"],
                "weights": result["weights"],
            }
            for result in results.values()
        ]
    )

    per_class = pd.DataFrame(
        [
            {"variant": result["variant"], **row}
            for result in results.values()
            for row in result["per_class"]
        ]
    )

    if "baseline" in results and "p2" in results:
        paired = _paired_table(results["baseline"]["image_f1"], results["p2"]["image_f1"])
    else:
        paired = {"note": "a paired test needs both variants"}

    errors = pd.DataFrame(
        [
            row for result in results.values() for row in result["errors"]
        ]
    )
    error_counts = (
        errors.pivot_table(index="error_type", columns="model", values="image_id", aggfunc="count")
        .fillna(0)
        .astype(int)
        .reset_index()
        if not errors.empty
        else pd.DataFrame()
    )

    tables = REPO_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(tables / "rq5_ablation.csv", index=False)
    per_class.to_csv(tables / "rq5_per_class.csv", index=False)
    pd.DataFrame([paired]).to_csv(tables / "rq5_paired_test.csv", index=False)
    if not error_counts.empty:
        error_counts.to_csv(tables / "rq5_errors.csv", index=False)

    write_runcard(
        run_id,
        config={
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "seed": seed,
            "variants": list(variants),
            "conf": conf,
            "nms_iou": nms_iou,
            "match_iou": match_iou,
        },
        config_path=config,
        device=device,
        artifacts={
            "ablation": rel_path(tables / "rq5_ablation.csv"),
            "per_class": rel_path(tables / "rq5_per_class.csv"),
            "paired": rel_path(tables / "rq5_paired_test.csv"),
        },
        extra={
            "dataset": {"name": eval_set.name, "images": len(eval_set.images)},
            "ap": {variant: result["ap"] for variant, result in results.items()},
            "operating_point": {
                variant: result["operating_point"] for variant, result in results.items()
            },
            "paired": paired,
        },
    )

    print("\nablation:")
    print(ablation.drop(columns=["weights"]).to_string(index=False))
    if "mAP_50_95" in paired or "mean_difference" in paired:
        print("\npaired test (per-image F1, baseline vs P2):")
        print(pd.DataFrame([paired]).to_string(index=False))
    if not error_counts.empty:
        print("\nerror counts:")
        print(error_counts.to_string(index=False))
    print(f"\ntables : {rel_path(tables / 'rq5_ablation.csv')}")
    print(f"         {rel_path(tables / 'rq5_per_class.csv')}")
    print(f"         {rel_path(tables / 'rq5_paired_test.csv')}")
    print(f"runcard: {rel_path(run_dir / 'runcard.json')}")
    return 0
