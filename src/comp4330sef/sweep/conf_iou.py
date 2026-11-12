from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from comp4330sef.config import REPO_ROOT, load_models_config, rel_path, resolve_model
from comp4330sef.data.eval_set import EvalSet, from_coco
from comp4330sef.eval.coco import evaluate_coco
from comp4330sef.eval.matching import build_gt_index, counts_at_conf
from comp4330sef.eval.results import detections_to_coco
from comp4330sef.eval.runcard import new_run_id, precision_check, write_runcard
from comp4330sef.eval.score import restrict_to_categories

DEFAULT_CONFS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70)
DEFAULT_NMS_IOUS = (0.3, 0.5, 0.7)
HK_NAMES = {"hk", "hk_test"}
REGRET_TOLERANCE = 0.01


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _paths(tag: str, table: str | None, best: str | None, answer: str | None) -> tuple[Path, Path, Path]:
    suffix = "" if tag.lower() in HK_NAMES else f"_{tag}"
    tables = REPO_ROOT / "results" / "tables"
    return (
        _resolve(table) if table else tables / f"rq2_sweep{suffix}.csv",
        _resolve(best) if best else tables / f"rq2_best{suffix}.csv",
        _resolve(answer) if answer else REPO_ROOT / "results" / f"rq2_answer{suffix}.json",
    )


def _cells(frame: pd.DataFrame, model: str, nms_iou: float, conf: float) -> pd.Series | None:
    block = frame[
        (frame["model"] == model)
        & np.isclose(frame["nms_iou"], nms_iou)
        & np.isclose(frame["conf"], conf)
    ]
    return None if block.empty else block.iloc[0]


def summarise(
    frame: pd.DataFrame,
    models: Sequence[str],
    tolerance: float = REGRET_TOLERANCE,
) -> dict[str, Any]:
    per_model: list[dict[str, Any]] = []
    for name in models:
        block = frame[frame["model"] == name]
        if block.empty:
            continue
        best = block.loc[block["f1"].idxmax()]
        per_model.append(
            {
                "model": name,
                "nms_iou": float(best["nms_iou"]),
                "conf": float(best["conf"]),
                "f1": round(float(best["f1"]), 4),
                "precision": round(float(best["precision"]), 4),
                "recall": round(float(best["recall"]), 4),
            }
        )

    grouped = frame.groupby(["nms_iou", "conf"], as_index=False)["f1"].mean()
    shared = grouped.loc[grouped["f1"].idxmax()]
    shared_iou = round(float(shared["nms_iou"]), 4)
    shared_conf = round(float(shared["conf"]), 4)

    regret: list[dict[str, Any]] = []
    for entry in per_model:
        cell = _cells(frame, entry["model"], shared_iou, shared_conf)
        f1 = float(cell["f1"]) if cell is not None else 0.0
        regret.append(
            {
                "model": entry["model"],
                "own_best_f1": entry["f1"],
                "f1_at_shared": round(f1, 4),
                "loss": round(entry["f1"] - f1, 4),
            }
        )
    worst = max((abs(item["loss"]) for item in regret), default=0.0)

    return {
        "per_model_best": per_model,
        "shared_best": {
            "nms_iou": shared_iou,
            "conf": shared_conf,
            "mean_f1": round(float(shared["f1"]), 4),
        },
        "shared_regret": regret,
        "max_regret": round(worst, 4),
        "regret_tolerance": tolerance,
        "one_config_serves_all": bool(worst <= tolerance),
    }


def _sweep_model(
    cfg: Any,
    eval_set: EvalSet,
    gt_path: Path,
    *,
    confs: Sequence[float],
    nms_ious: Sequence[float],
    match_iou: float,
    conf_floor: float,
    acc_batch: int,
    run_dir: Path,
    cfg_dict: dict[str, Any],
    config_path: str,
) -> list[dict[str, Any]]:
    gt_categories = {int(category["id"]) for category in eval_set.gt.get("categories", [])}
    gt_index = build_gt_index(eval_set.gt)

    torch.manual_seed(cfg.seed)
    model = YOLO(str(cfg.weights))
    precision = precision_check(model)
    if not precision["ok"]:
        raise SystemExit(f"{cfg.name}: expected fp32 weights, got {precision['actual']}")

    rows: list[dict[str, Any]] = []
    for nms_iou in nms_ious:
        results = model.predict(
            source=[str(path) for path in eval_set.images],
            imgsz=cfg.imgsz,
            conf=conf_floor,
            iou=nms_iou,
            max_det=cfg.max_det,
            batch=acc_batch,
            device=cfg.device,
            verbose=False,
            save=False,
        )
        if len(results) != len(eval_set.images):
            raise SystemExit(f"{cfg.name}: {len(results)} results for {len(eval_set.images)} images")

        records = detections_to_coco(results, eval_set.image_ids)
        cache = run_dir / f"det_{cfg.name}_nms{nms_iou:.2f}.json"
        cache.write_text(json.dumps(records), encoding="utf-8")

        evaluated = restrict_to_categories(records, gt_categories)
        ap = evaluate_coco(gt_path, records)["ap"]
        print(
            f"  nms_iou={nms_iou:.2f}: {len(records)} detections "
            f"({len(evaluated)} in protocol), mAP50={ap['mAP_50']}"
        )

        for conf in confs:
            operating = counts_at_conf(gt_index, evaluated, conf=conf, iou_thr=match_iou)
            rows.append(
                {
                    "dataset": eval_set.name,
                    "model": cfg.name,
                    "nms_iou": float(nms_iou),
                    "conf": float(conf),
                    "match_iou": float(match_iou),
                    "tp": operating["tp"],
                    "fp": operating["fp"],
                    "fn": operating["fn"],
                    "detections_kept": operating["detections_kept"],
                    "precision": operating["precision"],
                    "recall": operating["recall"],
                    "f1": operating["f1"],
                    "mAP_50": ap["mAP_50"],
                    "mAP_50_95": ap["mAP_50_95"],
                }
            )
        del results

    write_runcard(
        run_dir.name,
        config=cfg_dict,
        config_path=config_path,
        device=cfg.device,
        artifacts={"gt": rel_path(gt_path), "caches": rel_path(run_dir)},
        extra={
            "dataset": {
                "name": eval_set.name,
                "images": len(eval_set.images),
                "gt_instances": len(eval_set.gt.get("annotations", [])),
            },
            "grid": {"confs": list(confs), "nms_ious": list(nms_ious), "match_iou": match_iou},
            "precision": precision,
        },
    )

    del model
    torch.cuda.empty_cache()
    return rows


def run_rq2(
    *,
    config_path: str = "configs/models.yaml",
    models: Sequence[str] = ("yolov8n", "yolov8s", "yolov8m"),
    gt: str = "data/annotations/hk_test_coco.json",
    images: str = "data/hk/images/test",
    tag: str = "hk_test",
    confs: Sequence[float] | None = None,
    nms_ious: Sequence[float] | None = None,
    match_iou: float | None = None,
    table: str | None = None,
    best_table: str | None = None,
    answer_path: str | None = None,
    acc_batch: int = 8,
) -> int:
    cfg_dict = load_models_config(config_path)
    eval_cfg = cfg_dict.get("eval") or {}
    conf_floor = float(eval_cfg.get("conf_floor", 0.001))
    protocol_iou = float(eval_cfg.get("iou_match", 0.5)) if match_iou is None else float(match_iou)

    confs = tuple(confs) if confs else DEFAULT_CONFS
    nms_ious = tuple(nms_ious) if nms_ious else DEFAULT_NMS_IOUS
    for value in nms_ious:
        if not 0.0 < value <= 1.0:
            raise SystemExit(f"NMS IoU must be in (0, 1], got {value}")

    eval_set = from_coco(gt, images, name=tag)
    table_path, best_path, answer_file = _paths(eval_set.name, table, best_table, answer_path)

    print(
        f"dataset  : {eval_set.name} — {len(eval_set.images)} images, "
        f"{len(eval_set.gt.get('annotations', []))} instances"
    )
    print(f"grid     : {len(confs)} conf x {len(nms_ious)} NMS IoU, matching IoU {protocol_iou}")
    print(f"confs    : {list(confs)}")
    print(f"nms ious : {list(nms_ious)}")

    rows: list[dict[str, Any]] = []
    for name in models:
        cfg = resolve_model(cfg_dict, name)
        print(f"\n=== {cfg.name} ===")
        run_id = new_run_id(f"rq2_{cfg.name}")
        run_dir = REPO_ROOT / "results" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        gt_path = run_dir / "gt.json"
        gt_path.write_text(json.dumps(eval_set.gt, indent=2), encoding="utf-8")

        rows.extend(
            _sweep_model(
                cfg,
                eval_set,
                gt_path,
                confs=confs,
                nms_ious=nms_ious,
                match_iou=protocol_iou,
                conf_floor=conf_floor,
                acc_batch=acc_batch,
                run_dir=run_dir,
                cfg_dict=cfg_dict,
                config_path=config_path,
            )
        )

    frame = pd.DataFrame(rows).sort_values(["model", "nms_iou", "conf"]).reset_index(drop=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False)

    summary = summarise(frame, models)
    summary.update(
        {
            "dataset": eval_set.name,
            "confs": [float(value) for value in confs],
            "nms_ious": [float(value) for value in nms_ious],
            "match_iou": protocol_iou,
        }
    )
    answer_file.parent.mkdir(parents=True, exist_ok=True)
    answer_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_rows = [
        {**entry, "dataset": eval_set.name, "kind": "own_best"} for entry in summary["per_model_best"]
    ]
    best_rows.append(
        {
            "dataset": eval_set.name,
            "kind": "shared_best",
            "model": "all",
            "nms_iou": summary["shared_best"]["nms_iou"],
            "conf": summary["shared_best"]["conf"],
            "f1": summary["shared_best"]["mean_f1"],
            "precision": None,
            "recall": None,
        }
    )
    pd.DataFrame(best_rows).to_csv(best_path, index=False)

    print("\nper-model optimum:")
    for entry in summary["per_model_best"]:
        print(
            f"  {entry['model']:<9} iou={entry['nms_iou']:.2f} conf={entry['conf']:.2f}  F1={entry['f1']:.4f}"
        )
    shared = summary["shared_best"]
    print(
        f"\nshared optimum : iou={shared['nms_iou']:.2f} conf={shared['conf']:.2f}  "
        f"mean F1={shared['mean_f1']:.4f}"
    )
    print("regret of using the shared configuration:")
    for item in summary["shared_regret"]:
        print(
            f"  {item['model']:<9} own={item['own_best_f1']:.4f} "
            f"shared={item['f1_at_shared']:.4f} loss={item['loss']:+.4f}"
        )
    verdict = (
        "yes — one configuration serves all three sizes"
        if summary["one_config_serves_all"]
        else "no — the optimum shifts with model size"
    )
    print(f"\nRQ2 answer     : {verdict} (max regret {summary['max_regret']:.4f}, tolerance {REGRET_TOLERANCE})")
    print(f"\ntables         : {rel_path(table_path)}")
    print(f"                 {rel_path(best_path)}")
    print(f"answer         : {rel_path(answer_file)}")
    return 0