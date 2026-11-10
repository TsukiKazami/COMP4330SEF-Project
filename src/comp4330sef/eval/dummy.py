from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from comp4330sef.config import REPO_ROOT, load_models_config, rel_path, resolve_model
from comp4330sef.eval.results import detections_to_coco, image_id_from_stem, validate_records
from comp4330sef.eval.runcard import new_run_id, write_runcard


def _sample_images(data: dict[str, Any], split: str, limit: int) -> list[Path]:
    entry = data.get(split)
    if entry is None:
        raise SystemExit(f"dataset {data.get('path')!r} has no {split!r} split")
    entries = entry if isinstance(entry, (list, tuple)) else [entry]

    files: list[Path] = []
    for item in entries:
        path = Path(item)
        if path.is_dir():
            files.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            )
        elif path.is_file():
            files.append(path)
    if not files:
        raise SystemExit(f"no images found for split {split!r}")
    return files[:limit]


def run_dummy(config_path: str, model_name: str, limit: int, split: str) -> int:
    cfg_dict = load_models_config(config_path)
    cfg = resolve_model(cfg_dict, model_name)

    dummy_cfg = cfg_dict.get("dummy") or {}
    data_name = dummy_cfg.get("data", "coco8.yaml")
    data = check_det_dataset(data_name)
    images = _sample_images(data, split, limit)

    model = YOLO(str(cfg.weights))
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        half=cfg.half,
        max_det=cfg.max_det,
        batch=cfg.batch,
        device=cfg.device,
        verbose=False,
        save=False,
    )
    if len(results) != len(images):
        raise SystemExit(f"expected {len(images)} results, got {len(results)}")

    image_ids: list[int] = []
    for index, path in enumerate(images):
        image_id = image_id_from_stem(path)
        image_ids.append(image_id if image_id is not None else index)

    records = detections_to_coco(results, image_ids)
    stats = validate_records(records)

    stage: dict[str, list[float]] = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
    for result in results:
        speed = result.speed or {}
        for key, source in (
            ("preprocess_ms", "preprocess"),
            ("inference_ms", "inference"),
            ("postprocess_ms", "postprocess"),
        ):
            if source in speed:
                stage[key].append(float(speed[source]))
    timing = {key: round(sum(values) / len(values), 2) for key, values in stage.items() if values}

    run_id = new_run_id(f"phase0_dummy_{cfg.name}")
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    detections_path = run_dir / "detections.json"
    detections_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    card_path = write_runcard(
        run_id,
        config=cfg_dict,
        config_path=config_path,
        device=cfg.device,
        artifacts={"detections": rel_path(detections_path)},
        extra={
            "model": cfg.to_dict(),
            "dataset": {
                "name": data_name,
                "split": split,
                "images": [p.name for p in images],
            },
            "detection_stats": stats,
            "mean_stage_ms": timing,
        },
    )

    print(f"\nrun_id     : {run_id}")
    print(f"model      : {cfg.name}  ({rel_path(cfg.weights)})")
    print(f"dataset    : {data_name} [{split}]  {len(images)} images")
    print(
        f"params     : imgsz={cfg.imgsz} conf={cfg.conf} iou={cfg.iou} "
        f"half={cfg.half} batch={cfg.batch} device={cfg.device_str}"
    )
    print(f"detections : {stats['records']} records (invariant problems: {stats['problems']})")
    if timing:
        print("stage cost : " + "  ".join(f"{key}={value:.2f}ms" for key, value in timing.items()))

    print("\nper image:")
    for result, path in zip(results, images):
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        print(f"  {path.name:<28} {count:>3} boxes")

    print(f"\nruncard    : {rel_path(card_path)}")
    print(f"detections : {rel_path(detections_path)}")

    if stats["problems"]:
        print("\nFAIL: detection records violate COCO invariants")
        return 1
    print("\nOK: pipeline shape verified")
    return 0