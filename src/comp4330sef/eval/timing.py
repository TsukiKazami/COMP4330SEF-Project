from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(
    yolo_model: Any,
    image: np.ndarray,
    *,
    predict_kwargs: dict[str, Any],
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, float | int]:
    for _ in range(warmup):
        yolo_model.predict(source=image, **predict_kwargs)

    stages: dict[str, list[float]] = {"preprocess": [], "inference": [], "postprocess": []}
    totals: list[float] = []
    walls: list[float] = []

    for _ in range(iterations):
        _synchronize()
        start = time.perf_counter()
        results = yolo_model.predict(source=image, **predict_kwargs)
        _synchronize()
        walls.append((time.perf_counter() - start) * 1000.0)

        speed = results[0].speed or {}
        preprocess = float(speed.get("preprocess", 0.0))
        inference = float(speed.get("inference", 0.0))
        postprocess = float(speed.get("postprocess", 0.0))
        stages["preprocess"].append(preprocess)
        stages["inference"].append(inference)
        stages["postprocess"].append(postprocess)
        totals.append(preprocess + inference + postprocess)

    if not totals or max(totals) <= 0:
        raise SystemExit("no timing data: result.speed was empty for every iteration")

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _std(values: list[float]) -> float:
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    total_mean = _mean(totals)
    wall_mean = _mean(walls)

    return {
        "warmup": int(warmup),
        "iterations": int(iterations),
        "latency_total_ms": round(total_mean, 3),
        "latency_total_std_ms": round(_std(totals), 3),
        "latency_total_p50_ms": round(float(np.percentile(totals, 50)), 3),
        "latency_total_p95_ms": round(float(np.percentile(totals, 95)), 3),
        "latency_preprocess_ms": round(_mean(stages["preprocess"]), 3),
        "latency_inference_ms": round(_mean(stages["inference"]), 3),
        "latency_postprocess_ms": round(_mean(stages["postprocess"]), 3),
        "wall_mean_ms": round(wall_mean, 3),
        "wall_std_ms": round(_std(walls), 3),
        "fps": round(1000.0 / total_mean, 2) if total_mean > 0 else 0.0,
        "fps_wall": round(1000.0 / wall_mean, 2) if wall_mean > 0 else 0.0,
    }