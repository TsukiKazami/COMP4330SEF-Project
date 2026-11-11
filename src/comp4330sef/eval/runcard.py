from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from comp4330sef.config import REPO_ROOT, rel_path

RUNS_DIR = REPO_ROOT / "results" / "runs"

_PACKAGES = (
    "torch",
    "torchvision",
    "ultralytics",
    "pycocotools",
    "tidecv",
    "numpy",
    "scipy",
    "pandas",
    "pyyaml",
)


def git_state() -> dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return proc.stdout.strip()

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "short_sha": sha[:7] if sha != "unknown" else sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) and status != "unknown",
    }


def _cuda_index(device: int | str) -> int:
    text = str(device).split(":")[-1]
    return int(text) if text.isdigit() else 0


def device_info(device: int | str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "requested": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "precision": "fp32",
    }
    if info["cuda_available"]:
        index = _cuda_index(device)
        if index < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(index)
            info.update(
                {
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 1024**3, 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return info


def package_versions(names: tuple[str, ...] = _PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def model_dtype(yolo_model: Any) -> str:
    parameter = next(yolo_model.model.parameters())
    return str(parameter.dtype).replace("torch.", "")


def precision_check(yolo_model: Any, expected: str = "float32") -> dict[str, Any]:
    actual = model_dtype(yolo_model)
    return {"expected": expected, "actual": actual, "ok": actual == expected}


def new_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{git_state()['short_sha']}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel_path(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_runcard(
    run_id: str,
    *,
    config: dict[str, Any],
    config_path: str | Path,
    device: int | str,
    artifacts: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    card = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "config_path": rel_path(config_path),
        "config": config,
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": package_versions(),
        },
        "device": device_info(device),
        "artifacts": artifacts or {},
        "extra": extra or {},
    }

    path = run_dir / "runcard.json"
    path.write_text(json.dumps(_jsonable(card), indent=2, ensure_ascii=False), encoding="utf-8")
    return path