from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def rel_path(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class EvalConfig:
    name: str
    weights: Path
    imgsz: int
    conf: float
    iou: float
    half: bool
    max_det: int
    batch: int
    device: int | str
    seed: int

    @property
    def device_str(self) -> str:
        return f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": rel_path(self.weights),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "half": self.half,
            "max_det": self.max_det,
            "batch": self.batch,
            "device": self.device,
            "device_str": self.device_str,
            "seed": self.seed,
        }


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    with resolved.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{resolved} must contain a mapping at the top level")
    return data


def load_models_config(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    return load_yaml(path)


def resolve_model(cfg: dict[str, Any], name: str) -> EvalConfig:
    models = cfg.get("models") or {}
    if name not in models:
        raise KeyError(f"unknown model {name!r}; available: {sorted(models)}")

    merged = {**(cfg.get("defaults") or {}), **models[name]}
    weights = Path(merged["weights"])
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    if not weights.exists():
        raise FileNotFoundError(
            f"{rel_path(weights)} is missing, run: uv run python scripts/fetch_weights.py"
        )

    return EvalConfig(
        name=name,
        weights=weights,
        imgsz=int(merged["imgsz"]),
        conf=float(merged["conf"]),
        iou=float(merged["iou"]),
        half=bool(merged["half"]),
        max_det=int(merged["max_det"]),
        batch=int(merged["batch"]),
        device=merged["device"],
        seed=int(merged["seed"]),
    )