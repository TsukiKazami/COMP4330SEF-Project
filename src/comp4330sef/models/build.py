from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from comp4330sef.config import REPO_ROOT, rel_path

MODEL_DIR = REPO_ROOT / "configs" / "models"
P2_CONFIG = MODEL_DIR / "yolov8s-p2.yaml"


def _package_yaml(name: str) -> Path:
    import ultralytics

    return Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / name


def build_baseline(scale: str = "s", nc: int = 4) -> Any:
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    config = yaml_model_load(str(_package_yaml("yolov8.yaml")))
    config["scale"] = scale
    config["nc"] = nc
    model = DetectionModel(config, ch=3, nc=nc, verbose=False)
    model.eval()
    return model


def build_p2(nc: int = 4) -> Any:
    from ultralytics import YOLO

    if not P2_CONFIG.exists():
        raise SystemExit(f"missing {rel_path(P2_CONFIG)}")
    model = YOLO(str(P2_CONFIG)).model
    if int(model.yaml.get("nc", 0)) != nc:
        raise SystemExit(f"{rel_path(P2_CONFIG)} declares nc={model.yaml.get('nc')}, expected {nc}")
    model.eval()
    return model


def _flops(model: Any, imgsz: int) -> float | None:
    try:
        from ultralytics.utils.torch_utils import get_flops
    except Exception:
        return None
    try:
        return round(float(get_flops(model, imgsz)), 1)
    except Exception:
        return None


def describe(model: Any, *, variant: str, scale: str, nc: int, imgsz: int = 640) -> dict[str, Any]:
    params = sum(parameter.numel() for parameter in model.parameters())
    stride = [int(value) for value in model.stride.tolist()]
    with torch.no_grad():
        output = model(torch.zeros(1, 3, imgsz, imgsz))

    prediction = None
    features: list[torch.Tensor] = []
    if isinstance(output, tuple) and len(output) >= 2:
        prediction, extras = output[0], output[1]
        if isinstance(extras, dict):
            features = list(extras.get("feats") or [])
    elif isinstance(output, dict):
        features = list(output.get("feats") or [])
        prediction = output.get("pred")
    else:
        prediction = output

    spatial = [f"{int(tensor.shape[-1])}x{int(tensor.shape[-2])}" for tensor in features]
    return {
        "variant": variant,
        "scale": scale,
        "nc": nc,
        "params": params,
        "gflops_640": _flops(model, imgsz),
        "levels": len(stride),
        "strides": "/".join(str(value) for value in stride),
        "anchors": int(prediction.shape[-1]) if prediction is not None else None,
        "head_shapes": "/".join(spatial),
    }


def run_compare(
    *,
    scale: str = "s",
    nc: int = 4,
    imgsz: int = 640,
    out: str | None = None,
) -> int:
    print(f"baseline : yolov8{scale}.yaml  (nc={nc})")
    baseline = describe(build_baseline(scale, nc), variant=f"yolov8{scale}", scale=scale, nc=nc, imgsz=imgsz)
    print(f"interv.  : {rel_path(P2_CONFIG)}  (nc={nc})")
    modified = describe(build_p2(nc), variant=f"yolov8{scale}-p2", scale=scale, nc=nc, imgsz=imgsz)

    if baseline["levels"] != 3:
        raise SystemExit(f"baseline should produce 3 levels, got {baseline['levels']}")
    if modified["levels"] != 4:
        raise SystemExit(f"P2 model should produce 4 levels, got {modified['levels']}")
    if modified["strides"] != "4/8/16/32":
        raise SystemExit(f"P2 strides are {modified['strides']}, expected 4/8/16/32")

    delta = modified["params"] - baseline["params"]
    frame = pd.DataFrame([baseline, modified])
    frame["delta_params"] = [0, delta]
    frame["delta_params_pct"] = [0.0, round(100.0 * delta / baseline["params"], 2)]
    if baseline["gflops_640"] and modified["gflops_640"]:
        frame["delta_gflops"] = [0.0, round(modified["gflops_640"] - baseline["gflops_640"], 1)]
        frame["delta_gflops_pct"] = [
            0.0,
            round(100.0 * (modified["gflops_640"] - baseline["gflops_640"]) / baseline["gflops_640"], 2),
        ]

    path = Path(out) if out else REPO_ROOT / "results" / "tables" / "rq4_params.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    columns = [
        "variant",
        "params",
        "delta_params",
        "delta_params_pct",
        "gflops_640",
        "delta_gflops",
        "delta_gflops_pct",
        "levels",
        "strides",
        "anchors",
    ]
    print("\n" + frame[columns].to_string(index=False))
    print("\ndetection levels:")
    for row in frame.itertuples(index=False):
        print(f"  {row.variant:<14} strides {row.strides:<12} head {row.head_shapes}")
    pct_params = 100.0 * delta / baseline["params"]
    message = f"\nP2 delta: {delta:+,} parameters ({pct_params:+.2f}%), one extra stride-4 level"
    if baseline["gflops_640"] and modified["gflops_640"]:
        delta_flops = modified["gflops_640"] - baseline["gflops_640"]
        message += (
            f", {delta_flops:+.1f} GFLOPs "
            f"({100.0 * delta_flops / baseline['gflops_640']:+.1f}%)"
        )
    print(message)
    print("the P2 yaml rebalances the neck widths, so parameter count goes DOWN while")
    print("compute goes UP through the 160x160 level: the intervention buys resolution,")
    print("not capacity — say this when interpreting RQ5")
    print(f"\ntable : {rel_path(path)}")
    print("note  : the baseline is built from the stock yaml with nc=4, i.e. the "
          "architecture RQ5 fine-tunes; COCO weights are transferred with model.load()")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RQ4: baseline vs P2 budget check")
    parser.add_argument("--scale", default="s", choices=["n", "s", "m"])
    parser.add_argument("--nc", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    return run_compare(scale=args.scale, nc=args.nc, imgsz=args.imgsz, out=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
