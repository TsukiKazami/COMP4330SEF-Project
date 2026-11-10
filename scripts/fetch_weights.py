"""Phase 0: fetch the three COCO checkpoints and record their fingerprints."""

import hashlib
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

# Official Ultralytics parameter counts (docs: YOLOv8 n/s/m).
EXPECTED = {"yolov8n": 3_157_200, "yolov8s": 11_166_560, "yolov8m": 25_902_624}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {}
for name, expected in EXPECTED.items():
    path = WEIGHTS / f"{name}.pt"
    model = YOLO(str(path))  # downloads into weights/ when missing
    params = sum(p.numel() for p in model.model.parameters())
    manifest[name] = {
        "file": f"weights/{name}.pt",
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "params": params,
        "params_expected": expected,
        "params_match": abs(params - expected) <= 1000,
    }
    flag = "OK" if manifest[name]["params_match"] else "MISMATCH"
    print(f"{name}: params={params:,} [{flag}] size={path.stat().st_size/1e6:.1f}MB")

(WEIGHTS / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print("\nwrote weights/manifest.json")