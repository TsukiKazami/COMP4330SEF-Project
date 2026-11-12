from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Draft YOLO labels with a model outside the evaluated family")
    parser.add_argument("--set", default="data/annotations/test_set.csv")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--out", default="data/labels_draft")
    parser.add_argument("--model", default="weights/yolo11x.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    model_path = resolve(args.model)
    if "yolov8" in model_path.stem.lower():
        raise SystemExit("refusing to pre-annotate with an evaluated model; use yolo11x or similar")

    selection = pd.read_csv(resolve(args.set))
    image_root = resolve(args.images)
    paths = [image_root / str(name) for name in selection["file_name"]]
    absent = [str(path) for path in paths if not path.exists()]
    if absent:
        raise SystemExit(f"{len(absent)} images missing, first: {absent[0]}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    names = model.names
    if len(names) != 80 or str(names[0]).lower() != "person":
        raise SystemExit(f"{model_path.name}: expected COCO 80 names, got {len(names)}")

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=[str(path) for path in paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        verbose=False,
        save=False,
    )
    if len(results) != len(paths):
        raise SystemExit(f"{len(results)} results for {len(paths)} images")

    rows: list[dict[str, object]] = []
    for result, path in zip(results, paths):
        height, width = result.orig_shape
        boxes = result.boxes
        lines: list[str] = []
        confidences: list[float] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, label in zip(xyxy, scores, classes):
                centre_x = ((x1 + x2) / 2) / width
                centre_y = ((y1 + y2) / 2) / height
                box_w = (x2 - x1) / width
                box_h = (y2 - y1) / height
                lines.append(f"{label} {centre_x:.6f} {centre_y:.6f} {box_w:.6f} {box_h:.6f}")
                confidences.append(float(score))
        (out_dir / f"{path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        rows.append(
            {
                "image_id": int(path.stem),
                "file_name": path.name,
                "boxes": len(lines),
                "classes": len(set(classes)) if boxes is not None and len(boxes) else 0,
                "mean_conf": round(sum(confidences) / len(confidences), 4) if confidences else None,
                "min_conf": round(min(confidences), 4) if confidences else None,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    total_boxes = int(summary["boxes"].sum())
    empty = int((summary["boxes"] == 0).sum())
    print(f"images      : {len(summary)}")
    print(f"boxes       : {total_boxes}  (mean {total_boxes / max(len(summary), 1):.1f} per image)")
    print(f"empty drafts: {empty}")
    print(f"images with detections: {int(summary['boxes'].gt(0).sum())}")
    print(f"out         : {out_dir.relative_to(ROOT).as_posix()}/<image_id>.txt")
    print(f"summary     : {(out_dir / 'summary.csv').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
