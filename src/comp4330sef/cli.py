"""Single entry point: ``uv run comp4330 <command>``."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comp4330",
        description="YOLOv8 evaluation on Hong Kong street scenes",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dummy = subparsers.add_parser("dummy", help="Phase 0 end-to-end shape check on COCO8")
    dummy.add_argument("--config", default="configs/models.yaml")
    dummy.add_argument("--model", default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m"])
    dummy.add_argument("--split", default="val", choices=["val", "train"])
    dummy.add_argument("--limit", type=int, default=4)

    score = subparsers.add_parser("score", help="AP and P/R/F1 from a GT json and a detections json")
    score.add_argument("--gt", required=True, help="COCO annotation json")
    score.add_argument("--pred", required=True, help="COCO detection results json")
    score.add_argument("--conf", type=float, default=0.25, help="operating point for P/R/F1")
    score.add_argument("--iou", type=float, default=0.5, help="matching IoU for P/R/F1")

    selfcheck = subparsers.add_parser("selfcheck", help="validate our metrics against Ultralytics val")
    selfcheck.add_argument("--config", default="configs/models.yaml")
    selfcheck.add_argument("--model", default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m"])
    selfcheck.add_argument("--split", default="val", choices=["val", "train"])
    selfcheck.add_argument(
        "--no-reference", action="store_true", help="skip the Ultralytics cross-check"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "dummy":
        from comp4330sef.eval.dummy import run_dummy

        return run_dummy(args.config, args.model, args.limit, args.split)

    if args.command == "score":
        from comp4330sef.eval.score import format_result, score_run

        result = score_run(args.gt, args.pred, operating_conf=args.conf, match_iou=args.iou)
        print(format_result(result))
        consistency = result.get("consistency") or {}
        if not all(consistency.values()):
            print("\n[warn] P/R/F1 counts are not internally consistent")
        return 0

    if args.command == "selfcheck":
        from comp4330sef.eval.selfcheck import run_selfcheck

        return run_selfcheck(args.config, args.model, args.split, reference=not args.no_reference)

    raise SystemExit(f"unknown command {args.command!r}")