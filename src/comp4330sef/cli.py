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
    score.add_argument("--all-classes", action="store_true", help="keep off-protocol classes in P/R/F1")

    selfcheck = subparsers.add_parser("selfcheck", help="validate our metrics against Ultralytics val")
    selfcheck.add_argument("--config", default="configs/models.yaml")
    selfcheck.add_argument("--model", default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m"])
    selfcheck.add_argument("--split", default="val", choices=["val", "train"])
    selfcheck.add_argument(
        "--no-reference", action="store_true", help="skip the Ultralytics cross-check"
    )

    rq1 = subparsers.add_parser("rq1", help="RQ1: accuracy/speed scaling across yolov8 n/s/m")
    rq1.add_argument("--config", default="configs/models.yaml")
    rq1.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s", "yolov8m"])
    rq1.add_argument("--data", default="datasets/coco8.yaml", help="Ultralytics dataset yaml")
    rq1.add_argument("--split", default="val")
    rq1.add_argument("--gt", default=None, help="COCO annotation json; overrides --data")
    rq1.add_argument("--images", default=None, help="image root for --gt")
    rq1.add_argument("--tag", default=None, help="dataset label used in file names")
    rq1.add_argument("--table", default=None)
    rq1.add_argument("--per-class-table", default=None)
    rq1.add_argument("--marginal-table", default=None)
    rq1.add_argument("--acc-batch", type=int, default=8)
    rq1.add_argument("--warmup", type=int, default=10)
    rq1.add_argument("--iters", type=int, default=100)
    rq1.add_argument("--all-classes", action="store_true", help="keep off-protocol classes in P/R/F1")

    rq2 = subparsers.add_parser("rq2", help="RQ2: confidence / NMS-IoU sweep across sizes")
    rq2.add_argument("--config", default="configs/models.yaml")
    rq2.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s", "yolov8m"])
    rq2.add_argument("--gt", default="data/annotations/hk_test_coco.json")
    rq2.add_argument("--images", default="data/hk/images/test")
    rq2.add_argument("--tag", default="hk_test")
    rq2.add_argument("--confs", nargs="+", type=float, default=None)
    rq2.add_argument("--nms-ious", nargs="+", type=float, default=None)
    rq2.add_argument("--match-iou", type=float, default=None)
    rq2.add_argument("--table", default=None)
    rq2.add_argument("--best-table", default=None)
    rq2.add_argument("--answer", default=None)
    rq2.add_argument("--acc-batch", type=int, default=8)

    rq3 = subparsers.add_parser("rq3", help="RQ3: TIDE decomposition and error taxonomy")
    rq3.add_argument("--gt", default="data/annotations/hk_test_coco.json")
    rq3.add_argument("--tag", default="hk_test")
    rq3.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s", "yolov8m"])
    rq3.add_argument("--nms-iou", type=float, default=0.3)
    rq3.add_argument("--conf", type=float, default=0.25)
    rq3.add_argument("--match-iou", type=float, default=0.5)

    models = subparsers.add_parser("models", help="RQ4: baseline vs P2 parameter and shape check")
    models.add_argument("--scale", default="s", choices=["n", "s", "m"])
    models.add_argument("--nc", type=int, default=4)
    models.add_argument("--imgsz", type=int, default=640)
    models.add_argument("--out", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "dummy":
        from comp4330sef.eval.dummy import run_dummy

        return run_dummy(args.config, args.model, args.limit, args.split)

    if args.command == "score":
        from comp4330sef.eval.score import format_result, score_run

        result = score_run(
            args.gt,
            args.pred,
            operating_conf=args.conf,
            match_iou=args.iou,
            restrict_to_gt_classes=not args.all_classes,
        )
        print(format_result(result))
        if not all((result.get("consistency") or {}).values()):
            print("\n[warn] P/R/F1 counts are not internally consistent")
        return 0

    if args.command == "selfcheck":
        from comp4330sef.eval.selfcheck import run_selfcheck

        return run_selfcheck(args.config, args.model, args.split, reference=not args.no_reference)

    if args.command == "rq1":
        from comp4330sef.eval.rq1 import run_rq1

        return run_rq1(
            config_path=args.config,
            models=tuple(args.models),
            data=args.data,
            split=args.split,
            gt=args.gt,
            images=args.images,
            tag=args.tag,
            table=args.table,
            per_class_table=args.per_class_table,
            marginal_table=args.marginal_table,
            acc_batch=args.acc_batch,
            warmup=args.warmup,
            iterations=args.iters,
            restrict_classes=not args.all_classes,
        )

    if args.command == "rq2":
        from comp4330sef.sweep.conf_iou import run_rq2

        return run_rq2(
            config_path=args.config,
            models=tuple(args.models),
            gt=args.gt,
            images=args.images,
            tag=args.tag,
            confs=args.confs,
            nms_ious=args.nms_ious,
            match_iou=args.match_iou,
            table=args.table,
            best_table=args.best_table,
            answer_path=args.answer,
            acc_batch=args.acc_batch,
        )

    if args.command == "rq3":
        from comp4330sef.analysis.tide_run import run_rq3

        return run_rq3(
            gt=args.gt,
            tag=args.tag,
            models=tuple(args.models),
            nms_iou=args.nms_iou,
            conf=args.conf,
            match_iou=args.match_iou,
        )

    if args.command == "models":
        from comp4330sef.models.build import run_compare

        return run_compare(scale=args.scale, nc=args.nc, imgsz=args.imgsz, out=args.out)

    raise SystemExit(f"unknown command {args.command!r}")