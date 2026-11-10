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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dummy":
        from comp4330sef.eval.dummy import run_dummy

        return run_dummy(args.config, args.model, args.limit, args.split)
    raise SystemExit(f"unknown command {args.command!r}")