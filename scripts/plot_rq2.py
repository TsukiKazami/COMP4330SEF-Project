"""RQ2 figure: F1 against the confidence threshold, per model and NMS IoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="results/tables/rq2_sweep.csv")
    parser.add_argument("--answer", default="results/rq2_answer.json")
    parser.add_argument("--out", default="results/figures/fig_rq2_sweep.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.table)
    preferred = ["yolov8n", "yolov8s", "yolov8m"]
    present = list(frame["model"].unique())
    models = [name for name in preferred if name in present] + sorted(
        name for name in present if name not in preferred
    )
    dataset = str(frame["dataset"].iloc[0])

    answer = None
    path = Path(args.answer)
    if path.exists():
        answer = json.loads(path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 3.9), sharey=True)
    if len(models) == 1:
        axes = [axes]

    own_best = {}
    if answer:
        own_best = {entry["model"]: entry for entry in answer["per_model_best"]}

    for ax, model in zip(axes, models):
        block = frame[frame["model"] == model]
        for iou in sorted(block["nms_iou"].unique()):
            line = block[block["nms_iou"] == iou].sort_values("conf")
            ax.plot(
                line["conf"],
                line["f1"],
                marker="o",
                markersize=3.5,
                label=f"NMS IoU {iou:.1f}",
            )
        best = own_best.get(model)
        if best:
            ax.plot(
                best["conf"],
                best["f1"],
                marker="*",
                markersize=13,
                color="black",
                linestyle="none",
                label="own optimum",
            )
        if answer:
            shared = answer["shared_best"]
            label = f"shared conf={shared['conf']:.2f}" if ax is axes[0] else None
            ax.axvline(shared["conf"], color="grey", linestyle=":", linewidth=1.2, label=label)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("confidence threshold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("F1 at the operating point")
    axes[0].legend(fontsize=7, loc="upper right")

    if answer:
        verdict = "one config serves all" if answer["one_config_serves_all"] else "optimum shifts with size"
        subtitle = (
            f"shared optimum: NMS IoU {answer['shared_best']['nms_iou']:.1f}, "
            f"conf {answer['shared_best']['conf']:.2f} — {verdict} "
            f"(max regret {answer['max_regret']:.3f})"
        )
    else:
        subtitle = "shared optimum unavailable (run rq2 first)"
    fig.suptitle(f"{dataset}: {subtitle}", fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())