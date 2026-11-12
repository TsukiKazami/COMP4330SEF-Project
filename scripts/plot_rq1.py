"""RQ1 figure: accuracy/speed scaling of YOLOv8 n/s/m on the Hong Kong test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="results/tables/rq1_scaling.csv")
    parser.add_argument("--marginal", default="results/tables/rq1_marginal.csv")
    parser.add_argument("--out", default="results/figures/fig_rq1_scaling.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.table).sort_values("params").reset_index(drop=True)
    frame["params_m"] = frame["params"] / 1e6
    dataset = str(frame["dataset"].iloc[0])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    left.plot(frame["params_m"], frame["mAP_50_95"], marker="o", label="mAP@50:95")
    left.plot(frame["params_m"], frame["mAP_50"], marker="s", linestyle="--", label="mAP@50")
    left.set_xscale("log")
    left.set_xticks(frame["params_m"].tolist())
    left.set_xticklabels(
        [
            f"{str(row.model).replace('yolov8', '')}\n{row.params_m:.1f}M"
            for row in frame.itertuples(index=False)
        ]
    )
    left.minorticks_off()
    left.set_xlabel("checkpoint")
    left.set_ylabel("AP")
    left.set_title("(a) accuracy vs model size")
    left.grid(alpha=0.3, axis="y")
    left.legend(loc="upper left", fontsize=9)
    left.margins(x=0.25)

    note = []
    try:
        marginal = pd.read_csv(args.marginal)
    except FileNotFoundError:
        marginal = None
    if marginal is not None and not marginal.empty:
        for row in marginal.itertuples(index=False):
            note.append(
                f"{str(row.from_model).replace('yolov8', '')} to "
                f"{str(row.to_model).replace('yolov8', '')}: "
                f"+{row.delta_mAP_50_95:.4f} AP per +{row.delta_params_m:.1f}M "
                f"({row.map_per_mparam:.4f}/M)"
            )
        left.text(
            0.98,
            0.03,
            "\n".join(note),
            transform=left.transAxes,
            fontsize=8,
            va="bottom",
            ha="right",
        )

    right.plot(frame["latency_total_ms"], frame["mAP_50_95"], marker="o", color="tab:red")
    right.set_xlabel("latency (ms/frame, batch 1, fp32)")
    right.set_ylabel("mAP@50:95")
    right.set_title("(b) accuracy vs latency")
    right.grid(alpha=0.3)
    for row in frame.itertuples(index=False):
        right.annotate(
            f"{str(row.model).replace('yolov8', '')} ({row.fps:.0f} FPS)",
            (row.latency_total_ms, row.mAP_50_95),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    right.margins(x=0.18, y=0.15)

    header = args.title or (
        f"{dataset}: YOLOv8 n/s/m, 4 classes, {int(frame['images'].iloc[0])} images"
    )
    fig.suptitle(header, fontsize=11)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())