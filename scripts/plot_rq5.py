"""RQ5 figure: the P2 ablation and the reason it lost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from turtle import left

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["baseline", "p2"]
COLOURS = {"baseline": "tab:blue", "p2": "tab:orange"}


def _runcard() -> dict:
    cards = sorted(Path("results/runs").glob("rq5_eval_*/runcard.json"))
    if not cards:
        raise SystemExit("no results/runs/rq5_eval_*/runcard.json — run rq5 --evaluate first")
    return json.loads(cards[-1].read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/figures/fig_rq5_ablation.png")
    args = parser.parse_args()

    card = _runcard()
    ap = card["extra"]["ap"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # (a) AP by size — where P2 was supposed to help
    metrics = [
        ("AP_small", "small"),
        ("AP_medium", "medium"),
        ("AP_large", "large"),
        ("mAP_50_95", "all"),
    ]
    width = 0.38
    for position, variant in enumerate(ORDER):
        values = [float(ap[variant][key]) for key, _ in metrics]
        offsets = [index - width for index in range(len(metrics))] if position == 0 else [
            index for index in range(len(metrics))
        ]
        bars = left.bar(offsets, values, width=width, label=variant, color=COLOURS[variant])
        for bar, value in zip(bars, values):
            left.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
            )
    left.set_xticks(range(len(metrics)))
    left.set_xticklabels([label for _, label in metrics])
    left.set_ylabel("AP")
    left.set_title("(a) AP by object size")
    left.set_ylim(0, 1.02)
    left.grid(alpha=0.3, axis="y")
    left.legend(fontsize=8)

    # (b) validation curve — needs more epochs, not a better idea
    for variant in ORDER:
        path = Path(f"results/runs/rq5_{variant}/results.csv")
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.columns = [name.strip() for name in frame.columns]
        right.plot(
            frame["epoch"],
            frame["metrics/mAP50-95(B)"],
            marker="o",
            markersize=3,
            label=variant,
            color=COLOURS[variant],
        )
    right.set_xlabel("epoch")
    right.set_ylabel("validation mAP@50:95")
    right.set_title("(b) why: a cold start that has not converged")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    right.margins(x=0.02)

    paired_meta = card.get("extra", {}).get("paired", {})
    note = ""
    if paired_meta.get("mean_difference") is not None:
        note = (
            f"paired per-image F1: {paired_meta['mean_difference']:+.4f} "
            f"[{paired_meta['ci95_low']:+.4f}, {paired_meta['ci95_high']:+.4f}], "
            f"p={paired_meta['p_value']:.2g}, "
            f"wins {int(paired_meta['modified_wins'])}/{int(paired_meta['baseline_wins'])}"
        )
    epochs = card["config"].get("epochs", 30)
    fig.suptitle(f"hk_test, 4 classes, {epochs} epochs, T4 — {note}", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
