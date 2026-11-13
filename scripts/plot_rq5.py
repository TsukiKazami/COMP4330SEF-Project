"""RQ5 figure: the P2 ablation and the reason it lost.

Panel (a) splits AP by object size (from the evaluation runcard), which is where
the intervention was supposed to pay off. Panel (b) shows the validation curve
of both fine-tunes: the P2 model starts from zero because its new branch and
head are randomly initialised, and is still climbing when the budget runs out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    left.set_title("(a) the intervention misses its target")
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

    paired = Path("results/tables/rq5_paired_test.csv")
    note = ""
    if paired.exists():
        row = pd.read_csv(paired).iloc[0]
        note = (
            f"paired per-image F1: {row['mean_difference']:+.4f} "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}], "
            f"p={row['p_value']:.1e}, wins {int(row['modified_wins'])}/{int(row['baseline_wins'])}"
        )
    fig.suptitle(f"hk_test, 4 classes, 30 epochs, T4 — P2 does not beat the baseline. {note}", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
