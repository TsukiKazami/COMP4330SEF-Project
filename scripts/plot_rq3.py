"""RQ3 figure: TIDE dAP decomposition and the error mix by object scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["Miss", "Cls", "Bkg", "Loc", "Dupe"]
LABEL = {
    "Miss": "missed (FN)",
    "Cls": "class",
    "Bkg": "background",
    "Loc": "localisation",
    "Dupe": "duplicate",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tide", default="results/tables/rq3_tide.csv")
    parser.add_argument("--taxonomy", default="results/tables/rq3_error_taxonomy.csv")
    parser.add_argument("--out", default="results/figures/fig_rq3_errors.png")
    args = parser.parse_args()

    tide = pd.read_csv(args.tide)
    taxonomy = pd.read_csv(args.taxonomy)

    preferred = ["yolov8n", "yolov8s", "yolov8m"]
    present = [name for name in preferred if name in set(tide["model"])]
    present += sorted(set(tide["model"]) - set(present))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.1))

    # (a) TIDE dAP per error type, one bar per model
    types = [name for name in ORDER if name in set(tide["error_type"])]
    width = 0.8 / max(len(present), 1)
    for position, model in enumerate(present):
        block = tide[tide["model"] == model].set_index("error_type")
        values = [float(block.loc[name, "dap"]) if name in block.index else 0.0 for name in types]
        offsets = [index - 0.4 + width * (position + 0.5) for index in range(len(types))]
        left.bar(offsets, values, width=width, label=model)
    left.set_xticks(range(len(types)))
    left.set_xticklabels([LABEL.get(name, name) for name in types], fontsize=9)
    left.set_ylabel("TIDE dAP (percentage points)")
    left.set_title("(a) what each error type costs")
    left.grid(alpha=0.3, axis="y")
    left.legend(fontsize=8)

    # (b) error counts by scale bucket, pooled over models
    buckets = [name for name in ("small", "medium", "large") if name in set(taxonomy["scale_bucket"])]
    table = (
        pd.crosstab(taxonomy["scale_bucket"], taxonomy["error_type"])
        .reindex(index=buckets, columns=types, fill_value=0)
    )
    bottom = [0] * len(table)
    for name in types:
        right.bar(table.index, table[name], bottom=bottom, label=LABEL.get(name, name))
        bottom = [total + value for total, value in zip(bottom, table[name])]
    right.set_ylabel("error instances (operating point, 3 models)")
    right.set_title("(b) which sizes fail")
    right.grid(alpha=0.3, axis="y")
    if len(table):
        right.set_ylim(0, max(bottom) * 1.05)

    fig.suptitle(
        "hk_test, 4 classes, NMS IoU 0.3 / conf 0.25 — "
        "misses dominate the count, class confusion dominates the dAP",
        fontsize=10,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
