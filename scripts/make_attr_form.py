from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCALE = ("small", "medium", "large")
OCCLUSION = ("clear", "partial", "heavy")
ILLUMINATION = ("day", "dim", "night")
TEXT = ("low", "medium", "high")
ATTR_COLUMNS = ("scale_bucket", "occlusion_level", "illumination", "text_density")

COLUMNS = [
    "image_id", "file_name", "region", "year",
    "scale_bucket", "occlusion_level", "illumination", "text_density",
    "annotator", "note",
]


def allocate(total: int, counts: dict[str, int]) -> dict[str, int]:
    keys = sorted(counts)
    capacity = sum(counts[key] for key in keys)
    target = min(total, capacity)
    seats = {key: 0 for key in keys}
    quota = {key: target * counts[key] / capacity for key in keys}
    while sum(seats.values()) < target:
        open_keys = [key for key in keys if seats[key] < counts[key]]
        if not open_keys:
            break
        best = max(open_keys, key=lambda key: (quota[key] - seats[key], counts[key] - seats[key], key))
        seats[best] += 1
    return seats


def display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the attribute annotation forms for the test set")
    parser.add_argument("--set", default="data/annotations/test_set.csv")
    parser.add_argument("--attributes", default="data/annotations/attributes.csv")
    parser.add_argument("--out-dir", default="data/annotations")
    parser.add_argument("--annotators", type=int, default=6)
    parser.add_argument("--double", type=int, default=40)
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    selection = pd.read_csv(resolve(args.set))
    pool = pd.read_csv(resolve(args.attributes))
    pool["year"] = pool["captured_at_utc"].astype(str).str.slice(0, 4)

    merged = selection[["image_id"]].merge(
        pool[["image_id", "file_name", "region", "year"]], on="image_id", how="left"
    )
    missing = merged["file_name"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} selected image_id not found in the attributes pool")
    merged = merged.sort_values("image_id").reset_index(drop=True)

    names = [f"a{index + 1}" for index in range(args.annotators)]
    merged["annotator"] = [names[position % len(names)] for position in range(len(merged))]

    counts = {region: int(count) for region, count in merged.groupby("region").size().items()}
    seats = allocate(args.double, counts)
    double_ids: list[int] = []
    for region in sorted(seats):
        rows = merged[merged["region"] == region].sort_values("image_id").reset_index(drop=True)
        count = min(seats[region], len(rows))
        if count == 0:
            continue
        if count == 1:
            positions = [0]
        else:
            step = (len(rows) - 1) / (count - 1)
            positions = sorted({round(index * step) for index in range(count)})
        double_ids.extend(int(rows.loc[position, "image_id"]) for position in positions)
    merged["second_pass"] = merged["image_id"].isin(double_ids)
    merged["second_annotator"] = ""
    merged["note"] = ""
    for index, row in merged[merged["second_pass"]].iterrows():
        position = names.index(row["annotator"])
        merged.at[index, "second_annotator"] = names[(position + 1) % len(names)]

    form = merged.assign(**{column: "" for column in ATTR_COLUMNS})[COLUMNS]
    form_b = merged[merged["second_pass"]][
        ["image_id", "file_name", "region", "year", "second_annotator", "note"]
    ].rename(columns={"second_annotator": "annotator"})
    form_b = form_b.assign(**{column: "" for column in ATTR_COLUMNS})[COLUMNS]

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    form_path = out_dir / "attributes_form.csv"
    second_path = out_dir / "attributes_form_b.csv"
    form.to_csv(form_path, index=False)
    form_b.to_csv(second_path, index=False)

    print(f"primary pass : {len(form)} rows -> {display(form_path)}")
    for name, count in form["annotator"].value_counts().sort_index().items():
        print(f"  {name}: {count}")
    print(f"second pass  : {len(form_b)} rows -> {display(second_path)}")
    for name, count in form_b["annotator"].value_counts().sort_index().items():
        print(f"  {name}: {count}")
    print("districts in the second pass:", form_b["region"].nunique())
    print("\nallowed values")
    print("  scale_bucket   :", ", ".join(SCALE))
    print("  occlusion_level:", ", ".join(OCCLUSION))
    print("  illumination   :", ", ".join(ILLUMINATION))
    print("  text_density   :", ", ".join(TEXT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
