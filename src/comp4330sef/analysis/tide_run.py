"""RQ3: TIDE error decomposition and the instance-level taxonomy.

Runs on the detections cached by RQ2 at the joint operating point (NMS IoU 0.3,
conf floor 0.001), so no inference happens here and the whole analysis is CPU
only. TIDE's COCO loader keeps the 100 highest-scoring detections per image, so
the decomposition is relative, not an mAP estimate -- use pycocotools for AP.

TIDE writes its cache under ``TIDE_PATH``; on Windows the default under
``%LOCALAPPDATA%`` can raise PermissionError, hence the override below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from comp4330sef.analysis.taxonomy import chi_square, classify
from comp4330sef.config import REPO_ROOT, rel_path
from comp4330sef.eval.runcard import new_run_id, write_runcard
from comp4330sef.eval.score import restrict_to_categories

os.environ.setdefault("TIDE_PATH", str(REPO_ROOT / ".cache" / "tidecv"))

HK_NAMES = {"hk", "hk_test"}


def find_detections(model: str, nms_iou: float) -> Path:
    """Newest cached RQ2 detection file for ``model`` at ``nms_iou``."""
    pattern = f"rq2_{model}_*/det_{model}_nms{nms_iou:.2f}.json"
    matches = sorted((REPO_ROOT / "results" / "runs").glob(pattern))
    if not matches:
        raise SystemExit(
            f"no cached detections for {model} at NMS IoU {nms_iou:.2f}; run rq2 first "
            f"(looked for results/runs/{pattern})"
        )
    return matches[-1]


def load_detections(path: Path, categories: set[int]) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return restrict_to_categories(records, categories)


def _tide():
    try:
        from tidecv import TIDE, datasets
    except Exception as exc:  # pragma: no cover - dependency guard
        raise SystemExit(f"tidecv is not usable: {exc}") from exc
    return TIDE, datasets


def run_tide(
    gt_path: Path,
    records: Sequence[dict[str, Any]],
    name: str,
    work_dir: Path,
) -> dict[str, Any]:
    """TIDE dAP decomposition and error counts for one model."""
    TIDE, datasets = _tide()

    work_dir.mkdir(parents=True, exist_ok=True)
    det_path = work_dir / f"{name}_detections.json"
    det_path.write_text(json.dumps(list(records)), encoding="utf-8")

    tide = TIDE(mode=TIDE.BOX)
    run = tide.evaluate(
        datasets.COCO(str(gt_path)),
        datasets.COCOResult(str(det_path)),
        name=name,
    )
    main_errors = tide.get_main_errors().get(name, {})
    special_errors = tide.get_special_errors().get(name, {})
    counts = {error.short_name: len(run.error_dict[error]) for error in TIDE._error_types}
    return {
        "ap": round(float(run.ap), 4),
        "main_errors": {key: round(float(value), 4) for key, value in main_errors.items()},
        "special_errors": {key: round(float(value), 4) for key, value in special_errors.items()},
        "counts": counts,
        "max_dets": 100,
    }


def run_rq3(
    *,
    gt: str = "data/annotations/hk_test_coco.json",
    tag: str = "hk_test",
    models: Sequence[str] = ("yolov8n", "yolov8s", "yolov8m"),
    nms_iou: float = 0.3,
    conf: float = 0.25,
    match_iou: float = 0.5,
    tide_json: str | None = None,
    tide_table: str | None = None,
    taxonomy_table: str | None = None,
    chi_table: str | None = None,
    verbose: bool = True,
) -> int:
    gt_path = Path(gt)
    if not gt_path.is_absolute():
        gt_path = REPO_ROOT / gt_path
    document = json.loads(gt_path.read_text(encoding="utf-8"))
    categories = {int(category["id"]) for category in document.get("categories", [])}

    suffix = "" if tag.lower() in HK_NAMES else f"_{tag}"
    tables = REPO_ROOT / "results" / "tables"
    tide_json_path = Path(tide_json) if tide_json else tables / f"rq3_tide{suffix}.json"
    tide_table_path = Path(tide_table) if tide_table else tables / f"rq3_tide{suffix}.csv"
    taxonomy_path = Path(taxonomy_table) if taxonomy_table else tables / f"rq3_error_taxonomy{suffix}.csv"
    chi_path = Path(chi_table) if chi_table else tables / f"rq3_chi_square{suffix}.csv"

    run_id = new_run_id("rq3")
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset    : {tag} — {len(document.get('images', []))} images, "
          f"{len(document.get('annotations', []))} instances, {len(categories)} classes")
    print(f"operating  : NMS IoU {nms_iou}, conf {conf}, matching IoU {match_iou}")
    print(f"note       : TIDE integrates at the conf floor (max_dets 100 per image); "
          f"the taxonomy counts errors at conf {conf}")

    tide_results: dict[str, Any] = {}
    taxonomy_rows: list[dict[str, Any]] = []
    for model in models:
        det_path = find_detections(model, nms_iou)
        records = load_detections(det_path, categories)
        print(f"\n=== {model} ===")
        print(f"  detections : {rel_path(det_path)} ({len(records)} in protocol)")

        tide_results[model] = run_tide(gt_path, records, model, run_dir)
        rows = classify(document, records, model=model, conf=conf, match_iou=match_iou)
        taxonomy_rows.extend(rows)
        if verbose:
            main = tide_results[model]["main_errors"]
            ordered = "  ".join(f"{key}={value:.2f}" for key, value in main.items())
            print(f"  TIDE dAP   : {ordered} (AP {tide_results[model]['ap']:.2f})")
            counted = pd.Series([row["error_type"] for row in rows]).value_counts()
            print("  taxonomy   : " + "  ".join(f"{key}={value}" for key, value in counted.items()))

    frame = pd.DataFrame(taxonomy_rows)
    tide_json_path.parent.mkdir(parents=True, exist_ok=True)
    tide_json_path.write_text(json.dumps(tide_results, indent=2), encoding="utf-8")

    tidy = [
        {
            "model": model,
            "error_type": error_type,
            "dap": values,
            "count": tide_results[model]["counts"].get(error_type),
        }
        for model, result in tide_results.items()
        for error_type, values in result["main_errors"].items()
    ]
    pd.DataFrame(tidy).to_csv(tide_table_path, index=False)

    frame.to_csv(taxonomy_path, index=False)
    chi = pd.concat(
        [chi_square(frame, "scale_bucket"), chi_square(frame, "category_id")],
        ignore_index=True,
    )
    chi.to_csv(chi_path, index=False)

    write_runcard(
        run_id,
        config={"nms_iou": nms_iou, "conf": conf, "match_iou": match_iou, "models": list(models)},
        config_path="configs/models.yaml",
        device="cpu",
        artifacts={
            "gt": rel_path(gt_path),
            "tide": rel_path(tide_json_path),
            "taxonomy": rel_path(taxonomy_path),
            "chi_square": rel_path(chi_path),
        },
        extra={"dataset": tag, "tide": tide_results},
    )

    print("\nerror counts by model:")
    counts = (
        frame.pivot_table(index="error_type", columns="model", values="image_id", aggfunc="count")
        .reindex([name for name in ("Miss", "Loc", "Cls", "Dupe", "Bkg") if name in set(frame["error_type"])])
        .fillna(0)
        .astype(int)
    )
    print(counts.to_string())

    print("\nchi-square (error type x scale bucket):")
    print(
        chi[chi["factor"] == "scale_bucket"][
            ["group", "n", "chi2", "dof", "p_value", "cramers_v", "min_expected"]
        ].to_string(index=False)
    )

    print(f"\ntide       : {rel_path(tide_json_path)}")
    print(f"             {rel_path(tide_table_path)}")
    print(f"taxonomy   : {rel_path(taxonomy_path)}")
    print(f"chi-square : {rel_path(chi_path)}")
    print(f"runcard    : {rel_path(run_dir / 'runcard.json')}")
    return 0
