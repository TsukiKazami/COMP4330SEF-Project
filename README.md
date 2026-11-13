# COMP4330SEF-Project

A systematic evaluation of YOLOv8 on Hong Kong street scenes: model scaling,
confidence / NMS sensitivity, failure-mode diagnosis, and one targeted
architectural intervention with a controlled ablation.

**Headline results** (full detail in [`docs/results.md`](docs/results.md)):

| RQ | finding |
|---|---|
| RQ1 | larger is more accurate but the return per parameter drops 72% from n→s to s→m; `m` costs 2.9× the latency of `n` for +0.065 AP |
| RQ2 | one configuration (conf 0.25, NMS IoU 0.3) serves all three sizes (max regret 0.0025 F1); the default NMS IoU 0.7 is F1-suboptimal |
| RQ3 | misses dominate the error count (94% of them small objects, χ²=548.7, p<1e-100); class confusion dominates dAP but is confined to car/truck/bus |
| RQ4 | background errors are negligible, so the single intervention is a P2 (stride-4) head: −4.5% parameters, +29% FLOPs |
| RQ5 | fine-tuning alone doubles mAP (0.2235 → 0.4638); P2 with a matched 60-epoch budget adds +0.013 mAP but zero operating-point F1 at +24% latency |

## Project members

| Name | Student ID |
|------|-----------|
| Lau Chun Kit | 13897100 |
| Choy Tsz Lok | 14108370 |
| Chong Tsz Ho | 13990354 |
| Tse Tsz Sum | 14231009 |
| Cheung Tsz Hin Anderson | 14113287 |
| Fung Ming Chu | 14201890 |

## Repository map

```
configs/models.yaml              model defaults and eval protocol (single source of truth)
configs/models/yolov8s-p2.yaml   the RQ4 intervention (P2 detection head, nc=4)
configs/hk4.yaml                 Ultralytics dataset yaml (path is machine-specific)
data/annotations/hk_test_coco.json   frozen test set, 120 images / 1,210 boxes
data/annotations/hk_dataset_manifest.json   split + sha256 provenance
docs/plan.md                     the sprint plan
docs/results.md                  every number in the report, with its table
results/tables/                  committed CSVs, one set per RQ
results/figures/                 committed figures, one per RQ
scripts/build_hk_dataset.py      audit + convert + re-split (the data pipeline)
scripts/fetch_weights.py         download the three COCO checkpoints + verify them
scripts/plot_rq*.py              regenerate each figure from its CSV
src/comp4330sef/eval/            metrics, scoring, runcards, RQ1 and RQ5
src/comp4330sef/sweep/           RQ2 confidence / NMS-IoU sweep
src/comp4330sef/analysis/        RQ3 TIDE wrapper and error taxonomy
src/comp4330sef/models/          RQ4 baseline vs P2 budget check
```

## Setup

Requires [uv](https://docs.astral.sh/uv/)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/TsukiKazami/COMP4330SEF-Project
cd COMP4330SEF-Project
uv sync
uv run python scripts/fetch_weights.py     # yolov8n/s/m.pt + sha256 manifest
```

## Data

The images are **not** committed. Download the public MIT-licensed dataset and
rebuild the evaluation set:

1. Open <https://universe.roboflow.com/roy-b2g9i/hong-kong-vehicle-detection-8zkg0>,
   version **v3** → Download → **COCO JSON**.
2. Unzip so that `datasets/hk_vehicle/{train,valid,test}/_annotations.coco.json`
   exist.
3. Run the builder:

```powershell
uv run python scripts/build_hk_dataset.py --dry-run   # prints the audit first
uv run python scripts/build_hk_dataset.py
```

The builder reports that only 599 of the 2,354 images are usable: the v3 export
ships 18 category entries of which 12 are README / dataset-version strings, and
their 8,711 boxes are dropped. It keeps four COCO classes (car 3, motorcycle 4,
bus 6, truck 8), re-splits by scene group into 419 / 60 / 120, and writes the
frozen test set plus `hk_dataset_manifest.json`.

## Reproducing the results

```powershell
uv run comp4330 rq1 --gt data/annotations/hk_test_coco.json --images data/hk/images/test --tag hk_test
uv run comp4330 rq2
uv run comp4330 rq3
uv run comp4330 models
uv run comp4330 rq5 --train
uv run comp4330 rq5
uv run python scripts/plot_rq1.py
uv run python scripts/plot_rq2.py
uv run python scripts/plot_rq3.py
uv run python scripts/plot_rq5.py
```

Every run writes a runcard under `results/runs/` (ignored by git) recording the
git sha, device, package versions and protocol.

## Notes

- All experiments were run on a **Google Colab T4 (fp32)**.
- `configs/hk4.yaml` contains an absolute path because
  `ultralytics.data.utils.check_det_dataset` resolves relative paths against the
  global datasets directory. `rq5` regenerates a local copy at run time, so the
  committed file never has to be edited by hand.
- `rq2` caches its detections; `rq3` and `rq5` consume those caches, so running
  `rq2` first is the cheapest order.
- AR/AP numbers use the four-class protocol: COCO's 80-class taxonomy is never
  evaluated, and `person` / `bicycle` / `train` are absent from the data.