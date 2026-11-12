# Project Plan — sprint to the Sunday presentation

This plan replaces the original multi-week plan in git history.

Goal: answer the five research questions of the proposal on **Hong Kong street
scenes**, keeping RQ5 (controlled ablation). Scope is compressed, not deleted.

---

## 1. Scope boundaries

| Item | Decision |
|---|---|
| Detection models | unmodified COCO-pretrained `yolov8n / s / m` for RQ1–RQ3 (zero-shot) |
| Fine-tuning | RQ5 only, `yolov8s` (fallback `yolov8n` if the GPU is short on time) |
| Precision / size | FP32, `imgsz=640`, batch logged per run |
| Hardware | one machine: RTX 2070 SUPER 8 GB, logged in each runcard |
| Interventions | exactly one (CBAM **or** P2), chosen by the RQ3 diagnosis |
| Negative result | valid; RQ5 reports the mechanism instead of a win |
| Test set use | never trained on; frozen before RQ1 runs |

---

## 2. Dataset

**Primary source**: Roboflow Universe, *hong kong vehicle detection* v3
(<https://universe.roboflow.com/roy-b2g9i/hong-kong-vehicle-detection-8zkg0>),
2,354 images, MIT licence, Hong Kong dashcam (`Random-Dudes-Carcam_*`) and traffic
camera (`SK*_JPG`) frames. Export: **COCO JSON**.

**Why not our own annotation**: the test set and the RQ5 training pool both needed
manual labelling; at 2.5 days that is not feasible. This public dataset is Hong
Kong, MIT-licensed, and already labelled, so RQ5 can train on ~1.9k images.

**Parked fallback**: the 189-image Mapillary test set (`data/annotations/test_set.csv`),
its annotation kit (`docs/annotation_guideline.md`, `configs/annot_labels.txt`,
`scripts/preannotate.py`, `scripts/make_attr_form.py`,
`src/comp4330sef/data/labels_to_coco.py`) stays in the repo but is not on the
critical path. Only revive it if the Roboflow labels turn out unusable.

### 2.1 Class mapping

The dataset has 10 local classes; map them to 5 COCO classes. **The COCO ids are
mandatory** (`car=3, motorcycle=4, bus=6, train=7, truck=8`) because Ultralytics
returns `class_index + 1 = COCO id`; any other numbering silently gives mAP = 0.

| Dataset class | COCO class | COCO id |
|---|---|---|
| `car`, `taxi` | car | 3 |
| `motorcycle` | motorcycle | 4 |
| `bus`, `mini bus` | bus | 6 |
| `train`, `tram` | train | 7 |
| `truck`, `heavy truck`, `light truck` | truck | 8 |
| `vehicle` (ambiguous) | **dropped** | — |

Consequence to state in the report: no `person`, no `bicycle`, no traffic lights;
evaluation covers 5 classes, not 80.

### 2.2 Split policy

Roboflow's own split is random, so consecutive dashcam frames can land in both
train and test. Re-split **by source group** (filename prefix before the first
`_`, e.g. `Random-Dudes-Carcam`, `SK8150`): all frames of a group go to exactly
one of train / val / test. Target roughly 70 / 10 / 20, with the test split held
out and frozen (hash + commit) before RQ1 runs.

---

## 3. The five questions

Every RQ must appear in the slides as **one table + one figure**.

| RQ | Question (short) | Deliverable in this sprint | Compressed scope |
|---|---|---|---|
| RQ1 | accuracy/speed of n/s/m, does the COCO trade-off hold | `results/tables/rq1_scaling.csv`, scaling figure | 5-class AP, P/R/F1, latency, marginal mAP per +1M params |
| RQ2 | conf / IoU sensitivity, one config for all sizes? | `rq2_sweep.csv`, 1 heatmap + 1 conf-F1 curve | **9 conf × 3 IoU** (0.3/0.5/0.7); PR curves dropped |
| RQ3 | which error types dominate, tied to scene attributes? | `rq3_tide.json`, error composition chart, chi-square | TIDE on all models; **12–15** manual error instances (not 30); attributes `occlusion_level`, `text_density` human, `scale_bucket` computed from boxes |
| RQ4 | which single change does the diagnosis justify? | `configs/models/*.yaml`, module in `src/comp4330sef/models/`, param-delta + shape check | one intervention only |
| RQ5 | does it win under controlled ablation, and why? | `rq5_ablation.csv`, paired t-test, per-class deltas, TIDE re-run | `yolov8s`, **30 epochs**, identical hyper-parameters and data order; noise-floor rerun **skipped** (state as limitation) |

RQ4 depends on RQ3; RQ5 depends on RQ4. RQ1–RQ3 are independent of each other in
execution but RQ2's optimal configuration feeds RQ3.

---

## 4. Explicitly out of scope

- 80-class COCO evaluation, `person` class, pedestrian-occlusion narrative.
- RQ2 PR curves, optimum-stability analysis, full 9×5 grid.
- RQ3's 30-instance manual review and the illumination attribute.
- RQ5's second-seed noise floor, and the deep failure analysis unless the result
  is negative.
- Re-fetching Mapillary data; any further work on the parked 189-image set.

---

## 5. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| Roboflow images are 640-stretched (`v1`/`v2` were; `v3` unclear) | check real sizes after download; if distorted, keep only `v3` and note it in the report |
| Labels are auto-generated and noisy | spot-check 40 images, report box agreement; drop classes with unusable labels |
| Class imbalance (bus/train rare) | report instance counts per class; discuss low-count classes instead of quoting unstable AP |
| Split leakage between train and test | re-split by filename prefix, never by random |
| RQ5 training does not finish in time | switch to `yolov8n` and 20 epochs; report the reduced budget |
| Roboflow download unavailable | fall back to the parked 189-image test set + the annotation kit (needs ~2 h of team labelling) |
