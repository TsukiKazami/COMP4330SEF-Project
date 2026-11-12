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
| Hardware | Google Colab T4 16 GB, logged in each runcard |
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

The v3 COCO export ships 18 category entries, but only six carry a real vehicle
name; the other twelve are README / dataset-version strings, and their 8,711
boxes (on 1,755 images) are dropped. The six that survive map to **4** COCO
classes:

| Dataset class | COCO class | COCO id | boxes |
|---|---|---|---|
| `car`, `taxi` | car | 3 | 3,755 |
| `motorcycle` | motorcycle | 4 | 408 |
| `bus`, `mini bus` | bus | 6 | 921 |
| `truck` | truck | 8 | 871 |
| the 12 corrupted entries | dropped | — | 8,711 |

COCO ids are mandatory (`car=3, motorcycle=4, bus=6, truck=8`) because
Ultralytics returns `class_index + 1 = COCO id` for the 80-class checkpoints;
any other numbering silently gives mAP = 0. The fine-tuned RQ5 model maps its
0–3 indices with an explicit `class_ids=[3, 4, 6, 8]`.

Consequence to state in the report: no `train`, `tram`, `heavy truck`,
`light truck`, `person` or `bicycle`; evaluation covers 4 classes on 599
images (25% of the 2,354 downloaded), not 5 classes on the full set.

### 2.2 Split policy

Roboflow's split is random and leaks: 30 base names appear in two splits.
Re-split **by scene group** — the video uuid for frame sequences, the camera id
(`SK8119`, `RI000712`) for stills. 463 groups over 599 images (12 multi-image,
largest 24), assigned greedily to the 70 / 10 / 20 targets: **419 / 60 / 120**
images, 4,123 / 622 / 1,210 boxes. The test split is frozen before RQ1 runs and
recorded by sha256 in `data/annotations/hk_dataset_manifest.json`.

### 2.3 Data-quality audit

`scripts/build_hk_dataset.py` is the audit and the converter in one. It reports
1,755 images / 8,711 boxes dropped because their category name is a README or
dataset-version string, and prints the per-split, per-class counts. The report
must cite this: the public dataset is 25% usable for class-aware evaluation,
and the loss is a property of the export, not of our pipeline.

---

## 3. The five questions

Every RQ must appear in the slides as **one table + one figure**.

| RQ | Question (short) | Deliverable in this sprint | Compressed scope |
|---|---|---|---|
| RQ1 | accuracy/speed of n/s/m, does the COCO trade-off hold | `results/tables/rq1_scaling.csv`, scaling figure | 5-class AP, P/R/F1, latency, marginal mAP per +1M params |
| RQ2 | conf / IoU sensitivity, one config for all sizes? | `rq2_sweep.csv`, 1 heatmap + 1 conf-F1 curve | **9 conf × 3 IoU** (0.3/0.5/0.7); PR curves dropped |
| RQ3 | which error types dominate, tied to scene attributes? | `rq3_tide.json`, error composition chart, chi-square | TIDE on all models; **12–15** manual error instances (not 30); attributes `occlusion_level`, `text_density` human, `scale_bucket` computed from boxes |
| RQ4 | which single change does the diagnosis justify? | `configs/models/*.yaml`, module in `src/comp4330sef/models/`, param-delta + shape check | one intervention only |
| RQ5 | does it win under controlled ablation, and why? | `rq5_ablation.csv`, paired t-test, per-class deltas, **error-composition comparison (same taxonomy as RQ3, TIDE not re-run)** | `yolov8s`, **30 epochs**, identical hyper-parameters and data order; noise-floor rerun **skipped** (state as limitation) |

RQ3–RQ5 run at the joint optimum found in RQ2: `conf=0.25`, NMS `iou=0.3`
(the matching IoU stays at the protocol value 0.5). RQ1 intentionally keeps the
default `0.25 / 0.7` so that RQ2 can show the default is F1-suboptimal.

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
