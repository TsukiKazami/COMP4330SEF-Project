# Project Plan

## 1. Objective and scope

Evaluate **unmodified, pretrained YOLOv8** (n / s / m) on a locally built **Hong Kong street scene** test set, quantify how inference parameters change its behaviour, diagnose the dominant failure modes with TIDE, and then test **one** architecture change that the diagnosis justifies under a strict controlled ablation.

Explicit scope boundaries:

- **Zero-shot first.** Phases 2–4 use COCO-pretrained weights with **no fine-tuning**. The test set is never used for training or tuning of weights, at any phase.
- **Single hardware configuration.** All accuracy and latency numbers come from one machine, recorded per run.
- **FP32, fixed batch size, fixed `imgsz=640`** across all runs.
- **One intervention only.** Phase 5 modifies the architecture once, in one place; multi-module stacks are out of scope.
- **A negative ablation result is a valid result**, analysed through loss curves, gradient flow and feature redundancy (§6.7).

## 2. Research questions → phase map

| RQ | Question | Phase | Primary artifact |
|---|---|---|---|
| RQ1 | Accuracy/speed difference across YOLOv8n/s/m, and does the COCO trade-off hold? | Phase 2 | `results/tables/rq1_scaling.csv` |
| RQ2 | How do confidence and NMS IoU thresholds affect performance, and can one config serve all sizes? | Phase 3 | `results/tables/rq2_sweep.csv`, heatmaps |
| RQ3 | Which error types dominate, and are they correlated with occlusion / scale / text density? | Phase 4 | `results/tables/rq3_tide.json`, chi-square report |
| RQ4 | What single architectural change does the diagnosis justify, and what is its mechanism? | Phase 5 | `configs/models/*.yaml`, design note |
| RQ5 | Does it improve under controlled ablation, and if not, why? | Phase 6 | `results/tables/rq5_ablation.csv`, analysis note |

Phases 2→3→4 are strictly sequential in *interpretation* (Phase 3 needs the per-model baseline, Phase 4 needs the chosen "optimal" configuration), but the tooling can be built ahead of the results.

## 3. Environment and reproducibility baseline

The environment is already built. See [README](/README.md) for more information.

Conventions that apply to every experiment from Phase 2 onward:

| Item | Decision |
|---|---|
| Entry point | `uv run python -m comp4330sef.<module>` — no notebook-only results |
| Config | Every run is driven by a YAML file in `configs/`; no hard-coded paths or thresholds |
| Seed | `seed=0` fixed; record whether results are seed-stable on a sample |
| Precision | FP32 (`half=False`), `imgsz=640`, batch logged per run |
| Device | One fixed device, logged as a string in each runcard |
| Run record | Each run writes `results/runs/<run_id>/runcard.json` = config + git SHA + package versions + device + timestamps |
| Metric source | `pycocotools` for AP, `tidecv` for error decomposition — never hand-rolled AP |

## 4. Planned repository layout

```
configs/
  models.yaml            # n/s/m weights, imgsz, conf, iou, device, fp32
  sweep.yaml             # 9 conf x 5 iou grid
  ablation.yaml          # baseline vs modified, training hyperparameters
data/
  raw/                   # Mapillary / traffic-camera downloads   (ignored)
  images/                # normalised 640x640 JPEGs                (ignored)
  annotations/
    hk_test_coco.json    # COCO-format test set annotation
    attributes.csv       # per-image scale / occlusion / illumination / text density
    train_val_coco.json  # fine-tuning pool (Phase 5 only)
src/comp4330sef/
  data/                  # fetch, filter, resize, split, COCO conversion
  eval/                  # COCO metrics + latency/throughput harness
  sweep/                 # grid runner over conf x iou
  analysis/              # TIDE driver, chi-square, paired t-test, plots
  models/                # model YAMLs + CBAM / P2 definitions
  report/                # table and figure generation
results/
  runs/<run_id>/         # per-run artifacts (ignored)
  tables/                # committed CSV/JSON summaries
  figures/               # committed PNG/PDF figures
```

Figures committed to `results/figures/` are the ones that go into the report; raw Ultralytics `runs/` output stays local.

## 5. Phase plan

### Phase 0 — Setup verification and experiment conventions

**Goal:** Convert a working environment into a reproducible experiment harness.

Tasks:
- [ ] Download and checksum the three target checkpoints (`yolov8n/s/m.pt`); confirm each reports its expected parameter count.
- [ ] Create `configs/models.yaml` and the runcard schema; run one end-to-end dummy pass on 5 COCO images to prove the pipeline shape works.

### Phase 1 — Test set construction

**Goal:** Build the ≥200-image Hong Kong street scene test set, annotated and attribute-tagged, frozen before any model is evaluated on it.

Tasks:
- [ ] Acquire images from Mapillary's open API, filtered to Hong Kong coordinates; supplement with publicly accessible government traffic camera frames.
- [ ] De-duplicate near-identical frames (camera feeds repeat); log the source URL and capture date per image for attribution.
- [ ] Filter into a stratified pool targeting **≥200 images**, balanced across:
  - scale variance — small / medium / large objects present;
  - environmental occlusion — clear vs. heavy (pedestrian behind footbridge, behind a double-decker bus);
  - illumination and clutter — daytime, low-light, dense Traditional Chinese signage.
- [ ] Normalise to 640×640 through the standard Ultralytics preprocessing path; keep the original aspect-preserving resize and record it.
- [ ] Annotate every COCO-class instance with box labels; convert to `hk_test_coco.json`.
- [ ] Tag each image in `attributes.csv` with `scale_bucket`, `occlusion_level`, `illumination`, `text_density`.
- [ ] Double-annotate a ~40-image subset with two independent members; compute IoU-matched box agreement and **Cohen's kappa** for the attribute labels; adjudicate disagreements.
- [ ] Freeze the set: tag the commit, hash the annotation file, and state in the report that it was untouched during model development.
- [ ] Separately build the Phase 5 fine-tuning pool (train/val, disjoint from the test set, same source pipeline).

### Phase 2 — Baseline reproduction and model scaling (RQ1)

**Goal:** Establish the zero-shot baseline for n/s/m and measure whether the COCO accuracy/speed trend survives on local hardware.

Tasks:
- [ ] Build `src/comp4330sef/eval/` : COCO evaluation via `pycocotools`, plus a latency/throughput harness.
- [ ] Run all three models at the **default** `conf=0.25`, `iou=0.7` as the reference point.
- [ ] Record per model: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, per-class AP, parameter count, model file size, latency (ms/frame), throughput (FPS).
- [ ] Compute **accuracy gain per parameter increment** (Δ mAP per +1M parameters between n→s and s→m) and identify the inflection point.
- [ ] Timing protocol: 10 warm-up iterations discarded, then 100 timed iterations; report mean ± std and p50/p95; separate preprocess / forward / NMS stages. Timing runs use batch size 1 so the per-frame cost is honest; accuracy runs may use a larger batch (logged).
- [ ] Write `results/tables/rq1_scaling.csv` and the scaling figure; draft the RQ1 answer paragraph.

### Phase 3 — Parameter sensitivity sweep (RQ2)

**Goal:** Map the false-positive / false-negative trade-off and test whether one configuration serves all model sizes.

Tasks:
- [ ] Implement the grid runner from `configs/sweep.yaml`: **conf ∈ {0.1 … 0.9, step 0.1} × iou ∈ {0.3 … 0.7, step 0.1}** = 9 × 5 = 45 combinations **per model**, 135 runs total.
- [ ] Cache detections once per (model, conf, iou) triple where possible rather than re-inferring for every metric.
- [ ] For each cell compute Precision, Recall, F1, mAP@0.5; store as tidy long CSV `results/tables/rq2_sweep.csv`.
- [ ] Plot per size: conf-vs-F1 and conf-vs-recall curves, PR curves, and 2-D heatmaps (x = conf, y = iou) for mAP@0.5 and F1.
- [ ] Identify each size's optimum and its stability margin (how flat the optimum is) — this decides whether "one config for all sizes" is defensible.
- [ ] Fix the **optimal configuration per model** for Phase 4, and record that choice before looking at Phase 4 error counts.
- [ ] Sanity-check: every row's P/R/F1 must be internally consistent with its confusion counts.

### Phase 4 — Error diagnosis (RQ3)

**Goal:** Decompose the aggregate metric into error types and test whether error categories are statistically associated with scene attributes.

Tasks:
- [ ] Extract all misdetections at each model's optimal configuration using TIDE (`tidecv`): classification, localization, background (duplicate/background FP), missed detection.
- [ ] Produce the TIDE error-composition bar chart per model; report the dominant error and its absolute contribution.
- [ ] Stratified manual review of **≥30 erroneous instances**: sample across error categories and across the attribute strata so the sample is not dominated by one scene type.
- [ ] Categorise each instance into the proposal's four types (classification / localization / background confusion / missed detection) with a one-line scene description and a crop screenshot for the report.
- [ ] Run **chi-square tests** of error category × attribute (occlusion level, scale bucket, text density), report χ², df, p, and an effect size (Cramér's V); note where expected counts are too low and which exact test was used instead.
- [ ] Write the diagnostic conclusion: **which single intervention is justified** — CBAM into the backbone if background confusion dominates, or a P2 detection layer if small-object misses dominate — and state the expected mechanism, not just the choice.

### Phase 5 — Targeted architectural modification (RQ4)

**Goal:** Implement exactly one intervention, and set up the training pipeline that makes the ablation fair.

Tasks:
- [ ] Implement the chosen module in `src/comp4330sef/models/`: CBAM inserted at a fixed, documented backbone location **or** a P2 detection layer added to the head.
- [ ] Write the modified model YAML; verify it loads, that the parameter delta is what the design predicts, and that forward/backward pass shapes are correct.
- [ ] Confirm the modified model still runs inference with the same preprocessing and post-processing as the baseline (no accidental extra tricks).
- [ ] Train both baseline and modified model on the **identical** HK fine-tuning pool with **identical** hyperparameters (epochs, batch, LR, augmentation, seed, and the same images in the same order).
- [ ] Log training loss curves, per-epoch validation mAP, gradient norms at the insertion point, and peak memory.
- [ ] Run the training twice for at least one pair (seeded) to show the run-to-run noise floor — needed to interpret small differences honestly.
- [ ] Add a short design note explaining the mechanism: what feature the module is supposed to recover or suppress, and why the Phase 4 evidence points to it.

### Phase 6 — Controlled ablation and evaluation (RQ5)

**Goal:** Measure the intervention's effect. If it fails, then explain the mechanism.

Tasks:
- [ ] Evaluate baseline-finetuned and modified-finetuned on the **held-out frozen test set** at each model's optimal configuration.
- [ ] Report the full metric set (mAP@0.5, mAP@0.5:0.95, P, R, F1) plus latency and FPS — the modification must also be judged on its cost.
- [ ] Apply a **paired t-test** across per-image scores; report the mean difference, 95% CI, t, df, p, and effect size. State the assumption checks (normality) and use a non-parametric alternative if violated.
- [ ] Report per-class deltas to show *where* the change helped or hurt, not just the aggregate.
- [ ] If performance improves: identify which error category shrank (re-run TIDE) to confirm the stated mechanism actually acted.
- [ ] If performance fails to improve: run the failure analysis — loss curve divergence, gradient flow at the insertion point, feature-map similarity / redundancy with the existing C2f block, added parameters vs. gained accuracy — and write it up as a substantive finding.
- [ ] Re-run TIDE on the modified model to see whether the intended error type moved at all.

### Phase 7 — Integration and reporting

**Goal:** Turn results into the report.

## 6. Metric and measurement protocol

- **AP / mAP** — `pycocotools`, COCO IoU convention. mAP@0.5 and mAP@0.5:0.95 both reported.
- **Precision / Recall / F1** — at the stated conf and IoU, micro-averaged across classes.
- **Accuracy gain per parameter increment** — ΔmAP@0.5:0.95 divided by Δparameters (millions) for n→s and s→m; the inflection is where this ratio drops sharply.
- **Latency** — end-to-end per frame (preprocess + forward + NMS) and stage-wise; mean and p95 after warm-up.
- **Throughput** — FPS at batch size 1, derived from the same timed loop.
- **Inter-annotator agreement** — IoU-matched box agreement plus Cohen's kappa on attribute labels.
- **Error association** — chi-square (+ Cramér's V) for error category × attribute.
- **Ablation significance** — paired t-test on per-image scores (+ non-parametric fallback), with CI and effect size.

---
