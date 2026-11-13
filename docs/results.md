# Results — Hong Kong street-scene evaluation of YOLOv8

Consolidated findings for RQ1–RQ5. Every number below comes from a committed
table under `results/tables/` and a runcard under `results/runs/` (ignored by
git). Frozen artefacts: the task set in `data/annotations/` and
`hk_dataset_manifest.json`.

Hardware: one Google Colab T4 (16 GB), fp32, `imgsz=640`. The local RTX 2070
SUPER was abandoned after a driver-level BSOD during the first RQ1 attempt.

---

## 0. Snapshot

| RQ | Question | Answer |
|---|---|---|
| RQ1 | accuracy/speed across n/s/m, does the COCO trade-off hold? | Yes, but the marginal return per parameter falls 72% from n→s to s→m, and m costs 2.9× the latency for +0.065 AP. |
| RQ2 | do thresholds need to differ by size? | No: one configuration (conf 0.25, NMS IoU 0.3) is within 0.0025 F1 of every model's own optimum. The default NMS IoU 0.7 is F1-suboptimal. |
| RQ3 | which errors dominate, tied to scene attributes? | Misses dominate the count (50–65% of instances, 94% of them small objects, χ²=548.7, p<1e-100); class confusion dominates the dAP (17.4–21.4) but sits entirely inside car/truck/bus. |
| RQ4 | which single change does the diagnosis justify? | A P2 (stride-4) detection head: background errors are negligible, so the attention branch is not triggered. P2 costs +29% FLOPs and −4.5% parameters. |
| RQ5 | does it win under controlled ablation? | At the pre-registered 30 epochs, no: mAP 0.4389 vs 0.4638, per-image F1 −0.0464 [−0.0663, −0.0265], p=9.8e-6. With a matched 60-epoch budget the deficit disappears — mAP 0.4966 vs 0.4836 (+0.0130), AP_small +0.010, AR@100 +0.018 — but the operating-point F1 difference is exactly zero (p=0.999) at +24% latency. The 30-epoch loss was an optimisation deficit, not an architectural one. |

---

### 0.1 Pre-registered protocol

The plan was fixed before the runs. Its binding decisions, so that
the reader can tell protocol from post-hoc choice:

- Five RQs, each delivered as one table and one figure.
- The test split was frozen (hash committed) before any RQ1 run.
- RQ2 counts a shared configuration as adequate when no model loses more than
  **0.01 F1** against its own optimum; that tolerance was set before M2.
- RQ4 permits **exactly one** intervention, chosen by a rule written before RQ3:
  background confusion → attention module, small-object misses → P2 head.
- RQ5 trains both variants for **30 epochs** with identical data order and seed,
  and reports a paired test; a second seed is out of scope and stated as such.

---

## 1. Data

| Item | Value |
|---|---|
| Source | Roboflow *hong kong vehicle detection* v3, COCO JSON, MIT |
| Downloaded | 2,354 images, 18 category entries |
| Usable | **599 images, 5,955 boxes** — the other 1,755 images carry 8,711 boxes whose category names are README / dataset-version strings ("c10 - v1 2024-11-09 2-46am", "Roboflow is an end-to-end…") |
| Classes | 4 COCO ids: car 3, motorcycle 4, bus 6, truck 8 (taxi→car, mini bus→bus; no train/tram) |
| Split | by scene group, 419 / 60 / 120 (train / valid / test) |
| Test set | frozen before RQ1; sha256 `81f79926…30516b3b` in `hk_dataset_manifest.json` |
| Instances (test) | 1,210 — car 760, bus 226, truck 193, motorcycle 31 |
| Images | all 640×640 letterbox (fit, black edges); bbox area median ≈ 0.03% of the frame (≈11 px) |

---

## 2. RQ1 — model scaling

Operating point: conf 0.25, NMS IoU 0.7 (the documented default), `max_det=300`,
AP at conf floor 0.001.

| model | params | mAP@50:95 | mAP@50 | P | R | F1 | latency | FPS | AP_small | AP_med | AP_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 3.16 M | 0.1802 | 0.2828 | 0.6300 | 0.3884 | 0.4806 | 7.81 ms | 128 | 0.0624 | 0.4154 | 0.6544 |
| s | 11.17 M | 0.2235 | 0.3517 | 0.6405 | 0.4521 | 0.5300 | 12.38 ms | 81 | 0.0959 | 0.4565 | 0.6559 |
| m | 25.90 M | 0.2451 | 0.3845 | 0.6217 | 0.4917 | 0.5491 | 22.62 ms | 44 | 0.1053 | 0.4969 | 0.7921 |

Marginal return:

| step | Δparams | ΔmAP@50:95 | AP per +1 M | Δlatency |
|---|---|---|---|---|
| n→s | +8.01 M | +0.0433 | 0.0054 | +4.6 ms |
| s→m | +14.74 M | +0.0216 | 0.0015 | +10.2 ms |

Per class (mAP@50:95):

| class | GT | n | s | m |
|---|---|---|---|---|
| car | 760 | 0.2598 | 0.3348 | 0.3634 |
| bus | 226 | 0.2101 | 0.2612 | 0.2579 |
| truck | 193 | 0.2258 | 0.2586 | 0.3101 |
| motorcycle | 31 | 0.0252 | 0.0394 | 0.0488 |

**Reading.** The COCO trade-off holds in direction but not in value: accuracy
rises monotonically while the return per parameter collapses by 72% between the
two steps. `m` buys +0.065 AP over `n` at 2.9× the latency, so `s` is the last
step that is cheap. The per-class spread (±0.34 on car vs ±0.05 on motorcycle)
is larger than the model-size spread, and `m` is not even the best on bus.
Motorcycle AP is not interpretable at 31 instances and is reported for
completeness only.

**Sensitivity.** `max_det` 100 was raised to 300 before this run; mAP@50:95
changed by <0.002 and P/R/F1 not at all, so the truncation was not a confound.
Latency is the least stable quantity: a repeat of the 100-detection run measured
yolov8s at 10.4 ms against 12.4 ms here (±20%, shared T4), which is why the
argument is built on the parameter marginals rather than on milliseconds.

---

## 3. RQ2 — confidence and NMS-IoU sensitivity

Grid: 9 confidences × 3 NMS IoUs, matching IoU fixed at 0.5 so only the
deployment knob moves. One inference pass per (model, IoU) at the AP floor; all
grid cells are re-matched from the cached detections.

| model | own optimum | F1 | default (0.7, 0.25) | shared (0.3, 0.25) | regret at shared |
|---|---|---|---|---|---|
| n | (0.5, 0.20) | 0.4847 | 0.4806 | 0.4822 | +0.0025 |
| s | (0.3, 0.30) | 0.5377 | 0.5300 | 0.5360 | +0.0017 |
| m | (0.3, 0.30) | 0.5556 | 0.5491 | 0.5534 | +0.0022 |

`max regret = 0.0025` against a pre-registered tolerance of 0.01 → **one
configuration serves all three sizes** (`results/rq2_answer.json`,
`one_config_serves_all = true`).

Three further observations:

1. **The default NMS IoU is F1-suboptimal.** Moving 0.7 → 0.3 raises F1 by
   +0.0016 / +0.0060 / +0.0043, and the gain is entirely precision
   (+0.010 / +0.018 / +0.014) with recall flat: lower NMS IoU removes duplicate
   boxes over the same vehicle.
2. **mAP and F1 disagree about the IoU.** mAP@50:95 is highest at IoU 0.7 for
   every model (n 0.1802 vs 0.1730, s 0.2235 vs 0.2207, m 0.2451 vs 0.2406)
   while F1 is highest at 0.3. A permissive NMS keeps the low-score overlapping
   boxes that COCO AP rewards, and the same boxes cost precision at a single
   operating point. The "best configuration" therefore depends on the metric.
3. **Model ranking is configuration-invariant.** n < s < m in F1 in all 27 grid
   cells, so threshold choice never reorders the checkpoints.
4. Sanity check: at conf 0.70 all three IoUs give identical F1 (0.182 / 0.368 /
   0.380) because no overlapping boxes survive, i.e. the sweep really moves only
   the NMS knob.

**Decision carried forward.** RQ3–RQ5 run at `conf=0.25`, NMS `iou=0.3`; RQ1
keeps the default so that RQ2 can show the default is suboptimal.

---

## 4. RQ3 — failure diagnosis

Input: the detections cached by RQ2 at NMS IoU 0.3. TIDE integrates at the
confidence floor (its COCO loader keeps the top 100 detections per image); the
instance taxonomy counts errors at the operating point conf 0.25.

**TIDE dAP** (percentage points of mAP recoverable if that error were fixed):

| model | Cls | Miss | Loc | Bkg | Both | Dupe | TIDE AP |
|---|---|---|---|---|---|---|---|
| n | 17.42 | 11.00 | 3.86 | 1.88 | 0.30 | 0.00 | 28.50 |
| s | 18.32 | 12.56 | 4.17 | 3.05 | 0.32 | 0.00 | 35.28 |
| m | 21.40 | 13.34 | 4.87 | 2.02 | 0.15 | 0.00 | 38.02 |

`Both` is TIDE's `OtherError`: an error that would need both the class and the box
corrected. It never exceeds 0.32 dAP, so it does not change the ranking below.

**Instance taxonomy at the operating point** (each error counted once):

| model | Miss | Cls | Bkg | Loc | total |
|---|---|---|---|---|---|
| n | 610 | 147 | 99 | 17 | 873 |
| s | 491 | 174 | 91 | 19 | 775 |
| m | 439 | 170 | 148 | 22 | 779 |

**Misses are the count, and they are small.** 94% of missed ground truth is in
the small bucket (1,482 of 1,578 pooled); the association is strong and not an
artefact of the sample: χ² = 548.66, df = 6, p = 2.7e-115, Cramér's V = 0.336,
and per model V = 0.32 / 0.33 / 0.38. At the operating point `n` misses 50% of
its ground truth, `m` misses 36%.

**Class confusion is the dAP term, but it is a label boundary.** Every Cls error
is a detection of one vehicle class sitting on a ground truth of another:

| detection → ground truth | car | bus | truck |
|---|---|---|---|
| car | — | 81 | 116 |
| truck | 90 | 145 | — |
| bus | 14 | — | 44 |

No confusion involves motorcycle, and none involves a non-vehicle background
object. This is the COCO boundary between car, van/light truck and minibus in
Hong Kong traffic, not a feature-extraction failure; no neck module addresses it.

**Localisation and duplicates are negligible** (Loc 3.9–4.9 dAP, 17–22 instances;
Dupe ≈ 0 at NMS IoU 0.3), and **background confusion is small** (1.9–3.0 dAP).

---

## 5. RQ4 — the intervention

The plan's decision rule was pre-registered before RQ3 ran: background confusion
→ attention module; small-object misses → P2. Background errors are 1.9–3.0 dAP
and 91–148 instances, so the attention branch is not triggered. Misses are
second in dAP, 50–65% of instances, 94% small, and AP_small (0.06–0.10) is six
to ten times below AP_large (0.65–0.79). **The single intervention is a P2
detection head.**

| variant | params | Δparams | GFLOPs@640 | ΔGFLOPs | levels | strides | anchors |
|---|---|---|---|---|---|---|---|
| yolov8s (nc=4) | 11,137,148 | — | 28.6 | — | 3 | 8/16/32 | 8,400 |
| yolov8s-p2 (nc=4) | 10,638,016 | −499,132 (−4.48%) | 36.9 | +8.3 (+29.0%) | 4 | 4/8/16/32 | 34,000 |

Cross-checked against the stock nc=80 configs (11,166,560 → 10,884,336): the P2
yaml rebalances the neck widths, so **the intervention buys resolution, not
capacity**. This matters for interpreting RQ5 — any gain cannot be attributed to
a larger model, and any loss cannot be blamed on it either.

---

## 6. RQ5 — controlled ablation

Both variants fine-tuned from COCO weights on the same 419 images with the same
hyper-parameters and seed (`epochs=30`, `batch=8`, `imgsz=640`, `seed=0`,
`deterministic=True`); the only difference is the head. Evaluated on the frozen
120-image test split at conf 0.25 / NMS IoU 0.3.

| variant | params | mAP@50:95 | mAP@50 | P | R | F1 | latency | FPS | detections |
|---|---|---|---|---|---|---|---|---|---|
| baseline (yolov8s) | 11.14 M | **0.4638** | **0.6758** | 0.8500 | 0.7306 | **0.7858** | **10.44 ms** | 95.8 | 3,916 |
| p2 (yolov8s-p2) | 10.64 M | 0.4389 | 0.6642 | 0.7887 | 0.7066 | 0.7454 | 13.93 ms | 71.8 | 6,271 |

**Paired test** over the 120 images on per-image F1 (baseline − modified):
−0.0464, 95% CI [−0.0663, −0.0265], t = −4.62, p = 9.8e-6, Cohen's d = −0.42;
P2 wins 35 images, loses 66, ties 19.

**Where it failed** (from the evaluation runcard):

| | AP_small | AP_medium | AP_large | AR@100 |
|---|---|---|---|---|
| baseline | 0.3143 | 0.8294 | 0.9118 | 0.4973 |
| p2 | 0.3053 | 0.7739 | 0.8180 | 0.4973 |

The mechanism did not fire: the stride-4 level did not raise AP_small (+0.009
in the wrong direction), while AP_medium and AP_large fell by 0.055 and 0.094 —
the objects that were already strong. AR@100 is identical, so P2 finds the same
objects and ranks them worse, emitting 6,271 detections at the operating point
against 3,916 (precision 0.850 → 0.789). The error taxonomy agrees: every error
type grew (Miss 169 → 195, Cls 91 → 125, Bkg 37 → 70, Loc 28 → 34).

**Why: a cold start plus a re-routed pyramid.**

1. *Cold start.* Only 219 of 437 tensors transfer from the COCO checkpoint; the
   P2 branch and the four-class head are random. The P2 model begins at 0.000
   validation mAP against 0.303 for the baseline, and at epoch 30 it is still
   improving by ≈0.01 mAP per epoch (0.4661 validation, rising) while the
   baseline has converged (0.4915, flat since ≈epoch 26). The comparison
   therefore bounds P2 under a 30-epoch budget.
2. *Re-routed features.* The extra P2 stage changes the top-down path, so the
   P3–P5 features that carry the pretrained knowledge are recomputed through a
   different neck. The loss concentrating in medium/large objects is the
   signature of that disruption rather than of a resolution deficit.

**The largest result of the project is not the intervention but the fine-tune.**
The same checkpoint moves from 0.2235 mAP@50:95 zero-shot (RQ1) to 0.4638
(+0.240, 2.08×); per class car 0.3348 → 0.5217, bus 0.2612 → 0.5387, truck
0.2586 → 0.5251, motorcycle 0.0394 → 0.2696 (6.8×).

---

### 6a. Follow-up: a matched 60-epoch budget

The 30-epoch comparison is confounded by the intervention's cold start, so both
variants were retrained with 60 epochs and otherwise identical settings. The
baseline improves on its own (0.4638 → 0.4836), and P2 crosses it:

| variant | mAP@50:95 | mAP@50 | P | R | F1 | latency | detections |
|---|---|---|---|---|---|---|---|
| baseline 60 ep | 0.4836 | 0.7025 | 0.8266 | 0.7645 | 0.7943 | 11.65 ms | 3,087 |
| p2 60 ep | **0.4966** | **0.7259** | 0.8376 | 0.7587 | **0.7962** | 14.48 ms | 4,549 |

| | AP_small | AP_medium | AP_large | AR@100 |
|---|---|---|---|---|
| baseline 60 ep | 0.3490 | 0.8256 | 0.8676 | 0.5189 |
| p2 60 ep | **0.3591** | **0.8405** | 0.8496 | **0.5373** |

The paired test on per-image F1 is now exactly null: −0.0000 [−0.0197, +0.0196],
t = −0.0015, p = 0.999, wins 51 / losses 47 / ties 22. Per class the gain is
concentrated in the smallest and rarest class (motorcycle 0.2748 → 0.3340);
car, bus and truck are within 0.007 of each other.

**Revised reading.** The stride-4 head does what it was designed to do — AP_small
rises by 0.010, AP_medium by 0.015, AR@100 by 0.018 — but the effect is too
small to move the operating point, and it costs +24% latency and +29% FLOPs for
4.5% fewer parameters. The 30-epoch deficit was an optimisation artefact of the
cold start (219 of 437 tensors transferred, random P2 branch and head), not an
architectural penalty: given the same budget the re-routed pyramid recovers
completely, including the medium and large buckets that had appeared damaged.
RQ5 therefore ends as a qualified negative: the diagnosis was right, the
mechanism works, and the effect does not pay for its inference cost.

## 7. Cross-cutting findings



1. **Domain fine-tuning beats architecture selection** on this data: +0.240 mAP
   from 419 images, against +0.065 from scaling n→m, or +0.013 from P2 even
   after doubling its training budget.
2. **Small objects are the bottleneck** at every level of analysis: AP_small
   0.06–0.11 against AP_large 0.65–0.79; 94% of misses are small; the frame is
   640×640 letterbox with a median box of ≈11 px.
3. **The default operating point is not the best one**, but the gain from
   tuning it (+0.002…+0.006 F1) is small compared with fine-tuning.
4. **Metric choice decides the "optimal" configuration**: mAP prefers a
   permissive NMS, F1 prefers an aggressive one. Reporting both is necessary.

---

## 8. Limitations

- 120 test images; motorcycle has 31 instances, so its AP is indicative only.
- Four classes, no person/bicycle/train/tram; the COCO 80-class taxonomy is not
  evaluated.
- Single seed, single 30-epoch budget per variant; no noise-floor run, so the
  RQ5 difference is asserted against a paired test but not against run-to-run
  variance.
- Latency measured on a shared T4; repeats varied by up to 20%, so only the
  ordering and the n:m ratio are load-bearing.
- RQ3's manual review covers 12 instances (Loc/Cls/Bkg/Miss at 3 each; no Dupe
  cases exist) and the occlusion attribute was not annotated.
- The v3 export's 1,755 unusable images are a property of the dataset, not of
  the pipeline; a cleaner export would change the absolute numbers.

---

## 9. Deviations from the proposal

1. The test set is not the 200-image Mapillary set annotated by the group. A
   data-quality audit of the public Roboflow export showed only 599 of 2,354
   images carry a valid vehicle class, giving four COCO classes and a frozen
   120-image split; inter-annotator agreement is replaced by the audit.
2. The sweep is 9 conf × 3 NMS IoU instead of 9 × 5, and precision-recall curves
   are not plotted.
3. The manual error review classifies 12–15 instances instead of 30, and the
   illumination attribute is not annotated (only 40 of 599 filenames carry a
   day/night marker).
4. RQ5 reports one seed; the P2 branch starts from random weights by
   construction, so the result bounds the intervention under a matched budget.

---

## 10. Reproduction

```powershell
uv run python scripts\build_hk_dataset.py        # 599-image set, frozen split
uv run comp4330 rq1 --gt data\annotations\hk_test_coco.json --images data\hk\images\test --tag hk_test
uv run comp4330 rq2                              # 27 cells per model
uv run comp4330 rq3                              # TIDE + taxonomy + chi-square
uv run comp4330 models                           # baseline vs P2 budget
uv run comp4330 rq5 --train                      # then: uv run comp4330 rq5
uv run python scripts\plot_rq1.py
uv run python scripts\plot_rq2.py
uv run python scripts\plot_rq3.py
uv run python scripts\plot_rq5.py
```

| artefact | path |
|---|---|
| RQ1 | `results/tables/rq1_scaling.csv`, `rq1_per_class.csv`, `rq1_marginal.csv`, `results/figures/fig_rq1_scaling.png` |
| RQ2 | `results/tables/rq2_sweep.csv`, `rq2_best.csv`, `results/rq2_answer.json`, `results/figures/fig_rq2_sweep.png` |
| RQ3 | `results/tables/rq3_tide.json`, `rq3_tide.csv`, `rq3_error_taxonomy.csv`, `rq3_chi_square.csv`, `results/figures/fig_rq3_errors.png`, `data/annotations/rq3_manual.csv` |
| RQ4 | `results/tables/rq4_params.csv`, `configs/models/yolov8s-p2.yaml` |
| RQ5 | `results/tables/rq5_ablation.csv`, `rq5_per_class.csv`, `rq5_paired_test.csv`, `rq5_errors.csv`, `results/figures/fig_rq5_ablation.png` |
