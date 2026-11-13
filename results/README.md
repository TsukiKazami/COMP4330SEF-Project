# Results artefacts

Every file here is produced by a command in the top-level README and discussed
in `docs/results.md`. `rq1_*` rows are yolov8 n/s/m on the frozen 120-image test
split; `rq2`–`rq5` use the operating point conf 0.25 / NMS IoU 0.3.

| file | what it holds |
|---|---|
| `tables/rq1_scaling.csv` | params, AP, P/R/F1, latency, size-bucket AP per model |
| `tables/rq1_per_class.csv` | AP per class per model |
| `tables/rq1_marginal.csv` | marginal AP per +1M parameters between sizes |
| `tables/rq2_sweep.csv` | 9 confidences x 3 NMS IoUs x 3 models |
| `tables/rq2_best.csv` | per-model optimum and the shared optimum |
| `rq2_answer.json` | the "one configuration for all sizes?" verdict with regrets |
| `tables/rq3_tide.json` | TIDE dAP decomposition per model (raw) |
| `tables/rq3_tide.csv` | the same, tidy, for plotting |
| `tables/rq3_error_taxonomy.csv` | one row per error instance at the operating point |
| `tables/rq3_chi_square.csv` | error type x scale bucket and x class tests |
| `tables/rq4_params.csv` | baseline vs P2 parameters, GFLOPs, strides, anchors |
| `tables/rq5_ablation.csv` | 30-epoch baseline vs P2 |
| `tables/rq5_*_60ep.csv` | the matched 60-epoch follow-up |
| `tables/rq5_paired_test.csv` | paired t-test on per-image F1 |
| `figures/fig_rq*.png` | one figure per research question |
| `figures/rq3_crops/` | the 12 manually reviewed error instances |