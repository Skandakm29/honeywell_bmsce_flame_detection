# Flame detection: YOLOX-S-Lite (5 and 5+15 epochs) vs YOLOX-Nano-Lite baseline, at confidence 0.40

**Status:** complete. The pipeline finished on 2026-10-29 at 11:13 UTC (16:43 IST).
**Target board:** TI SK-AM62A-LP. Board performance has **not been measured** yet.
**Source of these numbers:** `MyDrive/HoneywellTraining/experiments/yolox_s_lite_flame/reports_e20/`
(`report.md`, `comparison.csv`, `metrics.json`). The earlier 5-epoch report (`reports/`, [RESULTS_e5.md](RESULTS_e5.md))
is unchanged. Code: [experiments/](../experiments/).

## 1. Summary

Three models were tested on the **same 15,882 Kaggle test images**, with the same code, all at
**confidence 0.40** (fixed in advance, not tuned on any split).

| Model | AP50 | mAP50-95 | F1 | Flames missed | False alarms |
|---|---|---|---|---|---|
| Baseline Nano (416, 20 epochs) | 0.818 | 0.514 | 0.780 | 1,975 | 1,997 |
| S-Lite (640, 5 epochs) | 0.860 | 0.557 | 0.811 | 1,839 | **1,516** |
| **S-Lite (640, 5+15 epochs)** | **0.870** | **0.573** | **0.818** | **1,613** | 1,696 |

The test images contain 9,022 flame boxes in total.

- **The 20-epoch S-Lite is the best model overall.** Compared with the baseline:
  - AP50 is +5.3 points, mAP50-95 is +5.9 and F1 is +3.7.
  - It misses 18% fewer flames and raises 15% fewer false alarms.
- **The extra 15 epochs gave a small gain** over the 5-epoch model: AP50 +1.0, mAP50-95 +1.7, recall +2.5.
  - At 0.40 it trades a little precision (−1.2) for that recall, so it raises 12% more false alarms
    than the 5-epoch model.
- **Training has levelled off.** The per-epoch validation check has been flat since epoch 5+11, so
  more epochs with this recipe are unlikely to help much.
- **Cost:** S-Lite has 4× the parameters, 9× the compute and 4× the file size of the baseline. Its speed
  on the AM62A is still unknown.

## 2. Models

| | Baseline | S-Lite, 5 epochs | S-Lite, 5+15 epochs |
|---|---|---|---|
| Architecture | TI YOLOX-Nano-ti-lite | TI YOLOX-S-ti-lite (edgeai-yolox `cbdd9d8`) | Same as 5 epochs |
| Input size | 416 × 416 | 640 × 640 | 640 × 640 |
| Classes | 1 (`flame`) | 1 (`flame`) | 1 (`flame`) |
| Initial weights | TI COCO checkpoint | TI COCO checkpoint `yolox-s-ti-lite_39p1_57p9` | `checkpoints/s_lite_e5_v1/best.pth` (the 5-epoch model) |
| Training data | Official FASDD split | Kaggle train split (63,546 images) | Kaggle train split (63,546 images) |
| Epochs | 20 (best: 17) | 5 (best: 5) | 15 more (best: 5+14) |
| Checkpoint | `HoneywellTraining/full_v1/best.pth`, reused read-only | `checkpoints/s_lite_e5_v1/best.pth` | `checkpoints/s_lite_e5p15_v1/best.pth` |

## 3. Data and protocol

- **Data:** Kaggle `yuulind/fasdd-cv-coco`, using its own splits.

  | Split | Images | Used for |
  |---|---|---|
  | train | 63,546 | Training the S-Lite models |
  | val | 15,884 | Per-epoch check (a fixed 2,000-image subset) and full-validation results |
  | test | 15,884, of which 15,882 were evaluated (2 quarantined) | The final comparison, run once per model |

  `fire` boxes are the `flame` target. Smoke-only and unannotated images count as negatives, so any
  detection on them is a false alarm.
- **Evaluation**, identical for all three models:
  - Predictions with score ≥ 0.001 go through NMS at IoU 0.65, keeping at most 100 boxes per image.
  - AP is COCO AP (pycocotools). AP does not depend on the confidence threshold.
  - Precision, recall and F1 come from one-to-one matching in score order at IoU 0.5, at confidence 0.40.
- **Confidence 0.40** is the baseline's reported operating point. It was fixed before testing. The test
  split was not used to choose checkpoints, tune settings or pick thresholds.
- **Latency:** Colab NVIDIA L4, batch 1, float32, CUDA-synchronised, after 10 warm-up images.
  **This is not AM62A performance.**

## 4. Results: test split (15,882 images, confidence 0.40)

| Model | AP50 | AP75 | mAP50-95 | Precision | Recall | F1 | False alarms/image | Missed-flame rate |
|---|---|---|---|---|---|---|---|---|
| Baseline Nano (416, 20 ep) | 0.8175 | 0.5589 | 0.5140 | 0.7792 | 0.7811 | 0.7801 | 0.1257 | 21.9% |
| S-Lite (640, 5 ep) | 0.8601 | 0.6133 | 0.5568 | **0.8257** | 0.7962 | 0.8107 | **0.0955** | 20.4% |
| **S-Lite (640, 5+15 ep)** | **0.8702** | **0.6288** | **0.5734** | 0.8137 | **0.8212** | **0.8175** | 0.1068 | **17.9%** |
| 5+15 vs baseline | +0.053 | +0.070 | +0.059 | +0.035 | +0.040 | +0.037 | −15% | −4.0 pts |
| 5+15 vs 5 ep | +0.010 | +0.016 | +0.017 | −0.012 | +0.025 | +0.007 | +12% | −2.5 pts |

| Model | Flames found (TP) | Flames missed (FN) | False alarms (FP) |
|---|---|---|---|
| Baseline | 7,047 | 1,975 | 1,997 |
| S-Lite, 5 ep | 7,183 | 1,839 | 1,516 |
| S-Lite, 5+15 ep | 7,409 | 1,613 | 1,696 |

## 5. Results: full validation split (confidence 0.40)

| Model | Precision | Recall | F1 | False alarms/image | Missed-flame rate |
|---|---|---|---|---|---|
| Baseline | 0.7125 | 0.5726 | 0.6349 | 0.1823 | 42.7% |
| S-Lite, 5 ep | 0.7399 | 0.6001 | 0.6627 | 0.1663 | 40.0% |
| S-Lite, 5+15 ep | 0.7191 | 0.6529 | 0.6844 | 0.2011 | 34.7% |

**Validation is harder than test for every model.** All three score lower on validation than on test,
and the ranking by F1 is the same on both. This is why the per-epoch numbers in `train.log` (section 8)
are lower than the test numbers. The reason the Kaggle validation split is harder has not been
investigated.

## 6. Model cost

| Model | Parameters | Compute (GMACs) | ONNX size (MB) | Checkpoint (MB) | GPU-only median (ms) | End-to-end median (ms) | End-to-end FPS |
|---|---|---|---|---|---|---|---|
| Baseline (416) | 2,241,998 | 1.452 | 8.70 | 8.74 | 11.52 | 21.60 | 45.8 |
| S-Lite, 5 ep (640) | 8,938,030 | 13.291 | 34.33 | 34.32 | 11.52 | 22.72 | 43.5 |
| S-Lite, 5+15 ep (640) | 8,938,030 | 13.291 | 34.33 | 34.32 | 11.59 | 22.77 | 42.7 |

- **Measured:** latency and FPS, on the L4 GPU. These differ from the T4 numbers in RESULTS_e5.md
  because the GPU is different.
- **Calculated:** parameters, compute and file sizes.
- **Median vs mean:** latency is the median time per image. FPS is 1000 ÷ the mean end-to-end time.
- On the GPU all three take about the same time, because none is big enough to keep it busy. On the
  AM62A the 9× difference in compute is expected to matter. That has not been measured.

## 7. Other fixed thresholds (test split)

For reference only. The operating threshold was fixed at 0.40 and was **not** chosen from this table.
A different threshold must be chosen on validation, not test.

| Model | Confidence | Precision | Recall | F1 | FP | FN | False alarms/image |
|---|---|---|---|---|---|---|---|
| Baseline | 0.25 | 0.6718 | 0.8238 | 0.7401 | 3630 | 1590 | 0.2286 |
| Baseline | 0.50 | 0.8403 | 0.7475 | 0.7912 | 1282 | 2278 | 0.0807 |
| S-Lite, 5 ep | 0.25 | 0.7242 | 0.8450 | 0.7799 | 2904 | 1398 | 0.1828 |
| S-Lite, 5 ep | 0.50 | 0.8760 | 0.7614 | 0.8147 | 972 | 2153 | 0.0612 |
| S-Lite, 5+15 ep | 0.25 | 0.7288 | 0.8592 | 0.7886 | 2885 | 1270 | 0.1817 |
| S-Lite, 5+15 ep | 0.50 | 0.8607 | 0.7956 | 0.8269 | 1162 | 1844 | 0.0732 |

All thresholds (0.10, 0.25, 0.40, 0.50 and 0.75, on validation and test) are in
`reports_e20/report.md` and `reports_e20/fixed_thresholds.csv`.

## 8. S-Lite continuation training (15 epochs)

| Setting | Value |
|---|---|
| Hardware | Google Colab NVIDIA L4; Python 3.12.3, PyTorch 2.5.1+cu124 |
| Starting point | The 5-epoch model's `best.pth` (weights only: new optimizer and EMA) |
| Batch | 16 × accumulate 1 (seed 42) |
| Optimizer | AdamW, lr 3e-4 (same as the 5-epoch run), weight decay 5e-4 |
| Learning-rate schedule | Restarted: 200-update warmup, then cosine decay to 0.1× over the 15 epochs |
| Other settings | EMA, gradient clipping at 10, early-stop patience 5 epochs (not triggered) |
| Augmentation | Horizontal flip and HSV, switched off for the last 2 epochs (same recipe as the baseline) |
| Checkpoint selection | Best AP50:95 on the same 2,000-image validation subset as the 5-epoch run |
| Checkpoints | Every 250 batches, written atomically to Google Drive |
| Resume events | 1: Google Drive disconnected during epoch 11. Training resumed from the last checkpoint (batch 530) after it was verified; the loss values after the resume match the ones before it. |
| Total training time | 16,675 s (≈ 4 h 38 min), about 18.4 min per epoch |

Per-epoch check on the 2,000-image validation subset. AP does not depend on the threshold. Precision
and recall in `history.json` use each epoch's best-F1 confidence (0.30), not 0.40, so they are not
shown here.

| Epoch (overall) | Train loss | AP50 | AP50:95 | Minutes |
|---|---|---|---|---|
| 5+1 (6) | 4.953 | 0.5407 | 0.2595 | 18.5 |
| 5+2 (7) | 4.827 | 0.6662 | 0.3302 | 18.3 |
| 5+3 (8) | 4.703 | 0.6868 | 0.3587 | 18.5 |
| 5+4 (9) | 4.590 | 0.6968 | 0.3707 | 18.4 |
| 5+5 (10) | 4.480 | 0.6990 | 0.3729 | 18.4 |
| 5+6 (11) | 4.355 | 0.7039 | 0.3762 | 18.4 |
| 5+7 (12) | 4.226 | 0.7067 | 0.3805 | 18.4 |
| 5+8 (13) | 4.122 | 0.7101 | 0.3856 | 18.4 |
| 5+9 (14) | 3.977 | 0.7138 | 0.3875 | 18.4 |
| 5+10 (15) | 3.873 | 0.7168 | 0.3883 | 18.4 |
| 5+11 (16) | 3.759 | 0.7183 | 0.3908 | 20.0\* |
| 5+12 (17) | 3.660 | 0.7201 | 0.3907 | 18.5 |
| 5+13 (18) | 3.566 | **0.7204** | 0.3917 | 18.4 |
| **5+14 (19)** | 3.437 | 0.7188 | **0.3930** | 18.4 |
| 5+15 (20) | 3.330 | 0.7185 | 0.3924 | 18.4 |

For comparison, the 5-epoch model ended at AP50 0.6978 and AP50:95 0.3719 on the same subset.

\* Epoch 5+11 includes the resume.

- **The drop at 5+1 is expected.** The learning rate restarted from its warmup and peak, which
  disturbs the model briefly. It was back above the 5-epoch level by 5+5.
- **The model plateaued from 5+11.** AP50:95 moved by less than 0.003 over the last 5 epochs, while the
  training loss kept falling.
- **`best.pth` is epoch 5+14** (highest AP50:95). All test results in this report use it.

## 9. Fairness caveats and limitations

- **Different training data:** the baseline was trained on the official FASDD split at 416 px.
  S-Lite was trained on the Kaggle split at 640 px.
- **Possible overlap:** the baseline's training images may also appear in Kaggle val/test. That would favour
  the baseline. Its split files are not available, so this could not be checked. The comparison is
  indicative, not conclusive.
- **Not a single 20-epoch run:** S-Lite 5+15 is 5 epochs followed by a separate 15-epoch run with a
  restarted learning-rate schedule, a fresh optimizer and EMA. A single 20-epoch schedule might give
  different results.
- **One run per model:** each model was trained once, so there is no spread across seeds. Small
  differences (about 1 point) between the two S-Lite models may be within run-to-run noise.
- **GPU differences:** these evaluations ran on an L4; the 5-epoch report used a T4. The S-Lite 5-epoch
  test AP is the same on both (0.8601 AP50, 0.5568 mAP50-95). A few TP/FP counts differ by 1–6 between
  the two GPUs.
- **No validation loss:** the training loop does not compute one.
- **Duplicates:** exact duplicate images across the Kaggle splits are reported, not removed.
- **Not measured:** AM62A latency, TIDL compilation and board memory, because the board has not been used yet.
- **Still images only:** results are per still image. Alarm behaviour on video is untested.

## 10. Output files (Google Drive)

All paths are under `MyDrive/HoneywellTraining/experiments/yolox_s_lite_flame/`.

| Folder | Contents |
|---|---|
| `checkpoints/s_lite_e5p15_v1/` | `best.pth` (epoch 5+14), `last.pth`, `history.json`, `config.json` |
| `logs/e5p15/` | `pipeline.log`, `train.log`, `status.json`, `resume_events.json`, `gpu_usage.csv` |
| `configs/` | `e5p15_v1.json`, `e5p15_init_checkpoint.json` |
| `validation/{baseline_c040,s_lite_e5_c040,s_lite_e20_c040}/` | `metrics.json`, threshold-sweep and fixed-threshold CSVs, `curves.png` |
| `test/{baseline_c040,s_lite_e5_c040,s_lite_e20_c040}/` | Same files as `validation/`, for the test split |
| `exports/s_lite_e20/` | `flame.onnx`, `export.json`, `model_contract.json` |
| `plots/` | `comparison_e20_metrics.png`, `comparison_e20_pr_false_alarms.png`, `e5p15/training_curves.png` |
| `reports_e20/` | `report.md`, `comparison.csv`, `fixed_thresholds.csv`, `metrics.json` |

## 11. Possible next steps

1. **Compile S-Lite with TIDL** and benchmark it on the SK-AM62A-LP. Board speed decides whether the
   9× compute is acceptable. If it is too slow, try S-Lite at a smaller input size (for example 416).
2. **Choose the operating threshold on validation** for the chosen model, depending on whether missed
   flames or false alarms matter more.
3. **Test on video clips** to check alarm behaviour over time.
4. **Look into why Kaggle validation is harder than test** (image sources, flame size).
5. More epochs with this recipe are **not** recommended, because training has plateaued.
