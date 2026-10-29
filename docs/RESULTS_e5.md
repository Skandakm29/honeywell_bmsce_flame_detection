# Flame detection: YOLOX-S-Lite (5 epochs) vs YOLOX-Nano-Lite baseline

**Status:** complete. The pipeline finished on 2026-10-29 at 04:03 UTC.
**Target board:** TI SK-AM62A-LP. Board performance has **not been measured** yet.
**Source of these numbers:** `MyDrive/HoneywellTraining/experiments/yolox_s_lite_flame/reports/`
(`report.md`, `comparison.csv`, `metrics.json`).

## 1. Summary

On the same 15,882 Kaggle test images, evaluated with the same code, **YOLOX-S-ti-lite beats the
YOLOX-Nano-ti-lite baseline on every accuracy metric**:

| | AP50 | mAP50-95 | F1 |
|---|---|---|---|
| Baseline | 0.818 | 0.514 | 0.775 |
| S-Lite | 0.860 | 0.557 | 0.805 |

S-Lite also makes 17% fewer false alarms per image. It needs this many times more than the baseline:

| | S-Lite vs baseline |
|---|---|
| Parameters | 4× |
| Compute | 9× |
| Model size | 4× |

Whether S-Lite runs fast enough on the AM62A is still unknown.

## 2. Models

| | Baseline (model A) | S-Lite (model B) |
|---|---|---|
| Architecture | TI YOLOX-Nano-ti-lite | TI YOLOX-S-ti-lite (edgeai-yolox `cbdd9d8`) |
| Input size | 416 × 416 | 640 × 640 |
| Classes | 1 (`flame`) | 1 (`flame`) |
| Initial weights | TI COCO checkpoint | TI COCO checkpoint `yolox-s-ti-lite_39p1_57p9` (SHA-256 `d29abb5f…`) |
| Training data | Official FASDD split | Kaggle `yuulind/fasdd-cv-coco` train split (63,546 images) |
| Epochs | 20 (best: epoch 17) | 5 (best: epoch 5) |
| Checkpoint | `HoneywellTraining/full_v1/best.pth`, reused read-only | `checkpoints/s_lite_e5_v1/best.pth` |

## 3. Protocol

- **Data:** Kaggle `yuulind/fasdd-cv-coco`, using its own train/val/test splits.
  - Train: 63,546 images. Val: 15,884. Test: 15,884, of which 15,882 were evaluated after 2 were quarantined.
  - `fire` boxes are the `flame` target. Smoke-only and unannotated images count as negatives.
- **Evaluation:** both models are scored on the same Kaggle validation and test images, with the same code:
  - Predictions with score ≥ 0.001 go through NMS at IoU 0.65, keeping at most 100 boxes per image.
  - AP uses COCO AP (pycocotools).
  - Precision, recall and F1 come from one-to-one matching in score order at IoU 0.5.
- **Confidence threshold:** each model uses the threshold that gives its maximum F1 on Kaggle **validation**.
  The test split was evaluated only once per model, with that threshold. The test split was not used to
  choose checkpoints, tune settings or pick thresholds.
- **Latency:** measured on a Colab Tesla T4 at batch 1 in float32, CUDA-synchronised, after 10 warm-up
  images. **This is not AM62A performance.**

## 4. Results: test split (15,882 images)

| Model | AP50 | AP75 | mAP50-95 | Precision | Recall | F1 | False alarms/image |
|---|---|---|---|---|---|---|---|
| YOLOX-Nano-ti-lite (baseline, 416) | 0.8175 | 0.5590 | 0.5140 | 0.7648 | 0.7857 | 0.7751 | 0.1373 |
| **YOLOX-S-ti-lite (640)** | **0.8601** | **0.6135** | **0.5568** | **0.8011** | **0.8086** | **0.8048** | **0.1140** |
| Change | +0.043 | +0.055 | +0.043 | +0.036 | +0.023 | +0.030 | −17% |

Precision, recall, F1 and false alarms are measured at each model's validation-selected confidence:
**0.38** for the baseline and **0.36** for S-Lite.

## 5. Model cost

| Model | Parameters | Compute (GMACs) | ONNX size (MB) | Checkpoint size (MB) | GPU-only median (ms) | End-to-end median (ms) | End-to-end FPS |
|---|---|---|---|---|---|---|---|
| Baseline (416) | 2,241,998 | 1.452 | 8.70 | 8.74 | 10.68 | 18.06 | 53.3 |
| S-Lite (640) | 8,938,030 | 13.291 | 34.33 | 34.32 | 10.67 | 19.67 | 50.0 |

- **Measured:** latency and FPS, on the T4 GPU.
- **Calculated:** parameters, compute (GMACs) and file sizes.
- **GPU-only time is the same for both models** because neither is big enough to keep the T4 busy.
  On the AM62A, the 9× difference in compute is expected to matter. That speed has not been measured.

## 6. Fixed confidence thresholds

| Model | Split | Confidence | Precision | Recall | F1 | TP | FP | FN | False alarms/image | Missed-flame rate |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | validation | 0.10 | 0.3978 | 0.7570 | 0.5215 | 9482 | 14356 | 3044 | 0.9038 | 0.2430 |
| baseline | validation | 0.25 | 0.5878 | 0.6549 | 0.6195 | 8203 | 5752 | 4323 | 0.3621 | 0.3451 |
| baseline | validation | 0.50 | 0.7768 | 0.5109 | 0.6164 | 6399 | 1839 | 6127 | 0.1158 | 0.4891 |
| baseline | validation | 0.75 | 0.9361 | 0.2211 | 0.3577 | 2769 | 189 | 9757 | 0.0119 | 0.7789 |
| baseline | test | 0.10 | 0.4874 | 0.8757 | 0.6263 | 7901 | 8309 | 1121 | 0.5232 | 0.1243 |
| baseline | test | 0.25 | 0.6722 | 0.8238 | 0.7403 | 7432 | 3625 | 1590 | 0.2282 | 0.1762 |
| baseline | test | 0.50 | 0.8402 | 0.7474 | 0.7911 | 6743 | 1282 | 2279 | 0.0807 | 0.2526 |
| baseline | test | 0.75 | 0.9582 | 0.4875 | 0.6462 | 4398 | 192 | 4624 | 0.0121 | 0.5125 |
| s_lite | validation | 0.10 | 0.4279 | 0.7989 | 0.5573 | 10007 | 13381 | 2519 | 0.8424 | 0.2011 |
| s_lite | validation | 0.25 | 0.6147 | 0.6953 | 0.6525 | 8709 | 5460 | 3817 | 0.3437 | 0.3047 |
| s_lite | validation | 0.50 | 0.8104 | 0.5280 | 0.6394 | 6614 | 1547 | 5912 | 0.0974 | 0.4720 |
| s_lite | validation | 0.75 | 0.9529 | 0.2294 | 0.3697 | 2873 | 142 | 9653 | 0.0089 | 0.7706 |
| s_lite | test | 0.10 | 0.5351 | 0.8973 | 0.6704 | 8095 | 7033 | 927 | 0.4428 | 0.1027 |
| s_lite | test | 0.25 | 0.7242 | 0.8450 | 0.7799 | 7624 | 2904 | 1398 | 0.1828 | 0.1550 |
| s_lite | test | 0.50 | 0.8760 | 0.7614 | 0.8147 | 6869 | 972 | 2153 | 0.0612 | 0.2386 |
| s_lite | test | 0.75 | 0.9705 | 0.5471 | 0.6997 | 4936 | 150 | 4086 | 0.0094 | 0.4529 |

## 7. S-Lite training

| Setting | Value |
|---|---|
| Hardware | Google Colab Tesla T4; Python 3.12.3, PyTorch 2.5.1+cu124 |
| Batch | 16 × accumulate 1 (seed 42) |
| Optimizer | AdamW, lr 3e-4, weight decay 5e-4 |
| Learning-rate schedule | 200-update warmup, then cosine decay to 0.1× |
| Other settings | EMA, gradient clipping at 10 |
| Augmentation | Horizontal flip and HSV, switched off for the last 2 epochs (same recipe as the baseline) |
| Checkpoint selection | Best AP50:95 on a 2,000-image subset of Kaggle val (seed 42) |
| Checkpoints | Every 250 batches, written atomically to Google Drive |
| Resume events | 1: runtime reset during epoch 4. Training continued from batch 1834 after the checkpoint was verified; it did not restart. |
| Total training time | 10,330 s (≈ 2 h 52 min) |

The table below is for the 2,000-image validation subset. Precision, recall and F1 are measured at the
confidence with maximum F1, which was 0.30 in epochs 1–4.

| Epoch | Train loss | AP50 | AP50:95 | AP_small | AR100 | Precision | Recall | F1 | No-flame images with a false alarm | Minutes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.262 | 0.4985 | 0.2136 | 0.1226 | 0.4825 | 0.4796 | 0.5255 | 0.5015 | 7.89% | 36.9 |
| 2 | 5.353 | 0.6280 | 0.3033 | 0.1746 | 0.5187 | 0.6289 | 0.6039 | 0.6161 | 2.22% | 37.0 |
| 3 | 4.909 | 0.6709 | 0.3442 | 0.1977 | 0.5473 | 0.6519 | 0.6312 | 0.6414 | 2.76% | 36.7 |
| 4 | 4.442 | 0.6951 | 0.3672 | 0.2085 | 0.5611 | 0.6688 | 0.6555 | 0.6621 | 2.37% | 25.5\* |
| 5 | 4.099 | 0.6978 | **0.3719** | see `history.json` | see `history.json` | see `history.json` | see `history.json` | see `history.json` | see `history.json` | 36.1 |

\* Epoch 4's time covers only the part trained after the resume.

## 8. The baseline as originally reported

These are the baseline's own numbers. They were measured on a **different** 2,000-image subset of the
official FASDD validation split, so they cannot be compared directly with sections 4–7.

| Best epoch | AP50 | AP50:95 | Precision | Recall | F1 | Confidence | No-flame images with a false alarm | ONNX parity check |
|---|---|---|---|---|---|---|---|---|
| 17 of 20 | 0.6269 | 0.3292 | 0.7010 | 0.5591 | 0.6221 | 0.40 | 2.38% | passed |

The Kaggle test split gives both models higher scores than either validation set, so these numbers are
lower than the ones in section 4.

## 9. Fairness caveats and limitations

- **Different training data:** the baseline was trained on the official FASDD split at 416 px for 20 epochs.
  S-Lite was trained on the Kaggle split at 640 px for 5 epochs.
- **Possible overlap:** the baseline's training images may also appear in Kaggle val/test. That would favour
  the baseline. Its split files are not available, so this could not be checked. The comparison is
  indicative, not conclusive.
- **Short training:** S-Lite trained for 5 epochs only, and AP was still rising at epoch 5.
- **Augmentation:** with 5 epochs, augmentation was switched off for 40% of training (the last 2 epochs).
- **No validation loss:** the training loop does not compute one.
- **Duplicates:** exact duplicate images across the Kaggle splits are reported, not removed.
- **Not measured:** AM62A latency, TIDL compilation and board memory, because the board has not been used yet.
- **Still images only:** results are per still image. Alarm behaviour on video is untested.

## 10. Output files (Google Drive)

All paths are under `MyDrive/HoneywellTraining/experiments/yolox_s_lite_flame/`.

| Folder | Contents |
|---|---|
| `checkpoints/s_lite_e5_v1/` | `best.pth`, `last.pth`, `history.json`, `config.json` |
| `logs/` | `pipeline.log`, `train.log`, `status.json`, `resume_events.json`, `gpu_usage.csv` |
| `configs/` | `e5_v1.json`, `run_settings.json`, `baseline_checkpoint.json` |
| `validation/{s_lite,baseline}/` | `metrics.json`, CSV files for the threshold sweep and fixed thresholds, `curves.png`, review sheets |
| `test/{s_lite,baseline}/` | Same files as `validation/`, for the test split |
| `exports/{s_lite,baseline}/` | `flame.onnx`, `export.json`, `model_contract.json` |
| `plots/` | `training_curves.png`, `comparison_metrics.png`, `comparison_pr_false_alarms.png`, `history.csv` |
| `reports/` | `report.md`, `comparison.csv`, `fixed_thresholds.csv`, `metrics.json` |

## 11. Possible next steps

1. **Compile S-Lite with TIDL** and benchmark it on the SK-AM62A-LP. Board speed decides whether S-Lite's
   9× compute is acceptable.
2. **Train longer:** a fresh 10-epoch run, or 5 more epochs continuing from this model.
3. **Report at confidence 0.40** for both models, as an extra fixed threshold.
