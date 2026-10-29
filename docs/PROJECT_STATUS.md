# Project handoff for teammates and AI agents
Updated 29 October 2026 (Asia/Kolkata). Read this first; historical experiments are labelled below.

## Objective and constraints
Build visible-flame localization and an eventual embedded video alarm pipeline for the Honeywell BMSCE project (six students). Smoke-only images are negatives. Use simple, reproducible changes; do not over-engineer. Deployment ceiling: 1 TOPS, not yet verified. ONNX handover; TI board/camera unavailable. SK-AM62A-LP is provisional. Numerical flame-size, miss-rate, false-alarm and alarm-delay acceptance targets remain undefined. Honeywell supplies no training data; it may provide unseen final test data.

## Current state
The 20-epoch full_v1 baseline completed on Colab Tesla T4 on 27 October at 23:05 IST. Coding agent stays on the laptop; Colab/lab machines execute code only. Lab RTX 5000 is reported, but its OS/driver and execution remain unverified. No training or monitors are currently active. Both upload and training monitors were deleted after their work finished. Colab disconnected after successful training/export. No cloud cleanup has occurred.

## Data and leakage precautions
FASDD_CV V9 (95,314 source images), CC BY-SA 4.0. Dataset source and hashes: DATASET.md.
Complete raw ZIP MD5, JPEG decode/dimensions, box checks and image hashes passed. Eight invalid records quarantined, 140 exact duplicates and 17,913 cross-split near-duplicate candidates excluded. Priority test > validation > train; original split membership preserved. Perceptual filtering is conservative and does not prove all source/video scenes are independent. Label/domain review remains incomplete; collection includes web/graphics/nonindustrial imagery.

| Split | Clean images | Usage |
|---|---:|---|
| Train | 34,618 | Actual full_v1 training |
| Monitor validation | 2,000 | Subset of full validation, used each epoch |
| Full validation | 26,765 | Full evaluation pending; includes the monitor subset |
| Held-out test | 15,870 | No test predictions/metrics yet; keep unused during tuning |

Train contains 11,178 flame-positive images and 28,719 flame boxes. Smoke/background negatives are retained.

## Experiments and evidence
| Experiment | Data/compute | Result and interpretation |
|---|---|---|
| CPU feasibility pilot | 256 train / 64 val, five epochs, laptop CPU | AP50 0.1826; AP50:95 0.0682. Engineering feasibility only; different splits, not directly comparable to full_v1. |
| CPU host-thread timing | Pilot batch 4 | Four threads ~0.515 s/batch versus fourteen ~17.147; full CPU training impractical here. |
| Checkpoint resume integration | Four batches, same CPU | Interrupted/resumed raw and EMA weights matched uninterrupted run exactly; two optimizer updates. GPU cross-device identity not promised. |
| full_v1 | 34,618 train / 2,000 monitor, Tesla T4, 20 epochs | Best epoch 17; AP50:95 0.32924849, AP50 0.62686530. Saved best differs from final epoch 20. |
| Export verification | Selected epoch 17, four validation inputs | ONNX checker passed; max absolute error 0.00120544 within atol 0.002 / rtol 0.001. |

Selected operating point (validation-derived, provisional): confidence 0.40, matching IoU 0.50; TP 804, FP 343, FN 634; precision 0.70096, recall 0.55911, F1 0.62205. Small-object AP 0.17295. Negative-image FP proportion 0.02376 is not a video false-alarm rate. Neither AP is ordinary accuracy. Thresholds are not calibrated probabilities or approved alarm requirements.

Recipe: configs/full_v1.json; TI source revision cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6, 2,241,998 parameters, 462 pretrained tensors matched, new flame classifier. Python 3.12.13, Torch 2.5.1+cu124, Torchvision 0.20.1+cu124; other direct dependencies pinned in requirements.txt. 416 input, batch 8 / accumulation 2, AdamW LR 0.0003, weight decay 0.0005, EMA, 200-update warmup/cosine, HSV/flip augmentation disabled for final two epochs, patience 5, seed 42. Checkpoints every 250 optimizer updates and epoch boundaries. Pinned code implements TI BGR/raw 0–255/top-left letterbox preprocessing.

Epoch curve plateaus near AP50:95 0.328–0.329 from epochs 12–20. More epochs alone is not the next priority. No competing architecture has been trained on the same splits. Selection reflects documented TI integration, not proven flame superiority or production readiness. Full results: ../results/full_v1_summary.json and ../results/full_v1_history.json.

## YOLOX-S-ti-lite comparison (29 October, Kaggle split)
Separate experiment; full_v1 was reused read-only and not modified. Data: Kaggle `yuulind/fasdd-cv-coco` with its own splits (63,546 train / 15,884 val / 15,882 evaluated test images), not the cleaned splits above. Code: ../experiments/ (Colab notebooks run_colab_e5.ipynb and run_colab_e5p15.ipynb). TI YOLOX-S-ti-lite, 640 input, single `flame` class, trained 5 epochs (Colab T4), then 15 more epochs from that best.pth at the same LR 3e-4 (Colab L4). On the Kaggle test split at confidence 0.40 (fixed in advance), all three models scored with the same code:

| Model | AP50 | mAP50-95 | Precision | Recall | F1 | False alarms/image |
|---|---:|---:|---:|---:|---:|---:|
| full_v1 baseline (Nano, 416) | 0.8175 | 0.5140 | 0.7792 | 0.7811 | 0.7801 | 0.1257 |
| S-Lite, 5 epochs | 0.8601 | 0.5568 | 0.8257 | 0.7962 | 0.8107 | 0.0955 |
| S-Lite, 5+15 epochs | 0.8702 | 0.5734 | 0.8137 | 0.8212 | 0.8175 | 0.1068 |

Indicative only: the baseline was trained on the official split, which may overlap Kaggle val/test. S-Lite costs 4× parameters and 9× GMACs (13.3 vs 1.5); AM62A/TIDL speed is NOT measured. S-Lite training plateaued from epoch 5+11. Full reports: RESULTS_e5.md and RESULTS_e20.md. Weights and run folders are on Drive under HoneywellTraining/experiments/yolox_s_lite_flame/.

## Artifact state
Drive run folder: https://drive.google.com/drive/folders/1c6NBxvgDwOQYrOMTDYjklne3NVDWJyOj
Drive contains best.pth, last.pth, flame.onnx, config/history/export/model_contract/operating_metrics JSON, training.log, validation_predictions.json and validation_examples.jpg. Local downloaded model backup is NOT yet verified. ONNX SHA256: 749db8cecc21b43a8b31262d3ff5f0e2a0036f83e3b4aa73ff656e562af67458. Colab CPU inference ~73.91 ms is host-specific, not TI or end-to-end latency. Models and data stay outside ordinary Git; see ARTIFACTS.md.

## Proposed next steps (not yet executed)
1. Download the complete run folder; verify model hashes and readable checkpoints before cloud cleanup.
2. Evaluate epoch 17 on all 26,765 cleaned validation images (tooling added: scripts/evaluate.py and notebooks/colab_runbook.ipynb; not yet run on real data). Review about 100 misses, 100 false detections and 50 correct detections; group by flame size, lighting, occlusion, domain and label/box quality. Keep test unused.
3. Correct only evidenced preprocessing/annotation problems. Version corrections; reevaluate baseline on the same corrected validation data. Check source/scene leakage without discarding difficult cases merely to inflate scores.
4. Run one controlled improvement at a time on the lab GPU: higher resolution if small-flame misses dominate; relevant training examples/confusing negatives if domain gaps dominate. Preserve 1 TOPS/latency constraints. Separate named runs; do not resume full_v1 with changed data/recipe. Gains are hypotheses, not promises.
5. Select operating threshold against agreed requirements, freeze model/threshold, then held-out and independent video tests. Validate temporal alarms, obstruction/tamper detection, calibration, telemetry and TI deployment. Those runtime modules are not implemented yet.

## Agent and collaboration rules
Preserve existing work and frozen assets. Never claim planned work is complete. Update this handoff, experiment evidence and relevant docs after meaningful changes. Use feature branches and reviewed PRs; do not directly push/merge main. Do not upload datasets, weights, environments or secrets to Git. Do not start duplicate training or tune against the held-out test. No new scheduled monitors unless requested. No deletion of cloud data/runtime until the GitHub push AND separate model/checkpoint backup are verified. Future agents need the frozen manifest hashes, artifact access and this repository; no reliance on chat history.
