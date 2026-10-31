# YOLOv9-S-TI-Lite: measured flame detection results

Training and full evaluation completed on the lab RTX A5000. The copied results archive passed SHA256 and ZIP CRC verification on 31 October 2026. All three full-run checkpoints loaded strictly on the laptop; the best checkpoint produced a finite CPU forward pass. Numeric results below come from the saved reports, not the feasibility pilots.

| Evaluation | Images | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Full validation | 15,884 | 74.60% | 44.31% | 80.65% | 58.73% | 67.97% |
| Test | 15,884 | 88.41% | 59.94% | 88.05% | 79.14% | 83.36% |

Precision/recall/F1 use confidence0.40 and one-to-one matching IoU0.50. AP uses COCO bbox IoU0.50:0.95, score floor0.001, NMS0.65 and max100 detections/image. Test TP7,140/FP969/FN1,882; full-validation TP7,357/FP1,765/FN5,169. FP counts boxes, not video alarm events.

## Training and selection

TI source f272548ef0fe96046f1bf2c3833e040a1ecf8e44; verified COCO initialization; one-class RGB0..1 adapter with TI Lite graph changes. Kaggle FASDD-CV-COCO version2:63,544 training images after two invalid records were quarantined,15,884 validation and15,884 test. Fire/flame boxes are the sole target; smoke-only/background images are negatives. Duplicates are reported, not removed; scene independence is unproven.

Three LR trials (1e-4,3e-4,1e-3), each3epochs on the same8,192 training images, selected1e-4 by monitor-validation AP. Fresh full training:640px, batch16/accumulation2, AdamW, AMP, EMA, warmup/cosine, seed42. Requested maximum40epochs; minimum30/patience10 stopped at30; best checkpoint epoch20. Mosaic probability0.5; configured close-mosaic last10 of40 epochs, so early stopping at30 did not run that final augmentation-off phase. No globally optimal hyperparameter claim.

## Comparison with existing YOLOX results

The existing RESULTS_e20.md reports Nano test mAP50:95=51.40% and S-Lite=57.34%, versus59.94% here. YOLOv9's reported test precision/F1 are higher, but recall79.14% is lower than S-Lite82.12%. Full-validation F1=67.97%, slightly below S-Lite68.44%. It is not uniformly superior. Teammate test evaluation used15,882 images versus15,884 here; exact common-image comparison remains pending. Architecture, training recipe, preprocessing, input resolution and Nano training splits differ; do not attribute gains solely to architecture. The Kaggle test has already been examined by the team.

## Export and retained artifacts

ONNX checker and CPU ONNX Runtime parity passed on3inputs. Max box error0.005859375input pixels (atol0.05), score error7.510185e-6. Inference parameters7,195,619; ONNX28,952,675bytes. SHA256332cb480c4a4f3d8cdfbb3f5545a2e08211f6ae3db492d1d08d22318e2c78544. Selected checkpoint SHA25648d8e654ac1e3381e122a52cdfca100d1840e8cf6db0294ac882b52ab1c4a51f.

Small original reports, configuration and history are under results/yolov9_final/. Large models, optimizer checkpoints, predictions/manifests and the verified1.4GB results archive remain in the laptop's ignored results/lab_backup_20261031/ directory. Dataset and weights are excluded from ordinary Git.

AM62A compilation, board latency/memory, INT8 quality,1TOPS compliance, video alarms and the mentor's complete hard-negative checklist remain unverified. Higher test metrics do not establish deployment readiness.

## Reproduce

Source: experiments/yolov9_flame/. For Windows, clone TexasInstruments/edgeai-tensorlab and check out the TI commit above. Run build_lab_package.py with --tensorlab pointing to that clone and --output Honeywell-Lab.zip. It embeds only six checksum-verified vendor modules plus their license. Follow START_HERE.md in the resulting archive; no GitHub upload of weights/data is needed. The generic Colab notebook is an alternative, not the historical lab environment. Nine engineering tests (including bitwise EMA equivalence and model ONNX parity) passed locally; the lab launcher also required tests/preflight before training. Checkpoint recovery is documented and source/config/manifest fingerprint guarded.
