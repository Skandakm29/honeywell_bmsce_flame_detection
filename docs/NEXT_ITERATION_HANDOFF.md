# Flame detection: next iteration handoff

**Status date:** 31 October 2026  
**Assignment:** improve and evaluate the existing flame detector as a video alarm system, with tightly bounded training time.

## 1. Scope and ownership

Our task is **visible-flame detection in RGB camera images/video**. Smoke-only scenes are negatives. In images containing both flame and smoke, only flame boxes are detection targets. The source dataset calls this class `fire`; its authors describe it as flame objects.

Another teammate is checking FASDD coverage of welding, grinding sparks, sunlight, reflections, lights, clothing, heaters and screens using CLIP-assisted search. **Do not repeat that dataset-search assignment here.** Its output can inform a later targeted improvement, but this task can start without it.

This assignment has three priorities:

1. Select an operating threshold using existing validation predictions.
2. Build and measure video alarm verification using the trained model.
3. If GPU access is available, test one short fine-tune against the unchanged checkpoint.

Do not start another long training run or a broad model/hyperparameter search. Improvements are hypotheses to test, not promised gains.

## 2. What has actually been completed

| Model | Input | Completed training | Test mAP@50 | Test mAP@50:95 | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| YOLOX-Nano-TI-Lite | 416 × 416 | 20 epochs | 81.75% | 51.40% | 77.92% | 78.11% | 78.01% |
| YOLOX-S-TI-Lite | 640 × 640 | 5 + 15 epochs | 87.02% | 57.34% | 81.37% | 82.12% | 81.75% |
| YOLOv9-S-TI-Lite | 640 × 640 | 30 epochs; selected epoch 20 | 88.41% | 59.94% | 88.05% | 79.14% | 83.36% |

Precision/recall/F1 use confidence **0.40**, matching IoU **0.50**. AP uses COCO bounding-box evaluation. Nano/S were evaluated on 15,882 test images; YOLOv9 on 15,884. Training splits, recipes and input sizes also differ. This is a reported comparison, not a controlled architecture study. S-Lite's 5-epoch and 20-epoch checkpoints are the same model approach.

YOLOv9 has higher reported test AP, precision and F1, but **S-Lite has higher recall**. YOLOv9 full-validation results are substantially lower than its test results:

| YOLOv9 split | Images | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Full validation | 15,884 | 74.60% | 44.31% | 80.65% | 58.73% | 67.97% |
| Test | 15,884 | 88.41% | 59.94% | 88.05% | 79.14% | 83.36% |

The split-performance difference has not been explained. Duplicate images across dataset splits are reported, not removed; source/video independence is unproven. These limitations must remain visible in reports.

YOLOv9 training, final validation/test evaluation and ONNX export are complete. ONNX checker and PyTorch/ONNX Runtime numerical parity passed on three inputs. This establishes export correctness on those inputs, not industrial reliability or board compatibility.

No new training is running in this workspace. The lab run finished, the user reports lab cleanup completed, and the scheduled Colab monitor was deleted. Do not restart Colab or recreate monitoring as part of this assignment.

## 3. Where to start

- Repository: <https://github.com/Skandakm29/honeywell_bmsce_flame_detection>
- YOLOv9 branch: `feature/yolov9-flame-colab`; published commit `7fe39a2`.
- Draft PR: <https://github.com/Skandakm29/honeywell_bmsce_flame_detection/pull/4>. Check its current state before assuming it is merged into `main`.
- Read `docs/RESULTS_YOLOV9.md` and `docs/RESULTS_e20.md`.
- Training/evaluation code: `experiments/yolov9_flame/`.
- Small original reports/configuration/history: `results/yolov9_final/`.

**Large model/data artifacts are intentionally not in GitHub.** Obtain the verified model and data from the project owner before GPU/video inference work. The owner's laptop currently retains:

```text
E:\Projects\honeywell\honeywell_bmsce_flame_detection\results\lab_backup_20261031\
```

Within that backup, relevant files include:

```text
outputs/full/best.pth
outputs/full/last.pth
outputs/full/training_summary.json
outputs/validation/predictions.json
outputs/validation/metrics.json
outputs/validation/export/       # ONNX, export report and model contract
outputs/test/                    # Historical final results; do not tune on this
```

Use the saved manifests accompanying the run. Prediction `image_id` values in our evaluator are **zero-based record positions**, not necessarily the original COCO image IDs. Preserve their mapping and verify manifest hashes before joining predictions to labels. Manifest image paths may need remapping on a different PC; retain an unchanged source copy.

Selected checkpoint SHA256:

```text
48d8e654ac1e3381e122a52cdfca100d1840e8cf6db0294ac882b52ab1c4a51f
```

ONNX SHA256:

```text
332cb480c4a4f3d8cdfbb3f5545a2e08211f6ae3db492d1d08d22318e2c78544
```

Use an isolated environment. Keep original weights, datasets and reports unchanged. Save every new experiment in a separate directory with its config, source commit, seed and artifact hashes.

## 4. Exactly what is missing

| Gap | Current evidence | What this assignment should deliver |
|---|---|---|
| Operating threshold | 0.40 is the historical reporting point, not established as best for the application | Validation precision/recall trade-off and a documented candidate threshold |
| Video alarm behaviour | Still-image boxes only | Measured event detection, alarm delay and false alarm events per camera-hour |
| Temporal verification | No measured video alarm policy | Compare immediate alarm with a few simple persistence policies |
| Short final training refinement | Mosaic remained at 0.5 through the completed run | One small mosaic-off fine-tune, retaining the original if it does not help |
| Coverage and generalisation | Specific nuisance coverage and scene independence unconfirmed | Consume the other teammate's audit; keep video development/evaluation scenes separate |
| Embedded deployment | ONNX export passed; AM62A/TIDL/INT8 performance unmeasured | Record as pending; measure only if board/toolchain access becomes available |

The target remains **TI SK-AM62A-LP**, with the user's stated **maximum 1 TOPS constraint**. Accelerator marketing specifications do not establish workload compliance. Desktop FPS cannot be reported as board FPS.

## 5. Priority 1 — threshold selection, no training

Use saved **full-validation predictions** rather than rerunning inference for each threshold.

1. Verify the checkpoint/manifest identities and reproduce the saved 0.40 precision, recall and F1 within numeric tolerance.
2. Sweep confidence from 0.10 to 0.80 in 0.05 steps. Keep matching IoU 0.50 and the existing NMS/evaluation definitions fixed.
3. Save precision, recall, F1, TP, FP and FN at every threshold. Also report FP boxes per image; do not rename it false alarms per hour.
4. Identify the maximum-validation-F1 threshold as a **provisional** candidate. Also show a higher-recall alternative and its precision cost. A safety operating point requires an application-specific requirement; do not invent one.
5. Keep 0.40 results alongside the candidate for continuity. Do not use the historical test split to choose thresholds.

Changing the operating threshold does not improve the underlying AP when the prediction set is unchanged. It changes the precision/recall trade-off.

**Deliverables:** `threshold_sweep.csv`, a precision–recall plot and a short recommendation. This is CPU post-processing work, with no GPU training required. Measure actual runtime rather than promising completion time on an unknown PC.

## 6. Priority 2 — video alarm verification, no training

Build a small reproducible video evaluation harness around the existing exported model. Follow `model_contract.json` for preprocessing, output interpretation and coordinate mapping; do not assume an Ultralytics output format.

Start with a small set of existing usable flame and non-flame videos. Record source/permission, duration, scene ID and first clearly visible flame time. For flame clips, identify the relevant flame region so an unrelated detection elsewhere is not counted as success. This is a small video evaluation task, not another 60,000-image review. Do not stage hazardous experiments.

Separate **development clips** for policy selection from **held-out clips** for final reporting, grouped by source video/scene. Adjacent segments of one recording must stay together. If independent clips cannot be obtained, label the output a feasibility demo, not a generalisation result.

Implement and compare only these initial policies:

- Immediate alarm when a qualifying detection appears.
- Spatially consistent detections present for at least 60% of samples in a 2-second window.
- The same policy with a 4-second window.

These are proposed starting settings, not industry requirements. Use timestamps and a fixed documented inference sampling rate so behaviour does not depend on playback speed. Associate boxes across time using a simple documented overlap/proximity rule; detections jumping between unrelated locations must not accumulate into one flame event. Windows require full observed duration before satisfying persistence.

Define alarm clearing and repeated-event counting before evaluation. For the first comparison, clear after 2 seconds without a qualifying detection; count a new event only after clearing. Report that rule so the false-alarm denominator cannot be misunderstood.

Report for each policy:

| Metric | Definition |
|---|---|
| Event detection rate | Flame events correctly alarmed / total annotated flame events |
| Time to alarm | Alarm timestamp minus first clearly visible flame timestamp; median and p95 over detected events, alongside missed-event count |
| False alarm frequency | False alarm events / total non-flame camera-hours |
| Observation size | Number of scenes/events and total positive/negative video duration |
| Processing performance | End-to-end latency and sustained throughput on the named device |

Save per-clip results and timestamps, plus a few short examples of failures. Run development comparisons on cached frame detections so the model is evaluated once per video, rather than once per policy. Lock the threshold and temporal policy before evaluating held-out clips.

**Deliverables:** inference/alarm harness, clip manifest, per-event CSV and comparison report. A short demo with zero false alarms is not evidence of a production false-alarm rate. Report the actual observed hours and counts.

## 7. Priority 3 — one short fine-tune, GPU only

**Hypothesis:** a few epochs on natural, non-mosaic images may refine the selected checkpoint. The original recipe scheduled mosaic-off for the final 10 of 40 epochs, but training stopped at 30, so that phase did not run. This does not establish that its omission caused the current limitations.

Use this single proposed experiment:

| Setting | Proposal |
|---|---|
| Starting weights | Strictly load the selected epoch-20 trained weights |
| Duration | 3 additional full-training-set epochs maximum |
| Input | Keep 640 × 640 |
| Data | Same training manifest and flame-only label policy; retain negatives |
| Mosaic | Off from the start; retain other existing mild augmentations |
| Optimizer | Fresh AdamW, peak LR `1e-5`, existing weight decay; record all parameter groups |
| Schedule | Short warmup, e.g. 100 optimizer updates, then cosine decay over this run |
| Batch | Use verified GPU fit; previously A5000 used batch 16, accumulation 2 |
| Stability | Keep AMP, EMA, clipping and atomic checkpoint saves |
| Selection | Same monitor-validation subset each epoch, plus the unchanged starting checkpoint as a candidate |

**Engineering prerequisite:** the current launcher is built around COCO initialization and matching-run resume. Add an explicit trained-weight fine-tune path if needed, with strict loading. Verify weights match the selected checkpoint before the first update and that optimizer/scheduler/EMA initialization is intentional. Do not pass the trained checkpoint into a permissive pretrained loader and assume it loaded correctly. Do not alter the completed run's config or pretend this is exact resume.

The existing evaluator hardcodes its 0.40 report. Keep that historical metric and calculate any selected-threshold report separately with clear definitions.

On the earlier A5000, full epochs took approximately **31 minutes**. Three more epochs therefore suggest **about 1.5–2 hours of training**, plus setup, validation and export. GPU access and speed must be measured again; the lab environment has been cleaned. This is an estimate, not a promise.

Set a **3-hour training cap**, save at epoch boundaries, and allow roughly **4 hours total GPU experiment time** including final checks. If a measured first epoch predicts exceeding the cap, shorten the trial and document it. Do not silently switch to CPU training, extend the run or launch a hyperparameter grid.

Select using validation only. Keep the original if the candidate does not improve the chosen validation criterion. Report any recall/precision trade-off rather than declaring a win from one metric. Run full validation for the selected candidate; run final held-out evaluation only after selection. The existing Kaggle test is already a known benchmark, not a new untouched holdout.

If the other teammate later confirms important missing negative conditions, propose a separate targeted data addition and short fine-tune. Do not mix that change into this experiment: we need to know whether mosaic-off refinement itself helped. Agree on the intended alarm policy before changing labels: a glowing electric heater can be negative, while an actual burner flame is still a physical flame. Screens showing fire are intended nuisance cases in the mentor's checklist; distinguish displayed flames from physical flames and document that policy when adding such examples.

## 8. Industry references and comparison limits

| Reference | Relevant documented behaviour |
|---|---|
| [Bosch AVIOTEC 8000i IR manual](https://cdn.commerce.boschsecurity.com/public/documents/User_manual_Operation_Manual_enUS_107063621003.pdf) | Flame verification is configurable 4–20 seconds, default 10; shorter settings can increase false alarms. Verification setting is not guaranteed total detection time. See printed pp. 19 and 43–44. |
| [Honeywell FS24X Plus manual](https://prod-edam.honeywell.com/content/dam/honeywell-edam/sps/his/en-us/products/gas-and-flame-detection/documents/industrial-fixed/fs24x-plus/sps-his-1701m5001-en-revc-fs24x-plus-user-manual.pdf) | Defined flame/fuel/distance response tests and interference tests involving sunlight, welding, heaters and lighting. This is a multispectrum IR detector, not an RGB model. See printed pp. 43–44. |
| [ISO 7240-29:2024](https://www.iso.org/standard/83347.html) | Requirements, test methods and performance criteria for visible-spectrum video fire detectors. Public scope is not a substitute for the full standard. |
| [UL 2684](https://www.shopulstandards.com/ProductDetail.aspx?UniqueKey=46642) | Standard for video and thermal image detectors used in fire alarm systems. |

These support measuring alarm behaviour and controlled nuisance rejection. They do not establish a universal industrial mAP/F1 target, validate our proposed settings or certify this project.

## 9. Definition of done

Return one short report containing:

1. Validation threshold comparison and selected candidate, with the original 0.40 baseline.
2. Video-policy results with actual durations, event counts, alarm delays, misses and false alarms per camera-hour; clearly identify development versus held-out scenes.
3. If attempted, original versus short-fine-tune validation results, elapsed time and retained checkpoint decision.
4. Code/configuration, manifests and hashes sufficient to reproduce the result. Re-export and repeat ONNX checker/parity if weights change.
5. Remaining limits: dataset nuisance coverage pending the other teammate's audit; no industry-equivalence claim; board/TIDL/INT8/1-TOPS status explicitly stated.

**Start with Priority 1. The main immediate gap is measured alarm behaviour, not another long training run.**
