# Analyse three flame models and choose a practical improvement

**Evidence date: 1 November 2026.** Give this file to the improvement teammate's AI with the model screenshots, session folders and original videos. The goal is a measured improvement that remains a candidate for **TI SK-AM62A-LP under the project's stated maximum 1 TOPS constraint**. Model stacking is an option to investigate, not a predetermined solution.

**Latest dataset audit supersedes the pilot-only coverage status below:** read `FASDD_FULL_NEGATIVE_AUDIT_REVIEW.md`. The new PDF lists106 confirmed condition examples; all IDs/splits/no-flame targets/image hashes match our dataset. 54 are current YOLOv9 train negatives,18val/34test. Welding is now documented. Heaters/screens remain not confirmed, and torch-cutting entries require policy review. Four verified missing-flame labels are in val/test, not train. Do not move evaluation examples into training or treat coverage as proven nuisance rejection. The new full-run execution/scores still require raw logs/ranking/verification CSVs for reproduction.

**Latest refinement status:** the three-epoch unchanged-data run finished; none beat original monitor AP, so original epoch20 remains selected. Epoch2 increased monitor recall/F1 with more FP boxes; newly refined weights have no established full-validation result in that archive. Read `YOLOV9_REFINEMENT_RESULTS.md`. Do not blindly rerun the earlier proposed three-epoch experiment below as though it is still untried.

**Read the actual evidence review first:** `THREE_MODEL_WEBCAM_REVIEW_20261101.md`. The subsequently supplied `results-20261101T114806Z-1-001.zip` contains three actual webcam sessions, not CLIP outputs. It shows shared false detections on orange fabric/objects and mixed YOLOv9 sessions; it does not supply synchronised raw clips for paired-error analysis. The CLIP notebook below is separate supporting dataset evidence. Findings in the companion review supersede earlier uncertainty about whether Nano/S had been run by a teammate.

**Active pair changed to Nano + YOLOv9:** S is currently unavailable. Do not wait for it or claim it was worse everywhere: its reported test recall was higher. The user supplied `Nano-20261101T122734Z-1-001.zip`; its ONNX hash matches the published Nano baseline. Nano and our existing YOLOv9 passed local CPU ONNX inference/finite-output checks on landscape/portrait synthetic inputs with the viewer adapters. This verifies runnable formats, not flame quality or board deployment. Nano was copied into the local test pack's `Nano/` folder. Use this pair for same-raw-video replay; all three-model history below remains background evidence.

## 1. Instructions for the receiving AI

You are analysing our existing flame-detection system. First inspect the evidence; then explain its limitations and recommend the smallest useful experiment. Implement only a bounded experiment after establishing a valid baseline. Preserve all original models/reports and use separate output directories. No long retraining, broad hyperparameter search, new hardware purchases or replacement of the architecture by default.

Ask for missing evidence once, clearly. Continue screenshot-based qualitative analysis where possible, but do not invent quantitative comparisons from unmatched screenshots. Cite every claimed failure with a filename, frame/video time and model identity. State whether each conclusion is observed, operator-confirmed, inferred or still unknown.

### What to request

- For **each** model: trained `flame.onnx`, matching `model_contract.json`, `export.json`, and the training config/checkpoint identity if fine-tuning is proposed.
- Screenshots plus each run's `session.json` and `frames.csv`; original videos if available.
- Ground truth: actual flame or nuisance, visibility/occlusion, and relevant video time intervals. For box metrics, actual box annotations are needed.
- Whether screenshots come from **identical frames**, the same scene at different times, or entirely different trials.
- The other teammate's CLIP coverage confirmations with filenames and train/val/test split; rankings alone are not confirmations.
- Board/SDK/compiler access and versions. Record deployment as pending if unavailable.

Source repository: <https://github.com/Skandakm29/honeywell_bmsce_flame_detection>

Working branch: `feature/yolov9-flame-colab`

Viewer: `scripts/live_flame.py`; instructions: `docs/TEAM_WEBCAM_SETUP.md`. Check the latest branch before using an older copy.

## 2. Project and evidence status

We detect **visible physical flames in RGB images/video**. Dataset `fire` boxes are flame targets; smoke-only/no-flame images are negatives. Do not claim UV/IR sensing, fuel identification, actual flicker modelling or industrial certification. Screens showing fire are nuisance cases for the intended alarm application; an actual burner flame is different from a glowing electric heater.

| Model | Input | Training | Reported test mAP@50:95 | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| YOLOX-Nano-TI-Lite | 416 | 20 epochs | 51.40% | 77.92% | 78.11% | 78.01% |
| YOLOX-S-TI-Lite | 640 | 5+15 epochs | 57.34% | 81.37% | 82.12% | 81.75% |
| YOLOv9-S-TI-Lite | 640 | 30 epochs; selected epoch 20 | 59.94% | 88.05% | 79.14% | 83.36% |

P/R/F1: confidence 0.40, matching IoU 0.50. AP: COCO bounding-box evaluation. Nano/S test count 15,882; YOLOv9 15,884. Recipes/splits/input sizes differ. Aggregate scores are indicative, not proof that one architecture is uniformly better. Do not use the known test results to tune a new combination.

Full-validation F1 is 63.49% Nano, 68.44% S and 67.97% YOLOv9. YOLOv9 full-validation recall is 58.73%, versus 79.14% on test. Investigate this split difference rather than explaining it without evidence. Source/scene independence is unproven and dataset cross-split duplicates are reported, not removed.

Detailed reports: `docs/RESULTS_e20.md`, `docs/RESULTS_YOLOV9.md`, `results/yolov9_final/`.

### Actual webcam findings available so far: YOLOv9 only

- Trained ONNX confirmed by SHA256 `332cb480c4a4f3d8cdfbb3f5545a2e08211f6ae3db492d1d08d22318e2c78544`.
- 4,258 processed frames in 408.39 seconds; about **10.42 processed FPS on the owner's laptop CPU**. This is not AM62A speed or sensor-to-display latency.
- Saved screenshots demonstrate real candle detections at several positions, scores around 0.47–0.80.
- Operator confirms detections when a box partly hid a still-burning candle: successful partial-occlusion examples.
- Operator confirms phone flashlight false positives from some angles; a saved false detection has score 0.76.
- Three saved unlit-candle screenshots have no detections. This is not a broad specificity measurement.
- An apparent flashlight interval contains detections continuously for about 3.07 seconds; no full video was recorded to verify the complete interval. Persistence alone cannot be assumed to solve it.
- No confirmed webcam recall/precision or alarm-rate measurement exists. CSV detections and selected screenshots do not supply full ground truth.

These findings cannot be transferred to Nano/S. Her screenshots for those models must be analysed before attributing strengths or failures to them.

### What the attached CLIP notebook actually contains

`FASDD_CLIP_Hard_Negatives (1).ipynb` is **dataset retrieval**, not evaluation of any flame detector. It searched 500 images from the no-fire/no-smoke-annotation pool using frozen CLIP ViT-B/32 and 134 prompts. Allocation: 334 train, 83 validation, 83 test. All nine top-50 contact sheets were recovered without rerunning. The saved verification table has **450 unreviewed entries and zero recorded confirmations**. Informal visual inspection may exist outside the notebook; request it rather than dismissing it or claiming it is recorded.

Only confirmed **training-split** examples support saying a condition occurred among training negatives. Validation/test examples were not training inputs. Coverage does not establish reliable rejection. Welding is not confirmed, not established absent. Do not infer flashing from a still image.

The recovered materials are available in the owner's `E:\Projects\honeywell\FASDD_CLIP_Recovered\`. Use the cleaned notebook/recovered files: the original notebook exposes a Kaggle credential in saved output, already reported for revocation. Never share that output or reuse the credential. No CLIP rerun is required to review the saved sheets.

## 3. First diagnose the comparisons

### A. Rule out inference mistakes before blaming training

The three models do not share identical input/output formats:

| Property | Nano / S | YOLOv9 |
|---|---|---|
| Input | BGR float32, raw 0–255 | RGB float32, 0–1 |
| Resize | Aspect-preserving, floor dimensions | Aspect-preserving, rounded dimensions |
| Padding | Top-left content; bottom/right 114 | Same placement/value before normalisation |
| Raw output | Decoded centre x/y, width/height, objectness, class score | x1/y1/x2/y2, flame score |
| Confidence | Objectness × class score | Flame score directly |
| Coordinate restoration | Nominal resize ratio | Actual rounded x/y resize scales |

Use the supplied contracts and repository implementation. Confirm NMS 0.65, max 100 boxes and starting confidence 0.40. Verify hashes, no accidental original COCO/pilot model, finite outputs and plausible box positions. CPU smoke tests of YOLOX adapters used synthetic outputs; actual trained Nano/S exports still need owner-side checks.

Do not treat a score of 0.8 from one model as equivalent to 0.8 from another. Confidence scores are not calibrated cross-model probabilities.

### B. Make failures comparable

Screenshots can show specific true detections, misses or false detections. They cannot establish rates, speed, stable video behaviour or complementary errors across unrelated frames. A fully hidden flame is not a visible-flame miss. Ambiguous glare/occlusion needs the operator's explanation or original footage.

If frames do not match, replay a small shared video set through all three, caching outputs once. Use `--source <video.mp4>` and record model/source hashes. Align by original video frame index/source time, not wall-clock time or webcam frame counters from different sessions. Use the same laptop for speed comparisons.

For each condition, create:

| Clip / frame | Ground truth and visible region | Nano result | S result | YOLOv9 result | Failure explanation / uncertainty |
|---|---|---|---|---|---|

Conditions: actual small/large-in-image flame; partial occlusion; illumination/background change; flashlight; sunlight/windows/reflections; clothing/lights; smoke/heater; screens. Welding/grinding only if evidence exists, otherwise not tested. Distinguish glare, blur, clipping/overexposure and flame size from proven causes. Pixel appearance alone cannot establish different fuel types.

For a labelled subset, count per-model TPs/FPs/FNs using one-to-one box matching IoU 0.50. Count specifically:

- Ground-truth flames caught by S but missed by YOLOv9, and the reverse.
- Nuisance detections made by both versus only one model, matched at the same location/time.
- Failures shared by all three.

The question is **whether another model corrects the first model's mistakes on the same evidence**. Higher recall in a separate summary does not prove this. Multiple detections in different locations are not agreement.

Keep development clips separate from held-out clips, grouped by recording/scene. Adjacent frames are correlated; do not claim hundreds of independent tests from one short candle recording. For existing still-image AP, use a common manifest and original-image coordinates. Any selected hard-case screenshot set is diagnostic and must not be reported as representative deployment accuracy.

## 4. Evaluate combinations only if errors are complementary

Start with single-model baselines. Then, on cached matched predictions, test **one justified pair** before considering all three.

| Option | When it could help | Main risk |
|---|---|---|
| Agreement gate | One model rejects nuisance detections made by another | Can lose real flames detected by only one; requires both inferences |
| Union of detections + NMS | Models find different real flames | Adds false detections; requires both inferences |
| Conditional second detector | A low-cost first model needs selected candidate verification | First-stage misses remain missed; trigger frequency/CPU work can remove expected savings |
| One improved deployable model | Common failures or two-model deployment too expensive | Needs a focused training experiment; gain not guaranteed |

For a first agreement prototype, use each model's threshold and match boxes in original coordinates (e.g. IoU 0.50 as a declared starting rule). Keep the primary model's box; avoid inventing an average probability. Very small flames can have poor inter-model IoU even when both detect them: inspect these before interpreting a gate failure.

For union, deduplicate boxes and report the operating-point trade-off; cross-model scores are not calibrated rankings. Score fusion/calibration needs development data. If cropped-region verification is proposed, evaluate it as a new input distribution with correct coordinate restoration: a detector trained on whole images may lose context on crops. Do not add an unsupported crop-classifier assumption.

Before trying a second network, compare a cheap spatially consistent temporal policy using the same single-model predictions. Measure both alarm delay and missed events; the known persistent flashlight failure can survive temporal filtering.

**True stacking** means training a further model on other models' predictions. It needs independent/out-of-fold training predictions and a held-out evaluation. With the current small, selected screenshot evidence it is not a justified first experiment. A tiny combiner does not eliminate the cost of running its detector inputs.

**Distillation** means using stronger model(s) during training to improve one smaller deployed model. Consider it later only if the teacher is demonstrably better on relevant failures. Confident but wrong flashlight predictions make poor teaching targets. Do not promise short training or an accuracy gain without a measured pilot.

Reject the combination if it does not beat a well-selected single-model operating point on held-out conditions, or if its cost cannot meet deployment needs. Report regressions, including weaker small-flame recall, not only average improvement.

## 5. Deployment is a separate gate

Our exact trained flame ONNX files have **not** been validated on AM62A. A teammate's report demonstrated 32-bit diagnostic conversion of a **COCO YOLOv9** with selected CPU fallbacks and four TIDL subgraphs. It explicitly states those diagnostic artifacts are not board-executable. Do not reuse its node IDs blindly on our graph, include YOLO26n as one of our three trained flame models, or call this deployment confirmation.

TI documents that unsupported operations can fall back to the Arm CPU. Graph breaks/data transfers can materially affect execution. Counted accelerator nodes and file sizes do not establish runtime utilisation, memory or speed. Match PC toolchain and board runtime versions; use documentation for that exact SDK/release.

For the chosen single model or pair, the deployment teammate must:

1. Compile the **exact trained ONNX hashes** for AM62A using the compatible toolchain; retain logs, supported/fallback operators and partition information.
2. Use representative training/development images for quantization calibration, including flame sizes and nuisance lighting; do not calibrate/tune on held-out evaluation data.
3. Measure detection quality after quantization, using the same held-out protocol, against float32 quality. Choose 8/16-bit settings based on evidence and cost, not a guaranteed speed/accuracy assumption.
4. Run on SK-AM62A-LP and measure the full pipeline: capture/decode, preprocessing, all inference calls, NMS/fusion/verification and output. Record sustained throughput, median/p95 latency, CPU use, memory and frame dropping. Camera buffering and any temporal waiting contribute to alarm delay.
5. For conditional inference, report typical **and worst-case** invocation frequency and latency; many candidate regions can make it expensive. Specify whether models share frames, run serially or concurrently; concurrency is not free acceleration.

Retain the project's **maximum 1 TOPS** workload limit even if the board advertises a different peak rating. Clarify whether it means workload operations/second or another agreed budget, plus the desired inference rate. Do not make a compliance claim before this definition is established.

For arithmetic screening only: if a model costs M GMAC/frame and runs at f inferences/second, and the agreed convention counts 1 MAC as 2 operations, workload is approximately `2 × M × f / 1000 TOPS`. Add all models/invocations. This is not hardware utilisation, latency or a proof of achieved throughput. Existing report estimates Nano at 1.452 GMAC and S at 13.291 GMAC at their stated resolutions; reprofile with the same convention. YOLOv9 comparable compute is not established here. Parameter counts/MB cannot substitute for operations.

Without physical/remote board access, deliver an **unverified deployment candidate** with operator/cost analysis and list required board measurements. Laptop success or ONNX parity alone cannot guarantee deployability.

## 6. Bounded improvement plan — hours, not another overnight search

1. **No training first:** validate adapters, align footage, produce the failure/complementarity table and compare threshold/temporal policies. Reuse cached predictions.
2. **If justified:** prototype one paired rule offline and measure whether it improves the trade-off. Retain a single-model baseline at both fixed 0.40 and its development-selected operating point; select nothing on test clips.
3. **If failures are shared, or a pair is impractical:** choose one model for a short targeted fine-tune. Prioritise confirmed training examples of observed failures such as flashlight glare while retaining diverse flame positives and other negatives. A case existing somewhere in FASDD does not prove sufficient training coverage. Never add evaluation clips/val/test retrieval hits to training without redefining and independently replacing the evaluation split.

YOLOv9 optional experiment: 3 epochs maximum, 640 input, start from the selected epoch-20 trained weights, low LR around `1e-5` as a proposal, mosaic off, retain AMP/EMA and mild augmentations. This tests one hypothesis. The original training stopped before its planned mosaic-off phase; that omission is not a proven cause of its errors. Do not simultaneously mix in new negatives and then attribute any gain solely to mosaic-off refinement: choose one intervention first.

The existing training launcher has a COCO pretrained loader and strict same-run recovery. A fine-tune needs explicit strict trained-weight loading, deliberate fresh optimizer/schedule, short warmup, changed minimum-epoch settings and a new run directory. Do not change the frozen recipe in place or use defaults that enforce 30 epochs. Do not repeat the original LR pilots.

Previous A5000 full epochs took about 31 minutes, suggesting roughly 1.5–2 hours training for three epochs plus setup/evaluation. That lab environment was cleaned; current GPU availability/speed is unknown. Measure the first epoch, cap training at 3 hours and the whole GPU trial around 4 hours. Save safely at boundaries and keep the original if validation does not improve the selected criterion. CPU inference/analysis is fine; avoid full CPU training.

After selecting any new weights, export/check ONNX, repeat numerical parity and hand the exact new hashes to the board teammate. Detector ensembling, distillation, calibration and model surgery do not automatically preserve prior export or board checks.

## 7. Required output from her AI

Return these before recommending deployment:

1. **Evidence audit:** which files/models/scenes were actually analysed; what is missing.
2. **Condition comparison and paired errors:** screenshot/frame references, confirmed strengths, shared failures and unique corrections. If only unmatched screenshots exist, stop at qualitative findings and request shared replay.
3. **Decision:** best single candidate, whether one pair adds a useful benefit, and its measured/estimated cost. It is valid to conclude stacking is unnecessary or unsupported.
4. **Metrics with denominators:** original box metrics when labels exist; event detection and time-to-alarm only with labelled intervals and a defined alarm/clear policy; false alarm events per camera-hour only with actual non-flame duration. Always report missed events alongside delay statistics. Screenshot counts are not event metrics.
5. **One bounded next experiment:** changed setting/data, expected mechanism, time budget, validation criterion, baseline and regressions. Distinguish proposal from completed work.
6. **Hardware status:** ONNX/quantization/compiler/board results individually marked measured, estimated or pending. No invented compliance or accuracy target.

## Primary deployment references

- [TI AM62Ax model workflow](https://texasinstruments.github.io/processor-sdk-doc/processor-sdk-edgeai-AM62AX/esd/docs/master/edgeai/inference_models.html): model preparation, compilation, evaluation and deployment; model preprocessing and calibration requirements.
- [TI TIDL operators](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/operators.md): support depends on operator attributes, shapes and toolchain release; unsupported operations can execute on CPU.
- [TI TIDL runner FAQ](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/tidlrunner/docs/faq.md): subgraph fragmentation/data movement overhead and the importance of actual runtime investigation.
- [TI quantization guidance](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/quantization.md): quantization options, calibration and layer restrictions; evaluate the accuracy/performance trade-off.

These are current reference pages, not evidence that a particular SDK version or our particular graph supports every proposed configuration. Pin the release actually used by the board team.
