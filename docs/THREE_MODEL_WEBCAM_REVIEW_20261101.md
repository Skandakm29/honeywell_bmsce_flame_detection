# Three-model webcam evidence: review and improvement decision

Reviewed 1 November 2026. Source: user-provided `results-20261101T114806Z-1-001.zip` (14,898,962 bytes, 125 entries). Archive SHA256: `11144185659194c10201c9f2be9f55bae78c4197e96ca3334bbd2c34171bba49`.

This review used the three CSV logs/session JSON files, all three supplied model collages, two labelled S-Lite evidence sheets, all 72 individual YOLOv9 screenshots through generated contact sheets, and selected original screenshots for detail. Nano/S collages cover their 25/16 individual screenshot sets. No model inference or retraining was run for this review. Screenshots include personal images; keep the archive/images in team storage, not public GitHub.

## 1. What the archive contains

| Session | Logged architecture | CSV rows | Logged duration | Individual screenshots | Median processing ms | p95 processing ms | Observed processed FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | Nano | 1,619 | 53.77 s | 25 | 13.86 | 18.59 | 30.09 |
| yolos | S | 347 | 49.83 s | 16 | 145.63 | 153.05 | 6.94 |
| yolov9 | YOLOv9 | 1,013 | 162.65 s | 72 | 159.70 | 172.70 | 6.22 |

All logged sessions use confidence 0.40, ONNX Runtime CPUExecutionProvider, version 1.31.1 and the same reported Linux x86-64 platform. Physical CPU details, clocks/loads and complete viewer source revision are not provided. Numbers are host/session observations, not controlled hardware benchmarks or AM62A measurements.

Processing time excludes capture/display. Nano's processing-only screenshot overlay often reads 50–75 FPS while the actual logged loop processes around 30 frames/s, consistent with a camera-limited feed; camera configuration was not recorded. Do not quote the overlay as end-to-end camera throughput. Live sessions process different sampled frames at different rates.

The sessions contain respectively 779, 141 and 175 detection-positive logged frames. These are **not** true-positive counts, accuracy percentages or comparable success rates: scenes, time and frame sampling differ.

## 2. Model identity

- Nano logged SHA256 `749db8cecc21b43a8b31262d3ff5f0e2a0036f83e3b4aa73ff656e562af67458` matches the existing published baseline export report.
- YOLOv9 logged SHA256 `332cb480c4a4f3d8cdfbb3f5545a2e08211f6ae3db492d1d08d22318e2c78544` matches the verified final trained flame ONNX.
- S logged SHA256 `8397ef9117c4b0de61cc089dbf359a2b73bd2ef355083955aea60ed141ec259c` is identifiable, but its matching export/checkpoint report is not in this archive. Confirm selected 5+15-epoch provenance with its owner.

These hashes are session metadata; actual ONNX files are not in the archive and were not rehashed here. Logs show actual Nano/S execution on a teammate's machine, beyond our earlier synthetic adapter checks, but do not replace adapter parity/model-provenance checks.

## 3. Important evidence-quality problem

These are sequential live-camera trials, **not the same recorded frames replayed through all models**. No raw video or complete ground-truth timeline is included. The `video_seconds` fields are empty. Cross-model error overlap or exact improvement from ensembling cannot be calculated from these runs.

YOLOv9 mixes at least two bodies of evidence:

- `collage_yolov9.jpg` contains 23 selected screenshots captioned roughly 16:03–16:04.
- Current `yolov9/frames.csv` spans 16:11:03–16:13:46 and the folder has 72 screenshots.
- Example: `capture_411.jpg` visibly shows a detection and the old collage captions it at 16:03:33; the current CSV row with sequence 411 is at 16:11:17 and has **no detection**.
- `capture_505.jpg` shows a 0.77 detection, but sequence 505 is not even present in the current CSV.

Likely explanation: session outputs/screenshots were reused or merged, but the archive does not prove exactly how. Do not join old screenshots to the current log by filename alone. Keep the older collage evidence qualitative. Use fresh unique output directories for every future run. The current repository viewer rejects an existing `frames.csv`; check which revision was used by the teammate.

Separate S-Lite montages `real_detections.jpg` and `false_positives.jpg` have captions around 14:03–14:18, outside `yolos/frames.csv` at 16:02–16:03. They are additional screenshot evidence, not labelled records of that CSV. Titles and green/red collage borders describe whether a box was drawn, not whether the prediction was correct.

## 4. Observed behaviour

| Condition | Nano evidence | S evidence | YOLOv9 evidence | Interpretation |
|---|---|---|---|---|
| Orange container/cap-like objects | False boxes in `baseline/capture_326.jpg` (~0.66), `483` (~0.51), `547` (~0.74), `660` (~0.77); other views rejected | False boxes in `yolos/capture_254.jpg` (~0.55), `1034` (~0.85); some views rejected | Older screenshots `capture_411/505` detect a small orange object (~0.55/0.77), while other container orientations are rejected | All three show some nuisance confusion. Angle, size and glare matter; rejection in one frame is not robust rejection of the object |
| Orange fabric/sock-like item | Large false box: `baseline/capture_1017.jpg` (~0.74), `1072` (~0.82) | False box: `yolos/capture_1372.jpg` (~0.78), `1424` (~0.80), with an earlier rejection `1329` | False boxes in `yolov9/capture_614.jpg` (~0.83), `1492` (~0.79), `1691` (~0.59), with some rejected views | Shared failure class across all three. No evidence that majority voting eliminates it |
| Visible lit match/flame | `baseline/capture_1568.jpg` (~0.74), `1596` (~0.81) visually box the bright flame | Labelled `real_detections.jpg` shows three lit-match examples, scores ~0.76–0.87, from a different session | Older `yolov9/capture_2573.jpg` (~0.86), `2600` (~0.88) visually box a flame | Flame detection demonstrated qualitatively; no comparable miss-rate/size/occlusion study from these examples |
| Phone light and metal cup | No comparable trial established in provided baseline set | Separate `false_positives.jpg` includes a reflective bright object labelled light-reflection FP (~0.66); original scene provenance limited | Later `capture_2399/2430/2494` show flashlight/cup without boxes; `2533` shows a false box (~0.41); some later angles rejected | YOLOv9 rejects some angles but is not immune; earlier owner's separate flashlight session also had FP at 0.76 |

Visual interpretation supports these object/flame descriptions, but ask the operator if any scene was replayed or another source was present. The separate S montage's captions are supplied labels, not an independently verified video timeline. Red/orange object detections are clear enough to prioritise further evaluation; avoid pretending their exact cause is proven to be colour alone. Texture, shape, glare and training context can also contribute.

No general conclusion about welding, grinding, smoke, heater rejection, fuel diversity or long-range performance is supported by this archive. No reliable flame false-negative rate or alarm events/hour can be computed.

## 5. Can stacking help?

**Not established by this evidence.** It demonstrates a shared nuisance failure, not proven complementary strengths on identical frames.

- **Agreement/majority:** all models already respond to orange fabric in their own trials. Agreement can preserve a shared error; separate trials do not prove simultaneous agreement either. Measure it on identical footage.
- **Union:** can recover unique true detections but also combines nuisance detections. There is no measured unique-recovery benefit here.
- **Higher threshold:** orange fabric scores reach about 0.80 or higher. Raising confidence slightly cannot be assumed to solve it; it may also remove real weak/small flames.
- **Persistence:** fabric can stay still or move continuously while looking flame-like to each network. A time delay is not a material/flame classifier. Measure both false triggers and actual-flame delay.
- **Conditional verifier:** could help only if it rejects the primary's false candidates while accepting its true candidates. None of the three has demonstrated that property on matched frames yet. The primary's missed flames cannot be recovered by verifying only its detections.

There is a practical low-cost experiment: replay short **raw** clips of the orange fabric, container, flashlight and actual flame through each model, then compute same-frame paired errors. Do not feed the already annotated screenshots back into detectors as an unbiased comparison: drawn boxes/text alter the input. If raw recordings are unavailable, record new clips once. Use these failure examples to define development tests, plus new scenes/objects held out for evaluating the chosen change.

## 6. Compute implications

On this reported host, Nano is much cheaper per processed frame than S or YOLOv9. Adding another full heavy model therefore has a real potential cost. An arithmetic sum of the logged median processing times is:

| Serial pair | Sum of individual median processing times | Meaning |
|---|---:|---|
| Nano + YOLOv9 | 173.56 ms | Small incremental host arithmetic cost compared with YOLOv9 alone |
| S + YOLOv9 | 305.33 ms | Almost twice the heavy-model host processing time |

These sums are **rough screening arithmetic, not measured ensemble latency**. They omit combination/transfer costs and use separate runs; medians need not add to the combined-run median. Do not extrapolate them to AM62A. Different precision, unsupported operators, memory and transfers can change the ordering.

Retain the 1 TOPS project limit. First establish each exact trained graph's comparable compute and required inference rate. Then compile and measure the full chosen pipeline on AM62A. Parser node percentages and ONNX sizes do not prove workload compliance.

TI cautions that unsupported operations/graph partitions can add data movement overhead. Any paired model needs evaluation as its own runtime pipeline. See [TI runner FAQ](https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/tidlrunner/docs/faq.md), [AM62Ax model workflow](https://texasinstruments.github.io/processor-sdk-doc/processor-sdk-edgeai-AM62AX/esd/docs/master/edgeai/inference_models.html) and [quantization guidance](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/quantization.md). Use the board team's actual SDK release.

## 7. Recommended order of work

1. **Repair the evidence, no training:** separate sessions and confirm S model provenance. Retain raw clips/source hashes and avoid output-folder reuse. Existing snapshots remain useful diagnostic evidence.
2. **One shared replay, no training:** record/reuse raw clips of orange objects/fabric, phone light/reflections and actual flames. Run all three on the same frames. Include similar-coloured flame/non-flame examples; do not solve it with a rule banning orange pixels.
3. **Select by paired errors:** assess one single model and one justified pair, including actual-flame misses. If all share the failure, skip stacking for that failure and favour a single-model improvement. Add temporal verification as a separately measured system layer.
4. **One short training experiment if justified:** fine-tune a selected model on confirmed training nuisance examples while retaining diverse flame positives/negatives. The CLIP audit can identify existing training cases. Keep these specific webcam failures as development evidence and evaluate on new independent objects/scenes. Three YOLOv9 epochs were previously estimated around 1.5–2 training hours on an A5000; profile current hardware, cap training at 3 hours and do not restart a 30-epoch run. Improvements are not guaranteed.
5. **Deployment gate:** exact ONNX export/parity, target-compatible compilation, quantized-quality check, then actual board latency/memory/throughput and the agreed compute-budget check. A diagnostic COCO-model compile is insufficient.

For the receiving AI's complete constraints, experiment definitions and implementation cautions, read `MODEL_IMPROVEMENT_AI_HANDOFF.md` alongside this review. The current evidence supports targeted nuisance-rejection work; it does not justify immediately deploying an ensemble.
