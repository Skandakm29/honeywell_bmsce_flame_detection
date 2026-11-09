# Flame Detection: Project Overview and Week 2 Summary

**Last updated:** 9 Nov 2026

**Labels used:** **Measured** = computed from saved results. **Estimate** = reasoned approximation. **Hypothesis** = likely explanation, not yet proven.

---

## 1. The project in one paragraph

We are building a **camera-based flame detector** that runs on an embedded board (TI **SK-AM62A-LP**) and raises a fire **warning and alarm** from live video. The detector is **YOLOv9-S-TI-Lite** (TI's edge-optimised YOLOv9, 1 class: flame, 640×640 input). Week 2 goals:
- false-alarm filtering
- confidence scoring
- debouncing and temporal logic
- a deployable model for the board

The final model choice weighs flame recall, false alarms, small-flame detection and board performance, **not mAP alone**.

---

## 2. Final week-2 system

```
Camera frame
  -> resize to 640 (letterbox, pad 114, RGB 0..1)
  -> YOLOv9-S-TI-Lite (V10 weights)
  -> decode + NMS (IoU 0.65)
  -> Small-box threshold: box counts if score >= 0.40, or (longest side < 16 px at 640 AND score >= 0.25)
  -> WARNING  = any counted box in this frame
  -> Temporal alarm layer -> ALARM
```

**Temporal alarm layer (locked 7 Nov, `investigation_week2/week2_final/rule.json`):**
- **Tracking:** each detection is linked to an existing track if the boxes overlap (IoU >= 0.1) or the centres are within 1.5× the box size. A track is dropped after 1 s unseen.
- **Persistence:** the ALARM turns ON when a tracked spot shows a flame in **>= 30% of frames over the last 2 s**.
- **Debounce:** the ALARM turns OFF after **3 s** with no flame.
- **Time-based** (seconds, not frames), so it works at 5–10 fps.
- **Not used:** flicker analysis, because flame flicker (about 1–10 Hz) needs about 20 fps or more, and the board runs at about 5–10 fps.

**Where the rule's numbers come from:**

| Item | Source |
|---|---|
| 30 s maximum time to alarm | **EN 54-10** flame-detector standard |
| 2 s window, 30% hit ratio, no score floor, no extra wait | **Tuned:** 144 combinations on our training clips + half of ONFIRE (fixed hash split), at 10 and 5 fps. Worst case over neighbouring settings; safety first, then fewest false alarms; simplest rule on a tie. |
| 3 s off-delay, tracking match, 1 s track drop | **Engineering defaults** (not from a paper) |
| 0.25 small-box threshold | Chosen on validation; kept because it added no false alarms on the tuning video |

---

## 3. Data

**Base dataset:** FASDD-CV v2 (Kaggle `yuulind/fasdd-cv-coco`, **version 2**).

| Split | Images | Fire boxes | No-fire images |
|---|---|---|---|
| train | 63,544 | – | – |
| val | 15,884 (all of `images/val`) | 12,526 | – |
| test | 15,884 | 9,022 | 10,435 (incl. 3,902 smoke-only) |

Only the **fire** class is scored; smoke is ignored.

### Added in week 2

**Hard negatives (not fire):**
- 18,497 Open Images photos: lamps, bulbs, sunsets, headlights, traffic lights, orange and hi-vis clothing, screens. Fire-tagged images were excluded.
- 3,883 heater and glowing-object images (Roboflow)
- 1,399 synthetic "fire shown on a screen" images
- D-Fire no-fire and smoke-only images

**Label cleaning:**
- 80 FASDD training images with unboxed flames were excluded.
- In V11, 326 doubtful images were dropped (CLIP + AI review, not human).

**Own footage (phone):**

| Shot on | Clips | Content | Use |
|---|---|---|---|
| 4 Nov | 30 (14 flame, 16 no-flame) | candles, diyas, far flames, occlusion; screens playing fire, lamps, LED lanterns, orange objects, torch, car lights | training + held-out test (split by location) |
| 5 Nov | 6 (3 flame, 3 no-flame) | walk-away candle, lighter in the dark, lantern; fire on a laptop screen | test, later V12 training |
| 6 Nov | 14 | candle, diya, lighter in 5 places at 1–10 m | 11 training (V12), 3 corridor clips held out for testing |
| 8 Nov | 25 + 1 photo | bonfires, candles, diyas, lighters, burning paper; unlit candles, torch, Diwali lights | testing only (`testing/videos_2026-11-08`, `NAMES.csv`) |
| 9 Nov | 5 | candle at 1/3/4 m in daylight; candle next to LED lantern, by a door, by a chair (dim); stairwell candle | testing only (`testing/videos_2026-11-09`) |

**ONFIRE** (public CCTV-style video) was used only for testing, and half of it for tuning the alarm rule.

---

## 4. Model history: what happened and why we moved on

| Model | What changed | Result | Why we moved on |
|---|---|---|---|
| **V9** (start) | Trained on FASDD only | Many false alarms: screen showing fire flagged in 68% of frames; small-flame recall 49% | Fix false alarms |
| **V10** | V9 fine-tuned on hard negatives + own frames (10 epochs, best = epoch 6) | Screen false alarms 68% → 4%; nuisance images flagged 7.8% → 2.3%; precision 88.1 → 90.3% | Small and far flames still missed |
| **V11** | Retrained from COCO, 40 epochs, zoom-out + small-flame paste + blur augmentation + label fixes | **Rejected:** false-alarm score 4.3× V10; small-flame recall flat | Lesson: build on V10, don't retrain from scratch |
| **V10 + box** | No retraining; small boxes count from 0.25 + temporal alarm layer | Small-flame recall 52.6 → 59.5% for only 38 extra false boxes (of about 800) | **Chosen week-2 system** |
| **V12** | V10 fine-tuned on own candle and lighter videos (3 augmentation variants × 8 epochs, then 4 epochs on the 6 Nov recordings) | On FASDD: more false boxes than V10 + box. On our own small indoor flames: clearly better | Candidate for next week |

---

## 5. Results on the FASDD test set (Measured, run once, never used for tuning)

| Metric | V9 | **V10 + box** | V12 |
|---|---|---|---|
| AP50 | 88.4% | 89.1% | 89.1% |
| AP50:95 | 59.9% | 60.2% | 60.1% |
| AP (small) | 33.6% | 33.8% | 33.7% |
| Precision | 88.1% | **90.0%** | 89.2% |
| Recall | 79.1% | 79.7% | **80.0%** |
| F1 | 0.834 | **0.845** | 0.844 |
| TP / FP / FN | 7,140 / 969 / 1,882 | 7,188 / **802** / 1,834 | **7,217** / 871 / **1,805** |
| Small-flame recall (365 flames) | 49.0% | **59.5%** | 52.9% |
| FAR (no-fire images flagged, of 10,435) | 2.74% (286) | **1.28% (134)** | 1.40% (146) |
| False boxes per 100 images | 6.1 | **5.0** | 5.5 |

**How these are measured:**
- Precision, recall, F1, TP, FP, FN and FAR use confidence 0.40 (V10 + box: 0.25 for boxes under 16 px) and IoU >= 0.5.
- AP doesn't depend on the threshold, so V10 and V10 + box have the same AP.

**Video alarm results (ONFIRE held-out half, 10 fps, Measured):**
- WARNING only: caught 6 of 6 fires, 3.3 false alarms per hour.
- ALARM (temporal layer): caught 4 of 6 fires, 2.0 false alarms per hour (about −40%).
- **That's why we keep two levels:** WARNING for sensitivity, ALARM for reliability.

---

## 6. Field tests on our own videos (Measured; same temporal layer for both models)

| Clip | V10 + box (ALARM on) | V12 (ALARM on) |
|---|---|---|
| Candle next to street lamp, hand passes (night) | 98% | 98% |
| Corridor candle, moving away | 98% | 98% |
| Lighter, far to close | 72% (alarm at 5.8 s) | 75% (alarm at 5.1 s) |
| Candle next to bright LED lantern (dim) | 69% | **98%** |
| Candle by door, moving away (dim) | 75% | **99%** |
| Candle in front of dark chair (dim) | 34% | **99%** |
| Candle at 1/3/4 m, **bright daylight** | **0%** | **0%** |
| Unlit candle + street lamp; orange Diwali lights; LED lantern (no flame) | no false alarm | no false alarm |

**Side-by-side videos:** `testing/PPT_grids_V10box_vs_V12/`

### Benchmark vs real world
- **V10 + box** is stronger on FASDD, which is mostly large, clearly visible fires.
- **V12** is stronger on our small indoor flames, because it was fine-tuned on similar footage.
- **Takeaway:** a model's ranking depends on the test data. FASDD alone doesn't reflect our target scenario.

---

## 7. Limitations and root causes

1. **Range is about 1 m for a candle** (1080p phone).
   - Frames are shrunk to 640 px, so a candle flame is about 20–25 px at 1 m and about 5 px at 4 m.
   - Below about 20 px it loses its flame shape. On FASDD, flames under 16 px are found about 15% of the time, vs about 75% above 64 px.
   - This is a resolution limit. Bigger fires are detected proportionally further away.
2. **Confirmation trade-off.**
   - The ALARM cuts false alarms by about 40%, but can miss very brief flames (2 of 6 ONFIRE fires). The WARNING still catches them.
3. **Small flames in bright daylight are missed**, even at 1 m (Measured on 9 Nov).
   - The flame is the same size as a detected dim-room flame (24 vs 23 px at input), but contrast is about 40% lower (0.46 vs 0.74) and the core looks pale yellow, not white-hot.
   - Our training videos are mostly dim rooms.
   - A lower threshold doesn't help: V10's best score on it was 0.01; V12 reached 0.51 in a few frames.

---

## 8. Board deployment (handed to the electronics side)

- **Raw-head ONNX of V10** (no decode in the graph), 640 and 416 (`investigation_week2/edge_export/`).
  - Outputs: `cls_s8/16/32` [1,1,H,W] and `reg_s8/16/32` [1,64,H,W]; reg_max 16; strides 8/16/32.
  - Ops: Conv, ReLU, MaxPool, Resize, Concat, Add, Slice.
  - Decode checked against the full model: within 0.006 px.
- **90 calibration images** for int8, all from the training split.
- **Final rule spec:** `investigation_week2/V10_final_rule_for_board.zip` (small-box threshold + temporal alarm).
- **Validation data pointers:**
  - FASDD v2 val split (15,884 images); file list in `edge_export/fasdd_val_15884_images.txt`
  - nuisance set (2,078 images) and our no-fire videos (shared via filebin)
- **Not yet measured:** int8 accuracy and frames per second on the AM62A; 416 accuracy.

---

## 9. Issues found and fixed this week

- **V11 autopsy bug:** a variable name clash in `autopsy11.py`, fixed by the lab agent.
- **Duplicate ONFIRE entries:** 12 ONFIRE videos were listed twice (as both fire and no-fire) in `videos.json`.
  - Fixed in the alarm tuning (run 3) by dropping them.
  - Earlier V10/V11/V12 ONFIRE video numbers counted them twice.
- **Automatic flame boxing for own videos was unreliable** (glow, lamps, reflections), so flame positions were marked by eye (AI) and reviewed per box.

---

## 10. Where things are (laptop, `E:\Projects\honeywell`)

| What | Path |
|---|---|
| V10 weights (chosen) | `investigation_week2/edge_export/src/FROZEN_V10/selected.pth` (SHA d046dab6); copy in `lab_backup_2026-11-07` |
| V10 ONNX (full / board) | `Flame_Detection_Test_Pack/YOLOv9_V10/flame.onnx` / `investigation_week2/edge_export/` |
| V12 weights + ONNX | `investigation_week2/lab_backup_2026-11-07/work/select_v12_rec/` |
| V9 weights / ONNX | `honeywell_bmsce_flame_detection/results/lab_backup_20261031/outputs/full/best.pth` / `Flame_Detection_Test_Pack/YOLOv9/flame.onnx` |
| Locked alarm rule + final reports | `investigation_week2/week2_final/` |
| Alarm rule code and protocol | `investigation_week2/temporal_final/` |
| Full lab backup (reports, logs, lists, code as run) | `investigation_week2/lab_backup_2026-11-07/` |
| Test videos | `testing/videos_2026-11-08`, `testing/videos_2026-11-09` |
| Comparison script | `testing/compare_v10box_v12.py` |
| Daylight analysis | `testing/compare_2026-11-09/` (`flame_contrast.json`, `lowconf_daylight.json`, `why_daylight_fails.jpg`) |
| Project notes for future sessions | `investigation_week2/CLAUDE.md` |

---

## 11. Next steps (proposed, not started)

1. **Small-flame dataset:**
   - candles, diyas and lighters at taped 1/2/3/5 m, phone on a stand
   - both daylit and dim rooms, plus matching no-flame clips
   - plus open-source small-flame data where licences allow
   - split by location into training and test, so the test matches our use case
2. **Fine-tune from V12** (the better small-flame start) with a stricter false-alarm limit. Re-test with the same videos and the same alarm rule.
3. **Two-stage detection prototype for range:** find bright, warm candidate spots on the full-resolution frame, then run YOLO on crops. Expected about 3× range for small flames (**Estimate**). Cost: more model runs per frame.
4. **Board numbers:** int8 accuracy, frames per second at 640 vs 416. These decide whether crops or a higher input size are affordable.

---

## 12. Weeks 3–4 plan (mentor meeting, 9 Nov)

**Original milestones:**
- W3 Spatial calibration: range rings, ground-plane geometry, system calibration
- W4 Obstruction detection: blind-zone mapping, occlusion monitoring

**Decision:** W3 and W4 are **combined** and reviewed together at the end of W4. Work starts with **obstruction detection**. A midweek check-in is possible on Thursday or Friday. Exams start 20 Nov; the schedule will be discussed next week.

**Requirement:**
- When the camera's view is partly **blocked by an added object** (a steel sheet, a box…), the system must **detect the obstruction** and **report how much of the area is blind**, i.e. the blind-zone extent.
- This is "indirectly an object-detection task, but the output must be the amount of blind zone".

**Design freedom:**
- **Option A:** add a new class (e.g. `obstruction`) to the existing flame model.
- **Option B:** build a separate model and integrate it.
- Either way, both must eventually run on **one processor** (AM62A).

**Open questions to settle first:**
1. **Blind zone as what:** % of image area, % of floor (ground plane, links to W3), or a map?
2. **Reference:** is there a known "clear" view to compare against (camera installed, then something appears), or must it work without one?
3. **Obstruction size:** how big or close must it be to count?
4. **Compute budget:** frame rate and latency on the AM62A, since multiple models or extra runs per frame cost speed.

**Starting idea (to discuss, not decided):**
- **Detect the obstruction:** a segmentation or detection model, or change detection against a reference frame.
- **Blind-zone %:** occluded pixels as a share of the frame. Later, project onto the ground plane (W3) to get floor area.
- Keep it modular, so it can run beside the flame model or be merged in as a class later.

## 13. Glossary

- **AP50 / AP50:95:** average precision at IoU 0.5 / averaged over IoU 0.5–0.95 (COCO).
- **Precision / recall:** share of predicted boxes that are correct / share of real flames found.
- **FAR:** share of no-fire images where the model drew any flame box.
- **TP / FP / FN:** correct boxes / false boxes / missed flames.
- **WARNING / ALARM:** any detection now / detection confirmed by the temporal layer.
- **Small-box threshold:** boxes under 16 px at the 640 input count from score 0.25 instead of 0.40.
