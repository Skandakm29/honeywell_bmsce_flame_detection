# Week 3 brief: Spatial Calibration (for the next agent)

**Written:** 11 Nov 2026. W3 is **independent** work; W4 (obstruction detection) is done separately by teammates.

**Everything needed is on GitHub, on the lab PC, or in the starter zip `W3_starter_V12.zip`.** No other machine is assumed.

## 1. The milestone

| Week | Milestone | Deliverable |
|---|---|---|
| **W3** | **Spatial Calibration:** range rings, ground-plane geometry and system calibration | Codebase expansion and integration with the existing system setup |

**Mentor, 9 Nov:**
- W3 and W4 are reviewed together at the end of W4.
- A midweek check-in is possible on Thursday or Friday.
- Exams start on **20 Nov**.
- Higher-resolution processing is allowed if it stays AM62A-compatible and latency is acceptable.

## 2. What W3 means (our interpretation; confirm with the mentor)
1. **System calibration:**
   - camera **intrinsics** (focal length, principal point, lens distortion), from a checkerboard
   - **extrinsics** (mounting height, tilt, roll), from a few known floor points
2. **Ground-plane geometry:** map any floor pixel to a real (x, y) position in metres, using a homography or intrinsics + height + tilt.
3. **Range rings:** draw distance circles on the floor in the image (e.g. 1/2/5/10 m), live on video and optionally as a top-down map.
4. **Integration with the flame detector:**
   - each detection and alarm gets a **distance in metres**, using the bottom centre of the box, which touches the floor
   - the rings can show **where a flame of a given size is detectable**

## 3. Facts from weeks 1–2 that matter (Measured)
- **Detection needs pixels:** a flame must be about 20–30 px tall at the 640-px model input.
  - A candle (about 3 cm) is reliable up to about **1 m** with a 1080p phone.
  - Detectable range grows roughly linearly with flame size.
- Frames are **letterboxed to 640** (aspect kept, image at the top-left, pad 114). Boxes are returned in **original-frame pixels**, so geometry works in original pixels.
- **Current system:** V12 + threshold **0.40** + temporal alarm layer (track by location, 2 s window, 30% hit ratio, 3 s off).
- **Hardware:** TI SK-AM62A-LP with a MIPI camera. The board camera model and lens are **not known yet**, so all geometry must load intrinsics from a file and work at any resolution.

## 4. Open questions to settle first
1. Which camera: the AM62A board camera (model, lens, resolution) or a phone for now?
2. Mounting: wall or ceiling, typical height and tilt? Is the floor assumed flat?
3. Ring distances, and output as a video overlay, a top-down map, or both?
4. Accuracy target (e.g. under 10% distance error at 5 m)?
5. On-site calibration process: checkerboard once, marked floor points, or automatic?
6. Integration point and output format (e.g. JSON per detection with `distance_m`)?

## 5. Suggested plan (to discuss)
1. **Calibration tool:** checkerboard intrinsics (OpenCV `calibrateCamera`) + floor homography from 4+ points, saved to `camera.json`.
2. **Geometry module:** pixel ↔ floor metres, distance and bearing of a box, undistortion.
3. **Range-ring overlay:** project floor circles into the image and draw them on video; optional top-down view.
4. **Detector integration:** add `distance_m` to detections and alarm events.
5. **Validation:**
   - tape marks at 1/2/3/5 m, camera on a stand at a known height
   - compare estimated vs measured distance
   - the zip's tiled-floor video is a first test: the tiles form a grid

## 6. GitHub
- **Repo:** https://github.com/Skandakm29/honeywell_bmsce_flame_detection
- **Latest code branch:** `feature/week2-final`
  - Create the W3 branch from it, e.g. `feature/w3-spatial-calibration`.
  - Don't merge into `main` without the team.

**Read these on that branch:**
- `docs/W3_BRIEF.md`: this file
- `docs/WEEK2_SUMMARY.md`: full week-2 write-up (system, data, models, metrics, limitations)
- `docs/WEEK2_CODE_INDEX.md`: what's in each folder

**Code on that branch:**
- `scripts/live_flame.py`: live and video viewer for the ONNX model (`prepare` = letterbox; `decode` = boxes in original pixels, NMS 0.65)
- `experiments/temporal_final/temporal_final.py`: alarm layer (`gate`, `track`, `raw_flags`, `alarm`)
- `experiments/field_tests/compare_v10box_v12.py`: runs models on a video with the alarm layer and writes side-by-side grids. Paths are hard-coded at the top; set `ROOT` and `MODELS` for the new machine.
- `experiments/edge_export/`: board export script (`export_raw.py --ckpt <pth> --name <tag>`) and board README
- `experiments/yolov9_v10/`: model definition (`workflow/model.py`) and TI tensorlab layers, needed to load `.pth` weights
- `experiments/yolov9_v12/`: V12 training package (how V12 was made)
- `results/week2/`: reports, the locked alarm rule (`rule.json`), test numbers

**Not in git (policy: no weights, ONNX, videos or data):** these come from the zip or the lab PC.

## 7. Starter zip: `W3_starter_V12.zip`

| File | What |
|---|---|
| `models/v12_best.pth` | V12 weights (SHA256 starts `4395912b`) |
| `models/flame_v12_full.onnx` | V12, full graph with decode: input `images` [1,3,640,640] RGB 0–1; output `detections` [1,8400,5] = x1,y1,x2,y2 (input pixels) + probability, pre-NMS. Works with `scripts/live_flame.py`. |
| `models/flame_v12_raw_640.onnx`, `_416.onnx` | Board versions without the decode head (see `README_V12.md`) |
| `README_V12.md`, `FINAL_RULE_V12.md`, `rule_v12.json` | Outputs, decode, preprocessing, accuracy, and the alarm rule for V12 |
| `videos/tiled_floor_candle_1m_3m_4m.MOV` | Fixed camera at floor level, candle on a tiled floor at about 1/3/4 m (1080p). Use it to try a homography from the tile grid. |
| `W3_BRIEF.md` | This brief |

## 8. Lab PC (for heavy runs)
- **GPU:** RTX PRO 4000 Blackwell, 24 GB. Windows.
- **Run by** its own Claude agent from zip packages (delivered through Google Drive).
- **Work folder:** `D:\Honeywell-V10-work`; Python venv: `D:\Honeywell-V10\.venv`; code: `D:\Honeywell-V10`, `D:\Honeywell-V12`
- **V12 weights:** `D:\Honeywell-V10-work\select_v12_rec\best.pth` (+ `export\flame.onnx`)
- **V10 weights:** `D:\Honeywell-V10-work\FROZEN_V10\selected.pth`
- **Datasets:** FASDD-CV v2 (Kaggle `yuulind/fasdd-cv-coco`, version 2) and the training lists in `D:\Honeywell-V10-work\manifests_v10` / `_v11` / `_v12`
- W3 geometry work doesn't need a GPU.

## 9. Other ongoing items (not W3)
- End-of-week fine-tune from V12 with small flames in **bright rooms** (the team is recording).
- Electronics teammate: int8 accuracy and frames per second for V12 at 640 vs 416. These decide the higher-resolution options.
