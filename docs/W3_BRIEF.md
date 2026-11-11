# Week 3 brief: Spatial Calibration (for the next agent)

**Written:** 11 Nov 2026. **Owner:** Utkrisht (our team). **W4 obstruction detection** is handled by teammates in parallel.

## 1. The milestone (from the project plan)

| Week | Milestone | Deliverable |
|---|---|---|
| **W3** | **Spatial Calibration:** range rings, ground-plane geometry and system calibration | Codebase expansion and integration with the existing system setup |
| W4 | Obstruction Detection: blind-zone mapping and occlusion monitoring | Software update integrated into the master code branch |

**Mentor, 9 Nov:**
- W3 and W4 are **combined** and reviewed together at the end of W4.
- A midweek check-in is possible on Thursday or Friday.
- The team has exams from **20 Nov**.

## 2. What W3 means for a camera flame detector (our interpretation; confirm with the mentor)
- **System calibration:** know the camera's geometry.
  - **Intrinsics:** focal length, principal point and lens distortion, from a checkerboard.
  - **Extrinsics:** mounting height, tilt and roll relative to the floor.
- **Ground-plane geometry:** map every image pixel on the floor to a real (x, y) position in metres. This uses a homography from 4 or more known floor points, or the intrinsics plus height and tilt.
- **Range rings:** draw distance circles on the floor in the image (e.g. 1, 2, 5, 10 m from the camera). They show coverage and how far a detection is.
- **How it links to the flame detector:**
  1. **Distance to a detected flame:** the bottom of the flame box touches the floor, so the ground plane gives its distance.
  2. **Expected flame size per distance:** this gives a coverage map of where a candle-sized or bigger fire can be detected.
  3. **For W4:** blind-zone area in m² on the floor, not just % of image.

## 3. Facts from weeks 1–2 that matter for W3 (Measured)
- **Detection needs pixels:** a flame must be about 20–30 px tall at the 640-px model input.
  - A candle (about 3 cm) is reliable up to about **1 m** with a 1080p phone.
  - Larger flames scale roughly linearly with size.
  - The range rings should show this "detectable range per flame size".
- The model sees frames letterboxed to 640 (top-left placed, pad 114). Map all pixel coordinates back to the **original frame** before any geometry.
- **Final week-2 system:** V12 + threshold 0.40 + temporal alarm layer (2 s window, 30% hit ratio, 3 s off).
- **Hardware:** TI SK-AM62A-LP with a MIPI camera. The board camera model and lens are **not known yet**. Calibration has to be redone for the real camera, so the code must take any intrinsics.
- **Higher resolution and a two-stage zoom** are approved by the mentor if latency is acceptable. They're not part of W3, but the geometry code should work at any resolution.

## 4. Open questions to settle early (ask the mentor or team)
1. **Which camera:** the AM62A board camera (model, lens, resolution) or a phone for now?
2. **Mounting:** wall or ceiling, typical height and tilt. Is the floor assumed flat?
3. **Range rings:** fixed distances (1/2/5/10 m)? Drawn on the live video, or as a top-down map?
4. **Accuracy target:** e.g. distance error under 10% at 5 m?
5. **Calibration process on site:** a checkerboard once? Marking 4 floor points? Automatic?
6. **Integration:** where does it plug in (the live viewer `scripts/live_flame.py`, the board pipeline)? Which output format (JSON per detection with distance in metres)?

## 5. Suggested plan (to discuss, not decided)
1. **Calibration tool:**
   - checkerboard intrinsics (OpenCV `calibrateCamera`) → saved in a `camera.json`
   - floor homography from 4+ clicked or marked points → saved alongside
2. **Geometry module:** pixel ↔ floor (x, y) metres; distance and bearing for a box's bottom centre; undistortion.
3. **Range rings overlay:** project circles from the floor into the image and draw them on video; optional top-down map.
4. **Detector integration:** add `distance_m` to each detection and alarm event; colour the rings by detectability for a given flame size.
5. **Validation:**
   - Test: tape marks at 1/2/3/5 m, phone on a stand at a known height; compare estimated vs measured distance.
   - Reuse the planned bright-room flame recordings at taped distances. They serve both W3 and the next fine-tune.

## 6. Where everything is

**Docs (read first):**
- `E:\Projects\honeywell\investigation_week2\CLAUDE.md`: running notes, rules (never edit originals; label Measured / Estimate / Hypothesis; short, plain answers)
- `E:\Projects\honeywell\investigation_week2\WEEK2_SUMMARY.md`: full week-2 write-up and file map
- `E:\Projects\honeywell\HANDOVER.md`, `PROJECT_CONTEXT.md`: original project background

**GitHub:**
- Repo: https://github.com/Skandakm29/honeywell_bmsce_flame_detection
- Branch: `feature/week2-final` (week-2 code; index in `docs/WEEK2_CODE_INDEX.md`). For W3, create a **new branch from it**, e.g. `feature/w3-spatial-calibration`.
- W4 teammates will work on their own branch; agree on a shared detection output format.
- Local clone: `E:\Projects\honeywell\honeywell_bmsce_flame_detection`. It's on `feature/yolov9-flame-colab` with uncommitted doc edits; don't touch those. Use a separate worktree.

**Models:**
- **V12 (main):** `investigation_week2\lab_backup_2026-11-07\work\select_v12_rec\best.pth` (SHA 4395912b); ONNX in the same folder `export\flame.onnx`
- Board ONNX files (V10 and V12, raw head, 640 and 416): `investigation_week2\edge_export\`
- Board handoff zips: `V12_board_handoff.zip`, `V10_final_rule_for_board.zip`

**Code to reuse:**
- Live viewer pre- and post-processing: `honeywell_bmsce_flame_detection\scripts\live_flame.py` (`prepare`, `decode`)
- Video runner with the alarm layer and grids: `testing\compare_v10box_v12.py`
- Alarm layer: `investigation_week2\temporal_final\temporal_final.py` (gate, tracker, alarm)
- Rules: `edge_export\rule_v12.json`, `week2_final\rule.json`

**Test videos:**
- `testing\videos_2026-11-09\IMG_7936.MOV`: candle at roughly 1/3/4 m on a tiled floor, fixed camera. The tiles are a usable grid to try a homography.
- `testing\videos_2026-11-08\`, `testing\samples\`
- Side-by-side videos: `testing\PPT_grids_V10box_vs_V12\`

**Lab PC:**
- RTX PRO 4000 Blackwell; `D:\Honeywell-V10-work`, venv `D:\Honeywell-V10\.venv`
- Its own Claude agent runs zipped packages delivered through Google Drive.
- W3 geometry work should run fine on the laptop CPU.

## 7. Other ongoing items (not W3, don't drop)
- End-of-week fine-tune from V12 with small flames in **bright rooms**; the team is recording.
- Electronics teammate: int8 accuracy and frames per second for V12 (640 vs 416). These numbers decide the higher-resolution options.
