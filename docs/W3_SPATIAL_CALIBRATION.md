# W3 spatial calibration

Status: implemented; verified **only by unit tests** (synthetic cameras). Nothing here is measured on a real
camera yet; see section 8 for the validation plan.

## 1. What W3 delivers
- `src/geometry/camera.py`: `Camera` loaded from `camera.json`. Pixel to floor metres, floor to pixel, distance and
  bearing of a detection box (bottom-centre), undistortion, rescaling to any frame size.
- `scripts/calibrate.py`: `intrinsics`, `floor`, `pose` subcommands that write `camera.json`.
- `src/geometry/overlay.py`: range rings, detectability band, top-down map, CLI.
- `--camera camera.json` in `scripts/live_flame.py` and `experiments/field_tests/compare_v10box_v12.py`.
- Tests: `tests/test_geometry.py`, `test_calibrate.py`, `test_overlay.py` (run with `python -m pytest tests`).
- Unchanged: V12, threshold 0.40, the alarm rule.

## 2. `camera.json`
```json
{ "schema": 1, "image_size": [1920, 1080], "K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist": [k1,k2,p1,p2,k3],
  "extrinsics": {"R": [[...]], "C": [x, y, z]},
  "pose": {"height_m": 1.5, "tilt_deg": 25, "roll_deg": 0},
  "homography": [[...]], "contact_height_m": 0.0, "max_range_m": 200 }
```
- `K` (or `hfov_deg` for a rough pinhole) is valid at `image_size`; it is rescaled to the frame size automatically.
  A change in aspect ratio of more than 1% raises an error (a crop is not a resize).
- Geometry priority: `extrinsics` > `homography` > `pose`. `calibrate.py floor` writes `extrinsics` (planar PnP),
  so the floor plane and every contact height use one rigid camera model.
- `dist` is the OpenCV model (no fisheye).

Conventions:
- Floor frame: X right, Y forward, Z up, metres (right-handed). Origin can be anywhere for `floor`; the camera foot,
  heading, height and tilt are recovered (`cam.camera_xy_m`, `heading_deg`, `height_m`, `tilt_deg`).
- `tilt_deg` positive = looking down. `bearing_deg` 0 = along the camera heading, positive to the right.
- `distance_m` is the horizontal floor distance from the camera foot to the box bottom-centre.
- `contact_height_m`: the box bottom is the flame base, usually above the floor (candle/holder 0.05-0.15 m). The
  ray is intersected with the plane z = contact_height_m. Default 0; override per run with `--contact-height-m`.
- Points above the horizon, at or above camera height, or beyond `max_range_m` return `None`/NaN, never a number.

## 3. Calibration workflow
Run from the repo root; each step merges into `--out` (default `camera.json`).

1. Intrinsics (checkerboard, 15+ views at varied tilts; `--cols/--rows` are inner corners):
   ```
   python scripts/calibrate.py intrinsics --video board.mp4 --cols 8 --rows 6 --square-m 0.025 --out camera.json
   ```
   Prints RMS reprojection error (px). Re-running drops any old floor calibration (K changed).
2. Floor, from 6+ tape marks spread near and far (not collinear). `marks.csv` rows: `u,v,x_m,y_m`
   (pixel in the raw frame; floor metres with X right, Y forward):
   ```
   python scripts/calibrate.py floor --points marks.csv --out camera.json
   python scripts/calibrate.py floor --interactive --image frame.png --out camera.json   # click + type metres
   ```
   Prints in-sample and leave-one-out error in cm per point (4 points give no error estimate) and the recovered
   camera height/tilt. If no `intrinsics` was run yet: add `--hfov-deg 66 --size 1920x1080` (rough).
3. Or pose only, with the height and tilt measured by hand:
   ```
   python scripts/calibrate.py pose --height 1.5 --tilt 25 --roll 0 --hfov-deg 66 --size 1920x1080 --out camera.json
   ```

## 4. Overlay
```
python src/geometry/overlay.py --camera camera.json --video clip.mp4 --frame 150 --rings 1 2 3 5 10 \
    --flame-height-m 0.3 --min-px 25 --contact-height-m 0.05 --out preview.png --map topdown.png
python src/geometry/overlay.py --camera camera.json --video clip.mp4 --video-out rings.mp4
```
In code, build once and apply per frame (a one-shot `draw_rings` rebuilds everything and is slow):
```python
from geometry import Camera
from geometry.overlay import RingOverlay, detectability_mask
cam = Camera.load("camera.json")
ov = RingOverlay(cam, w, h, radii=(1, 2, 3, 5, 10), band_mask=detectability_mask(cam.with_resolution(w, h), 0.3))
frame = ov.apply(frame)
```
Rings are floor circles around the camera foot, clipped at the horizon, with a guard against lens-distortion
fold-back. Labels closer than 18 px vertically are skipped.

## 5. Detector integration
- `scripts/live_flame.py --camera camera.json [--contact-height-m 0.05]`: adds a `detections_geo` column to
  `frames.csv` (JSON list per frame, one entry per box, same order), draws rings, labels boxes "flame 0.62 3.2 m".
- `experiments/field_tests/compare_v10box_v12.py ... --camera camera.json`: rings and distance labels on the grid,
  `<out>/geo/<clip>_<panel>.jsonl` (per frame: `frame, t, alarm, dets[{box, score, ...geo}]`) and
  `first_alarm_geo` in the summary (`dets`, `geo_age_s`: the latest boxes within the alarm window, because the hit
  ratio can hold an alarm on a frame with no box).
  Also new: `--root` (default unchanged), `--model-v12 <onnx>`, `--max-seconds`; a missing V10 model skips that panel.
- Geo fields per detection: `distance_m, bearing_deg, x_m, y_m, contact_height_m` (None when above the horizon).
  `x_m, y_m` are in the `camera.json` floor frame.

## 6. Known limits
- Flat floor assumed. Slopes, steps, ramps and raised platforms give wrong distances.
- A low camera (a few cm to ~1 m) looks at the floor at a very shallow angle: near the horizon a 1 px error is many
  metres. Expect usable accuracy only at short range; far values are unreliable.
- Contact height bias: a wrong `contact_height_m` shifts distance, more so for low cameras and large heights.
- Aspect-ratio change (crop/zoom) invalidates K; recalibrate for the cropped mode. Letterboxing is not modelled.
- The detectability band is a pixel-size criterion only (flame at least `min_px` tall at the 640 input, upright
  flame, given contact height). It is **not** a measured detection rate; lighting, flame type and V12's behaviour
  are not modelled. Off-axis the band is slightly larger than a circle (pinhole size scales with depth).
- Intrinsics from a phone are only valid for the same lens and mode used for the checkerboard.

## 7. Verified so far
- Unit tests on synthetic cameras (round trips, rescale invariance, horizon handling, homography frame shift,
  contact-height continuity, rings/band, autocal): 39 pass (`pytest tests/test_geometry.py tests/test_calibrate.py
  tests/test_overlay.py tests/test_autocal.py`).
- Real camera: `fire6.mp4` (phone fixed on a stand, 0.77 m high, tilt about 21 deg, candle moved along taped marks
  0.5-3 m, room dark). See section 8.

## 8. Real-camera validation: fire_6 (manual calibration)
Setup: phone 1920x1080, HFOV 66 deg, height 0.77 m, tilt 20.75 deg, roll 0. The tape zero was 0.415 m in front of the
camera foot (tape offset), and the candle base sits 3 cm above the floor (`contact_height_m` 0.03).
Calibration file: `experiments/w3_validation/calibrations/camera_fire6.json`.

**Geometry (leave-one-out on hand-read candle bases):** 0.499 / 1.006 / 1.490 / 2.006 / 2.498 / 3.001 m for
0.5 / 1 / 1.5 / 2 / 2.5 / 3 m, under 1% everywhere.

**Detector boxes -> distance (V12 + zoom crop + temporal rule, 10 fps, calibrated at 1.5 m):**

| True (m) | Frames with detection | ALARM frames | Median estimate (m) | Error |
|---|---|---|---|---|
| 0.5 | 139/140 | 140/140 | 0.506 | +1.2% |
| 1.0 | 147/150 | 150/150 | 1.017 | +1.7% |
| 1.5 | 150/150 | 150/150 | 1.515 | cal |
| 2.0 | 130/130 | 130/130 | 2.017 | +0.9% |
| 2.5 | 140/140 | 140/140 | 2.512 | +0.5% |
| 3.0 | 160/160 | 160/160 | 2.991 | -0.3% |

No ALARM in the 9 s before the candle appears. Mentor target: under 5% error. Raw numbers:
`experiments/w3_validation/results/`.

## 9. Automatic calibration (tilt, roll, optional f; you give the height)
`src/geometry/autocal.py`, `python scripts/calibrate.py auto --video clip.mp4 --height 0.77 --hfov-deg 66 --debug dbg.png`
(or `--image`; K from `camera.json` if no `--hfov-deg`; `--frames N --t0 --t1` choose the median window).
Median of ~30 frames -> gamma/CLAHE -> line segments (LSD, else Canny+Hough) -> RANSAC vertical vanishing point
(door frames, wall corners, furniture) -> tilt/roll from `K^-1 * vp`. A second orthogonal VP gives
`f^2 = -(v1-c).(v2-c)`, used only with `--estimate-f` and only if its bootstrap std is under 10%. The height is
never estimated. It **refuses** (clear error, no file change) with fewer than 8 vertical inliers or a bootstrap
std above 1 deg. Valid for |tilt| < 45 deg (above that the vertical and horizon VPs are ambiguous).
`auto` writes `pose` and drops any floor calibration, like `pose`.

**Measured: unit tests** (synthetic lines, 0.1-0.3 px noise): tilt/roll within 0.5 deg for tilt 8-40 deg, f within 2%
with 40 receding lines; refusal on too few or too noisy lines (`tests/test_autocal.py`).

**Measured: real clips** (`experiments/w3_validation/autocal_fire6.py`, median of 30 frames, nominal hfov as given):

| Clip | Manual (marks fit) | Auto | Verdict |
|---|---|---|---|
| `fire6.mp4`, dark room, h=0.77, hfov 66 | tilt 20.75, roll 0 | **refused** (0-9 s: 7 segments, 6 vertical inliers; whole clip: 10 segments, 8 inliers, tilt std 23 deg) | not usable |
| `tiled_floor_candle_1m_3m_4m.MOV`, h=0.23, hfov 69 | tilt 4.4 (h, f assumed), roll 0 | tilt **5.10** (bootstrap std 0.42), roll -0.03 (0.07); 92 vertical inliers, RMS residual 0.65 deg, 43 horizontal inliers | +0.7 deg vs manual; f from 2 VPs discarded (std 19%) |

Caveats:
- fire6 median frame has mean brightness 9/255; after CLAHE only a door edge and a few furniture edges are visible. Lowering
  the minimum segment length (30/20/12 px) gave 15/33/68 segments but tilt 30/-7/24 deg with std 4-22 deg: no
  reliable vertical VP. Dark empty rooms are outside what this method can do; use `pose` or `floor` there.
- The tiled-floor "manual" value is itself a fit with assumed f and h, so +0.7 deg is a consistency check, not an
  error against ground truth. One scene only. The 6-mark distance check on fire6 could not be run because
  auto-calibration refused; no auto-calibrated distance error has been measured.
- Focal length from two VPs is poorly conditioned at small tilt; do not rely on it without many long
  horizontal lines.

## 10. Zoom crop (detection at small flame sizes)
At full frame a 1920x1080 image is shrunk 3x to the 640 input, so a small candle at 1-2.5 m becomes a few pixels and
V12 can miss it (glow and reflections also hurt). Zoom crop: blur the gray frame, take the brightest point, crop a CxC
square around it (C = 320 px), resize to 640, run V12, then map boxes back to full-frame pixels so the calibration
still applies. Full frame vs zoom on fire_6 (frames with a detection at score >= 0.40):

| Mark | 0.5 m | 1 m | 1.5 m | 2 m | 2.5 m | 3 m |
|---|---|---|---|---|---|---|
| Full frame | 140/140 | 8/150 | 139/150 | 18/130 | 3/140 | 160/160 |
| Zoom 320 | 139/140 | 147/150 | 150/150 | 130/130 | 140/140 | 160/160 |

The model is unchanged (no retraining, same ONNX). Zoom is a demo stage in `experiments/w3_validation/` and is not
yet in `live_flame.py`. Still to check: other bright sources (lamps, torch, LED, phone screen) pulling the crop,
several flames at once, and run time on the board.

## 11. Temporal alarm on the HUD
Same locked rule as Week 2 (`W3_starter_V12/rule_v12.json`, code `experiments/temporal_final/temporal_final.py`):
tracking by location; WARNING = any box >= 0.40 in the frame; ALARM = a track seen in >= 30% of frames over 2 s;
ALARM off after 3 s with no qualifying track. The W3 render scripts call `TF.track`, `TF.raw_flags` and `TF.alarm`.

## 12. Reproducing the W3 results
Data is not in git. Put `W3_starter_V12/` (model + rule) and `w3_demo/drive_videos/fire6.mp4` in the folder above
the repo, or set `W3_DATA` (that folder), `W3_VIDEO` (clip) and `W3_OUT` (where videos/JSON go).

| Script (`experiments/w3_validation/`) | What it does |
|---|---|
| `fit_pose.py` | fits height/tilt to known-distance pixels |
| `fire6_validate.py` | leave-one-out geometry check on fire_6 |
| `fire6_detect_render.py` | detector distances (full frame) + ring video |
| `zoom_test.py` | detection counts per crop size per mark |
| `fire6_zoom_alarm_render.py` | full vs zoom, both with the temporal alarm; writes `fire6_zoom_alarm_results.json` |
| `final_inference_render.py` | single-view presentation video (zoom + alarm + distance + rings) |
| `autocal_fire6.py` | automatic vs manual calibration on a clip |

## 13. Manual vs automatic calibration: which to use
- **Manual (default):** measured height plus a few taped floor points (`calibrate.py floor` or `pose`). Proven on
  real video (section 8). This is also how fixed cameras are usually commissioned.
- **Automatic (option):** `calibrate.py auto` with the measured height; tilt and roll come from vertical edges. Works in
  lit rooms with straight edges; refuses in dark or bare scenes. Distance accuracy not yet measured on a taped scene.
- Recommended install flow: automatic first estimate, confirmed with 2-3 taped points.

## 14. Open items
1. No-flame clip (lamp, torch, LED, phone screen with fire) to check zoom false alarms.
2. Lit-room clip from the fire_6 position to measure automatic-calibration distance error.
3. Move zoom + alarm HUD into `scripts/live_flame.py` (behind a flag).
4. Second room and the board camera; zoom latency on the board.
5. Ring distances for the HUD: not yet given by the mentor (currently the tape marks).

## 15. Prompt for an AI assistant continuing this work
Copy the block below into a new AI session.

```text
You are continuing Week 3 (spatial calibration) of the Honeywell/BMSCE flame-detection project.
Repo: github.com/Skandakm29/honeywell_bmsce_flame_detection, branch feature/w3-spatial-calibration
(based on feature/week2-final). Never merge into main. Do not commit weights, ONNX, videos or datasets.

Read first: docs/W3_BRIEF.md, docs/WEEK2_SUMMARY.md, docs/WEEK3_SUMMARY.md, then
docs/W3_SPATIAL_CALIBRATION.md.

Already done (do not redo):
- The detector is V12 (YOLOv9-S-TI-Lite, v12_best.pth sha256 4395912b...), threshold 0.40, with the locked temporal
  rule (rule_v12.json: tracking by location, 2 s window, hit ratio 0.30, alarm off after 3 s; code in
  experiments/temporal_final/temporal_final.py). The model was NOT retrained or changed in W3.
- src/geometry/camera.py: pinhole camera (K, distortion, pose or extrinsics or homography) stored in camera.json
  (schema 1). pixel_to_floor / floor_to_pixel / box_to_floor: bottom-centre of a box -> ray -> plane at
  contact_height_m -> distance and bearing from the camera foot.
- scripts/calibrate.py: intrinsics (checkerboard), floor (taped points, PnP, leave-one-out error), pose
  (measured height/tilt/HFOV), auto (tilt/roll from the vertical vanishing point; you give the height).
- src/geometry/overlay.py: range rings, detectability band (flame >= 25 px at the 640 input), top-down map.
- scripts/live_flame.py and experiments/field_tests/compare_v10box_v12.py take --camera camera.json and log
  distance/bearing.
- experiments/w3_validation/: fire_6 real-video validation (manual calibration, distances within about 1% from
  0.5-3 m), a zoom crop (320 px crop around the brightest spot, resized to 640) that fixes misses at 1-2.5 m, and render
  scripts with the WARNING/ALARM HUD. Results in experiments/w3_validation/results/.

Key facts: the camera height must be measured (it cannot come from the image). In fire_6 the tape zero was 0.415 m
in front of the camera foot. The local Python env has no scipy and no torch (onnxruntime + OpenCV only).
Unit tests: pytest tests/test_geometry.py tests/test_calibrate.py tests/test_overlay.py tests/test_autocal.py

Open work: section 14 of docs/W3_SPATIAL_CALIBRATION.md (false-alarm check for zoom, measured accuracy of
automatic calibration, zoom in live_flame.py, board camera). Before changing anything, run the tests, keep
changes small, and report measured numbers, not guesses.
```
