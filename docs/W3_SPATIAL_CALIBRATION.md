# W3 spatial calibration

Status: implemented and measured on real video. Same-clip check on fire_6 (section 8.1), independent test on
15 clips at 1.5 m (section 8.2), zoom on weak candidates and the live alarm rule in `live_flame.py` (sections 10-11).
50 unit tests pass. The V12 model is unchanged (no retraining).

## 1. What W3 delivers
- `src/geometry/camera.py`: `Camera` loaded from `camera.json`. Pixel to floor metres, floor to pixel, distance and
  bearing of a detection box (bottom-centre), undistortion, rescaling to any frame size.
- `scripts/calibrate.py`: `intrinsics`, `floor`, `pose` subcommands that write `camera.json`.
- `src/geometry/overlay.py`: range rings, detectability band, top-down map, CLI.
- `scripts/calibrate.py auto`: tilt and roll from vertical edges in the room (section 9).
- `src/detection/zoom.py`: zoom on weak candidate boxes (section 10).
- `src/detection/temporal.py`: the locked Week 2 alarm rule fed one frame at a time, for live use (section 11).
- `scripts/live_flame.py`: `--camera camera.json` (distance, bearing, rings), `--zoom`, `--alarm-rule rule_v12.json`.
  `experiments/field_tests/compare_v10box_v12.py` also takes `--camera`.
- `experiments/w3_validation/`: real-video validation scripts, calibrations, results and the demo renderer (section 12).
- Tests: `tests/test_geometry.py`, `test_calibrate.py`, `test_overlay.py`, `test_autocal.py`, `test_zoom.py`,
  `test_temporal_stream.py` (run with `python -m pytest tests`).
- Unchanged: the V12 model and its full-frame threshold 0.40, and the alarm rule's settings (`rule_v12.json`).

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
- `scripts/live_flame.py --zoom` (section 10): adds `detections_from_zoom` and `zoom_crops_xywh` columns; zoom boxes
  are orange, crop squares cyan. Settings: `--zoom-confidence 0.20`, `--zoom-candidate-min 0.10`, `--zoom-crop 320`,
  `--zoom-max-crops 3`; `--confidence` stays the full-frame threshold (0.40).
- `scripts/live_flame.py --alarm-rule rule_v12.json` (section 11): CLEAR / WARNING / ALARM on screen and
  `alarm_state`, `best_hit_ratio` columns. Time is video seconds for files, a monotonic clock for cameras.
- Full live command:
  `python scripts/live_flame.py --source <video|camera> --model flame_v12_full.onnx --camera camera.json --zoom --alarm-rule rule_v12.json`

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
- Unit tests: geometry on synthetic cameras (round trips, rescale invariance, horizon handling, homography frame
  shift, contact-height continuity, rings/band, autocal), zoom (candidates, crop windows, merge, mapping back) and
  the live alarm rule: 50 pass (`pytest tests/test_geometry.py tests/test_calibrate.py tests/test_overlay.py
  tests/test_autocal.py tests/test_zoom.py tests/test_temporal_stream.py`).
- Real camera: `fire6.mp4` (phone at 0.77 m, candle on taped marks 0.5-3 m, dark room) and 15 clips at 1.5 m
  (two setups, lights on/off, no-flame clips). See section 8.

## 8. Real-camera validation

### 8.1 fire_6, same clip (manual calibration)
Setup: phone 1920x1080, HFOV 66 deg, height 0.77 m, tilt 20.75 deg, roll 0. The tape zero was 0.415 m in front of the
camera foot (tape offset), and the candle base sits 3 cm above the floor (`contact_height_m` 0.03).
Calibration file: `experiments/w3_validation/calibrations/camera_fire6.json`.

**Geometry (leave-one-out on hand-read candle bases):** 0.499 / 1.006 / 1.490 / 2.006 / 2.498 / 3.001 m for
0.5 / 1 / 1.5 / 2 / 2.5 / 3 m, under 1% everywhere. Caveat: height, tilt and the tape offset were fitted on this same
clip, so this is a same-clip (leave-one-out) number, not an independent test. Detector distances on the same clip:
under 2%. Independent test (calibrate on one clip or position, test on others): see the Phase A results.

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

### 8.2 Independent test: 15 clips with the camera at 1.5 m (Phase A)
Clips (Drive folder `Flame_Videos`, not in git): two camera setups, lens 1.5 m above the floor, tape marks measured
from the point below the lens. Setup "39": 1, 1.5, 2, 2.5, 3 m. Setup "44": 0.5, 1, 1.5, 2 m. Candle runs with the room
light on and off, runs without tape marks (camera slightly moved), and 6 no-flame clips (dim, lit, empty room).
Files are renamed locally: `m39_*`/`m44_*` = marked candle runs, `u39_*`/`u44_*` = unmarked, `nf*` = no flame.

Calibration was kept separate from the test data:
- Tilt and field of view from the **tape marks only** (height fixed at the measured 1.5 m): setup 39 tilt 42.3 deg,
  HFOV 61.5 deg (fit residual 0.4 cm over 5 tapes); setup 44 tilt 53.9 deg with the same HFOV. The angles written on
  the clips (39, 44 deg) were not the true downward tilt; the tapes measure it, so hand-measured angles are not needed.
  The phone's HFOV is about 62 deg, not the 66 deg assumed for fire_6.
- Contact height (flame base above floor, 11.4 cm) from the 1 m mark of one clip only (`m39_1to3_lighton`).
- Calibration file: `experiments/w3_validation/calibrations/camera_1p5m_setup39.json`.

Results (detector at 10 fps, locked alarm rule; method D = the chosen design of section 10):

| Check | Result |
|---|---|
| No-flame clips (6, about 1.5 min) | 0 boxes, 0 false alarms for every method |
| Distance, setup 39 (1-3 m), independent | median error within 2% at every mark |
| Distance, setup 44 (0.5-2 m) | 9-22% short: the flame box there is about 300 px tall because it includes the light streak and floor glow under the flame, so the box bottom lands below the candle. Calibration fits the tapes; the box is the problem. |
| Daylight-glare runs (`m39_*_lightoff`) | V12 barely sees the candle (best score usually under 0.08); no method after the model fixes this |

Raw numbers: `experiments/w3_validation/results/phaseA_1p5m_results.json`; scripts `phaseA_detect.py` (runs V12 on
every clip and caches full-frame and zoom boxes) and `phaseA_table.py` (calibration + table).

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

## 10. Getting small or weak flames detected: options tested, design chosen
Why flames were missed on fire_6: the model **saw** the candle at every distance but scored it 0.16-0.29 at 1, 2 and
2.5 m (threshold 0.40); box sizes were similar at all distances, so it is appearance (glow, overexposure, reflection),
not size. The temporal rule cannot fix this: it only confirms boxes the model already gives.

Options tested (same clips, same alarm rule):
1. **Brightest-spot zoom** (first version): crop 320 px around the brightest blurred pixel, resize to 640, run V12.
   Caught every fire_6 mark, but replaced the full frame (misses big fires and second flames), followed lamps or
   reflections (brightest spot), and turned the floor reflection into flame-like blobs (104 reflection boxes at 2 m).
2. **Lower full-frame threshold** with the alarm rule (`fire6_threshold_sweep.py`): 0.20 catches most marks on fire_6,
   0.15 all; below 0.20 reflection boxes grow fast (7 -> 81 -> 260).
3. **Zoom on weak candidates** (chosen): see below.
4. Not tested yet: full frame at 960 input (needs a re-export with torch), flicker and reflection checks.

**Chosen design (`src/detection/zoom.py`, `live_flame.py --zoom`):**
- The full frame always runs at 0.40, so nothing the old pipeline found is lost.
- Weak full-frame boxes (score 0.10-0.40) are candidates; up to 3 per frame, strongest first.
- Around each, a 320 x 320 px crop is cut from the original frame (centre snapped to a 32 px grid so it does not
  jitter), resized to 640 (2x upscale from 1080p), and run through V12. Crop boxes at >= 0.20 are mapped back to
  full-frame pixels.
- A crop box is dropped as a duplicate if it overlaps a kept box (IoU >= 0.30) or either box's centre lies inside the
  other. Frames without weak candidates cost one model run, as before.

Phase A (section 8.2), alarm coverage while the candle is visible:

| Method | Candle clips (median, range) | No-flame clips | CPU per frame (laptop) |
|---|---|---|---|
| A: full frame 0.40 (before) | 68% (0-100%) | 0 false alarms | about 100 ms |
| B: full frame 0.20 | 93% (0-100%) | 0 false alarms | about 100 ms |
| C: full 0.40 + zoom boxes 0.40 | 99% (0-100%) | 0 false alarms | about 180 ms |
| **D: full 0.40 + zoom boxes 0.20 (chosen)** | **99% (0-100%)** | **0 false alarms** | **about 180 ms** |

D over C: recovers the 3 -> 1 m daylight run (0% -> 78%) and lifts the 44 deg light-off run (92.5% -> 96%) at the same
compute; about twice as many boxes on glow/reflection next to the flame, none of which caused a false alarm. On the
1.5 m clip D was the only method that reached 3 m. Laptop CPU times are only for comparing methods; board timing is
still to be measured.

Known limits: zoom still marks the floor reflection under the candle (visible in the demo at 2 and 2.5 m; the
headline distance uses the full-frame flame first); daylight glare defeats V12 itself; the box-bottom distance is
wrong when the box includes floor glow (setup 44). Not yet tested: lamps, torch, LED, phone screen, several flames.


## 11. Temporal alarm (offline and live)
The locked Week 2 rule (`rule_v12.json`): tracking by location; WARNING = any box in the frame; ALARM = a track seen
in >= 30% of frames over 2 s; ALARM off after 3 s with no qualifying track. Not re-tuned in W3.
- Offline (whole recorded clip): `experiments/temporal_final/temporal_final.py` (`TF.track`, `TF.raw_flags`,
  `TF.alarm`), used by the validation scripts.
- Live (one frame at a time): `src/detection/temporal.py` (`AlarmRule.from_json('rule_v12.json')`,
  `update(t, boxes)` -> CLEAR / WARNING / ALARM). Same rule and numbers; on all 14,996 Phase A frames it gives
  exactly the same ALARM frames as the offline code (0 differences), for both the full-frame and the zoom pipeline.
- Boxes passed to it are already thresholded (full frame 0.40, zoom 0.20).


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
| `fire6_fullframe_temporal.py` | fire_6 full frame vs brightest-spot zoom with the alarm rule; counts boxes off the candle |
| `fire6_threshold_sweep.py` | fire_6 full frame at thresholds 0.40-0.10 with the alarm rule |
| `phaseA_detect.py` | runs V12 (full frame + zoom on candidates) on the 1.5 m clips at 10 fps and caches the boxes |
| `phaseA_table.py` | tape-mark calibration, contact height from one mark, and the method table (section 8.2) |
| `demo_render.py` | W3 demo video: calibration card, detection, zoom inset, live alarm, labelled range rings |

Phase A clips go in `test_videos_1p5m/` next to the repo (renamed as in section 8.2). Demo:
`python experiments/w3_validation/demo_render.py --video m39_1to3_lighton.mp4 --camera
experiments/w3_validation/calibrations/camera_1p5m_setup39.json --model flame_v12_full.onnx --rule rule_v12.json
--out W3_demo.mp4`

## 13. Manual vs automatic calibration: which to use
- **Manual (default):** measured height plus a few taped floor points (`calibrate.py floor` or `pose`). Proven on
  real video, including an independent test at 1.5 m (section 8.2). This is also how fixed cameras are usually
  commissioned. Hand-measured tilt angles are not needed: the tape marks give the tilt.
- **Automatic (option):** `calibrate.py auto` with the measured height; tilt and roll come from vertical edges. Works in
  lit rooms with straight edges; refuses in dark or bare scenes. Distance accuracy not yet measured on a taped scene.
- Recommended install flow: automatic first estimate, confirmed with 2-3 taped points.

## 14. Open items
1. Bright non-flame sources (lamp, torch, LED candle, orange lights, phone screen playing fire): check zoom and the
   alarm do not fire (the 8 Nov clips cover most of these).
2. Reflection check for zoom boxes (mirror shape, flicker in step with the flame above, floor geometry): flag, not
   delete, so a real second flame is never dropped.
3. Distance when the box includes floor glow (setup 44): use the flame's bright core or the box top instead of the
   box bottom, then re-run section 8.2.
4. Daylight glare: V12 misses the candle; needs camera exposure control or more training data.
5. Automatic calibration: measure its distance error on a taped scene with straight vertical edges.
6. Board: zoom and alarm latency on the AM62A; full frame at 960 input as a comparison.
7. More setups (2-3 heights, diya and lighter, a second room) for the full comparison table.


## 15. Prompt for an AI assistant continuing this work
Copy the block below into a new AI session.

```text
You are continuing Week 3 (spatial calibration) of the Honeywell/BMSCE flame-detection project.
Repo: github.com/Skandakm29/honeywell_bmsce_flame_detection, branch feature/w3-spatial-calibration
(based on feature/week2-final). Never merge into main. Do not commit weights, ONNX, videos or datasets.
Do not add AI co-author or attribution lines to commits or PRs.

Read first: docs/W3_BRIEF.md, docs/WEEK2_SUMMARY.md, docs/WEEK3_SUMMARY.md, then
docs/W3_SPATIAL_CALIBRATION.md (sections 8, 10, 11 and 14 matter most).

Already done (do not redo):
- Detector: V12 (YOLOv9-S-TI-Lite, v12_best.pth sha256 4395912b...), NOT retrained or changed in W3. Full-frame
  threshold 0.40. Locked alarm rule rule_v12.json (tracking by location, 2 s window, hit ratio 0.30, off after 3 s).
- Geometry: src/geometry/camera.py (camera.json schema 1; box_to_floor = bottom-centre of the box -> ray -> plane at
  contact_height_m -> distance and bearing from the camera foot), src/geometry/overlay.py (range rings,
  detectability band), scripts/calibrate.py (intrinsics, floor = taped points with PnP, pose, auto = tilt/roll from
  vertical edges; the height must always be measured).
- Detection: src/detection/zoom.py (full frame at 0.40 + up to 3 crops of 320 px around weak boxes 0.10-0.40, crop
  boxes kept at >= 0.20, duplicates dropped) and src/detection/temporal.py (the locked alarm rule fed one frame at a
  time; identical to experiments/temporal_final/temporal_final.py on 14,996 test frames).
- scripts/live_flame.py --camera camera.json --zoom --alarm-rule rule_v12.json runs all of it live.
- Validation (experiments/w3_validation/, results/): fire_6 same-clip distance error under 2%; independent test on
  15 clips at 1.5 m: setup 39 within 2% at 1-3 m, setup 44 9-22% short (box includes floor glow), 0 false alarms on
  6 no-flame clips, zoom design D alarms on the candle 99% median vs 68% before. demo_render.py makes the demo video.

Key facts: the phone HFOV is about 62 deg; contact height about 11 cm for the tall candle; the local Python env has
no scipy and no torch (onnxruntime + OpenCV only). Tests: python -m pytest tests/test_geometry.py
tests/test_calibrate.py tests/test_overlay.py tests/test_autocal.py tests/test_zoom.py tests/test_temporal_stream.py

Open work: section 14 of docs/W3_SPATIAL_CALIBRATION.md. Before changing anything, run the tests, keep changes small,
calibrate and test on different clips, and report measured numbers, not guesses.
```
