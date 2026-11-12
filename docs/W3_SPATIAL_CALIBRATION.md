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
Only unit tests on synthetic cameras (round trips, rescale invariance, horizon handling, homography frame shift,
contact-height continuity under noisy clicks, rings/band). No real-camera number exists yet.

## 8. Validation plan (to run; fill the table)
1. Camera on a stand at a measured height and tilt, flat floor. Run `intrinsics` once (same lens/mode).
2. Tape marks at 1, 2, 3, 5 m along the heading plus 2-3 off-axis marks. Calibrate `floor` with 6+ marks; keep 3
   extra marks out of the fit as a check set.
3. Put the candle/flame on each mark (note `contact_height_m`), record, run `live_flame.py --camera` headless,
   take the median `distance_m` over frames with detections.

| True distance (m) | Bearing (deg) | Median estimate (m) | Error (m) | Error (%) | Frames detected | Notes |
|---|---|---|---|---|---|---|
| 1 | 0 | | | | | |
| 2 | 0 | | | | | |
| 3 | 0 | | | | | |
| 5 | 0 | | | | | |
| 3 | off-axis | | | | | |

Target to agree with the mentor (open question): e.g. under 10% error at 5 m.
