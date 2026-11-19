# Week 3 summary: spatial calibration

Branch `feature/w3-spatial-calibration` (from `feature/week2-final`). Full details and an AI hand-off prompt are in
[W3_SPATIAL_CALIBRATION.md](W3_SPATIAL_CALIBRATION.md).

## Built
- **Camera calibration** (`scripts/calibrate.py`, `src/geometry/camera.py`): a one-time setup that saves `camera.json`.
  Manual (measured height plus a few taped floor points, or a checkerboard) or automatic (height plus tilt from straight edges
  in the room).
- **Distance and bearing** for every detection: the bottom of the box is projected onto the floor using geometry only.
- **Range rings and a detectability zone** on the video (`src/geometry/overlay.py`).
- **Detector hookup:** `live_flame.py --camera` logs distance and bearing next to each detection.
- **Zoom crop** (demo stage): V12 also runs on a 320 px crop around the brightest spot, so small flames are detected.
- **HUD** with the Week 2 temporal rule: WARNING per frame, ALARM after 30% of frames over 2 s, off after 3 s.

## Results (fire_6 real video, phone at 0.77 m)
- Distance error **under 2% at every mark from 0.5 to 3 m** (geometry alone: under 1%) (mentor target: under 5%).
- With zoom, the alarm is confirmed at every mark from 0.5 to 3 m; no alarm before the candle appears.
- V12 unchanged: no retraining, same ONNX.
- 39 unit tests pass.

## Next
No-flame test for zoom, measured accuracy for automatic calibration, zoom in `live_flame.py`, second room and board camera.
