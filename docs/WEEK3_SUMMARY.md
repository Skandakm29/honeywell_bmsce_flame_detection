# Week 3 summary: spatial calibration

Branch `feature/w3-spatial-calibration` (from `feature/week2-final`). Full details, reproduction steps and an AI
hand-off prompt are in [W3_SPATIAL_CALIBRATION.md](W3_SPATIAL_CALIBRATION.md).

## Built
- **Camera calibration** (`scripts/calibrate.py`, `src/geometry/camera.py`): a one-time setup that saves `camera.json`.
  Manual (measured height plus a few taped floor points; the default) or automatic (height plus tilt from straight
  edges in the room).
- **Distance and bearing** for every detection, from geometry only.
- **Range rings with distance labels and a detectability zone** on the video (`src/geometry/overlay.py`).
- **Zoom on weak candidates** (`src/detection/zoom.py`): the full frame runs at 0.40 as before; weak boxes (0.10-0.40)
  get a 320 px crop that is re-checked by V12 and kept at >= 0.20.
- **Live alarm** (`src/detection/temporal.py`): the locked Week 2 rule (30% of frames over 2 s, off after 3 s) fed one
  frame at a time; identical to the offline code on 14,996 test frames.
- **Live pipeline:** `live_flame.py --camera camera.json --zoom --alarm-rule rule_v12.json`.
- **Demo video:** `experiments/w3_validation/demo_render.py`.

## Results
| Test | Result |
|---|---|
| fire_6 (camera 0.77 m), same clip | distance error under 2% from 0.5 to 3 m |
| 15 clips, camera 1.5 m, calibrated from tapes only (independent) | distance within 2% at 1-3 m (setup 39); 9-22% short in setup 44, where the box includes floor glow |
| No-flame clips (6) | 0 false alarms |
| Alarm on the candle while visible (median over candle clips) | 68% before, 99% with zoom on weak candidates |

V12 is unchanged (no retraining, same ONNX). 50 unit tests pass.

## Next
Bright non-flame sources (lamps, LEDs, screens), a reflection check for zoom boxes, distance when the box includes
floor glow, daylight glare, board latency, and more camera heights and flame types.
