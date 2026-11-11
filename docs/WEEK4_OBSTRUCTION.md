# Week 4: obstruction detection

**Last updated:** 11 Nov 2026

## Requirement (mentor, 9 Nov; `WEEK2_SUMMARY.md` section 12)
When the camera's view is partly blocked by an added object (a steel sheet, a box…), the system must detect the obstruction and report how much of the view is blind.

**Decision (week 4):** add a class `obstruction` to the **same** YOLOv9-S-TI-Lite flame model. One model and one pass per frame on the AM62A. There is no second model and no extra pass.

## How it works

```
Camera frame -> letterbox 640 -> YOLOv9-S-TI-Lite (2 classes) -> per-class NMS
   flame boxes        -> unchanged flame path (small-box threshold, temporal alarm)
   obstruction boxes  -> blind share = union area of obstruction boxes / frame area
```

**The model** (`experiments/obstruction_w4/`):
- The trained 1-class model grows to 2 classes by widening only the last classifier layer.
- Before training, flame output is bit-identical. It is then fine-tuned with V10's unchanged training loop.
- No real labelled obstruction data exists yet, so training uses **synthetic obstructions** drawn on the training images: boxes, steel sheets, boards, cloth, hands, lens covers and defocused objects, entering from the edges, inside the view or covering it all.
- Selection only accepts a checkpoint whose flame numbers stay within 1 point of the starting model and whose obstruction false alarms stay at or below 2%. See `PROTOCOL_W4.md`.

**The live viewer** (`scripts/live_flame.py`):
- It detects a 2-class export automatically from its 6 output fields.
- It draws obstruction boxes in blue and shows `VIEW BLOCKED ~xx% of frame`.
- It logs `obstructions_xyxy_score` and `blind_percent` in `frames.csv`.
- With a 1-class model, its output is unchanged (checked frame by frame against the previous version).

```powershell
python scripts\live_flame.py --model <w4 flame.onnx> --source 0 [--obstruction-confidence 0.40]
```

## Status

| Item | State |
|---|---|
| Model growth, data, metrics, selection, export, viewer | Done; unit tests in `tests/test_obstruction_class.py`; CPU smoke run of every step passed |
| Training on the lab GPU | **Not run yet** (`experiments/obstruction_w4/START_HERE.md`) |
| Real-camera test | **Not run yet**: needs the trained model and the recordings below |
| Board (raw-head export for 2 classes, int8, AM62A speed) | Not done |

## Real-camera test (after training; reported once)
**Record** with the phone or webcam **on a tripod**. In each clip, start with a clear view for 5 s, then bring the object in, hold it for 10 s and take it away. Name clips `<place>_<object>_<position>_<coverage>.mp4`.

| Set | Clips |
|---|---|
| Objects | cardboard box, steel or metal sheet, a book or board, cloth, a hand, a full lens cover |
| Position and coverage | left, right, top, bottom, centre; about 10, 25, 50, 100% of the view |
| Distance from the lens | 10 cm, 50 cm, 1–2 m |
| Lighting | bright room, dim room, daylight by a window |
| Must NOT alarm | empty room 2 min; a person walking past; lights switched on and off; a close plain wall in the normal view; camera auto-exposure changes |
| With flame | candle visible next to a partial obstruction (both classes must work); candle hidden behind the obstruction |

**Run** `live_flame.py --headless --source <clip>` on each clip. From `frames.csv`, report:
- obstruction detected (yes or no) and the delay after the object stops moving
- blind-percentage error against a hand-marked box
- false obstruction frames in the must-not-alarm clips
- flame recall in the flame clips, compared with V12 on the same clips

An obstruction alarm should use the same idea as the Week 2 flame alarm: obstruction in at least X% of frames over Y s. X and Y are set on half the clips and reported on the other half.
