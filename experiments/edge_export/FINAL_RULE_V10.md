# Final week-2 model: V10 + small-box threshold + alarm rule

**The model file does not change.** Use the V10 ONNX you already have (`flame_v10_raw_640.onnx`). Everything below runs on the A53 **after** decode + NMS.

## 1. Small-box threshold (per detection, after NMS)
A detection counts as a flame if:
- score >= **0.40**, or
- its longest side is **< 16 px at the 640 input** and score >= **0.25**

(At 416 the same 16 px rule applies to the 416 input, but this has not been measured.)

## 2. Two-level output
- **WARNING:** any frame with a counted detection.
- **ALARM:** a time-based confirmation rule (below).

## 3. Alarm rule (time based, works at 5–10 fps)
1. **Track by location.** Match each counted detection, highest score first, to an existing track if IoU >= 0.1 or the centre distance is <= 1.5 × the longer box side. Otherwise start a new track. Drop a track after **1.0 s** unseen.
2. **Window.** For each track over the last **2.0 s**:
   `hit_ratio` = frames with a counted detection / max(frames processed in the window, 2.0 × fps)
3. **Alarm ON** when some track has `hit_ratio >= 0.30`. No extra hold time and no minimum score.
4. **Alarm OFF** after **3.0 s** with no qualifying track.

## Measured (held-out video, 10 fps, lab PC)
- **WARNING:** caught all 12 held-out fire videos; median time about 5 s on ONFIRE.
- **ALARM:** about 40% fewer false alarms than WARNING (ONFIRE no-fire video: 3.3 → 2.0 per hour), but it missed 2 of 6 ONFIRE fires that WARNING caught.
- Results at 5 fps were close to 10 fps.

**Known limits:**
- Flames only a few pixels in size (about 4 m or more for a candle) are missed.
- Screens showing fire, orange objects and flame-shaped LEDs can still trigger.

Machine-readable copy: `rule.json`.
