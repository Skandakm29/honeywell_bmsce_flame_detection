# Alarm logic for the board (runs on the A53 after decode + NMS)

The numbers in `<…>` come from `rule.json`, written by `temporal_final.py`. All timing uses timestamps in **seconds**, never frame counts, so the same rule works at any frame rate of 5 fps or more.

## Per frame
1. **Gate** each detection after NMS. It counts if:
   - score >= 0.40, **or**
   - its longest side at the 640 input is < 16 px **and** score >= `<small_box.threshold>` (0.25 if it adds no false alarms; if it is null, use 0.40 for every box).
2. **Track by location.** Match each counted detection, highest score first, to an existing track if IoU >= 0.1 **or** the centre distance is <= 1.5 × the longer box side. Otherwise start a new track. Drop a track after 1.0 s with no detection.
3. **For each track over the last `<window_s>` seconds:**
   - `hit_ratio` = frames where the track had a counted detection / max(frames processed in the window, window_s × fps)
   - `mean_score` = mean score of those detections
4. `flag` = some track has `hit_ratio >= <min_hit_ratio>` and `mean_score >= <min_mean_score>`.

## Alarm state
- **OFF → ON:** `flag` has been true continuously for `<alarm_on_after_s>` s.
- **ON → OFF:** no `flag` for 3.0 s.
- Log each alarm with its timestamp and the track's box, for evidence capture (week 6).

## Why this is robust
- It is tuned on varied real fires and hours of real no-fire video, at 10 and 5 fps.
- It uses the worst case over neighbouring settings, so small changes to the numbers do not break it.
- It requires a fire to alarm within 30 s, the EN 54-10 limit.
- It does not use flicker, which would need about 20 fps or more.

## Cost
A few small loops per frame. It has no learned model.
