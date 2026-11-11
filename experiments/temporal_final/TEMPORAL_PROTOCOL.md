# Temporal alarm rule for V10 + small-box threshold: protocol (fixed before running)

**Goal:** turn V10's per-frame detections into a stable fire **alarm** that holds up in any scene and frame rate, **without increasing false alarms**. It is tuned once and reported once on held-out video.

## Owner decisions (7 Nov)
- **Week-2 model:** V10 (frozen) + small-box threshold. No new training here.
- **Option A, an explicit waiver of the "ONFIRE is test-only" rule, limited to tuning this alarm rule.**
  - ONFIRE is split 50/50 by `md5('onfire-split-v1' + file name) % 2`.
  - Half is used for tuning; the other half is reported once and never used to re-tune.
  - Both file lists are written to `temporal_final/onfire_split.json` before any tuning. A rerun refuses a different split.
  - Model and threshold selection are unaffected.
- **Comparison:** the same rule tuned on our own training clips only (option B) is shown side by side, so the effect of the split is visible.
- **Far-flame clips (corridor 4 m and 10 m)** are reported per clip. The rule cannot create detections; if they stay missed, the report says so.

## Background
- **EN 54-10:** a flame detector must respond within **30 s** to a reference fire. That is our hard limit for time to alarm.
- **Video fire detection literature:** temporal smoothing and tracking cut false alarms, at the cost of about 2 s of delay.
- **Flicker (about 1–10 Hz)** needs about 20 fps or more, so it is **not used**.

## Rule (structure fixed)
1. **Gate.** A detection counts if its score is >= 0.40, or if its longest side at the 640 input is < 16 px **and** its score is >= the small-box threshold.
2. **Track by location.** A detection joins a track if IoU >= 0.1 or its centre is within 1.5 × the longer box side. A track ends after 1.0 s unseen.
3. **Over the last W s:** a track's `hit_ratio` (frames seen / frames processed, with a part-filled window never counting as 100%) and `mean_score`.
4. **Alarm on** when some track has `hit_ratio >= h` and `mean_score >= m`, continuously for `hold` s. **Alarm off** after 3 s with none.
5. **False alarm:** an alarm rising edge outside a fire interval, with a 30 s refractory period.

**Grid (144 rules, 4 numbers):**
- W in {1, 2, 3} s
- h in {0.3, 0.5, 0.7, 0.9}
- m in {0, 0.45, 0.55, 0.65}
- hold in {0, 1, 2} s

## Small boxes must not add false alarms
- **Small-box thresholds tried in order:** 0.25, 0.30, 0.35, then none (0.40 for all).
- For each, the rule is tuned, and its **worst-case false alarms per hour** on the tuning no-fire video (at 10 and 5 fps) is compared with the same rule **without** the small-box gate.
- **The lowest threshold that adds no false alarms is kept.** So a small LED or lamp that scores 0.25–0.40 cannot raise the false-alarm rate.
- Small boxes must also persist as a tracked flame for the rule's window before any alarm.

## Choosing the rule (robust)
- **Scoring:** each rule is scored by its **worst case** over 10 fps and 5 fps (every other logged frame) and over its grid neighbours, one step on each axis.
- **Must hold:**
  - worst-case event recall >= 90% on the tuning fires
  - every detected fire alarms within 30 s
- **Then, in order:**
  1. fewest worst-case false alarms per hour
  2. highest worst-case alarm coverage (share of fire time with the alarm on)
  3. lowest median time to alarm
  4. **simpler rule on a tie** (no score condition, then no hold time, then the longer window)
- **If no rule holds:** highest worst-case recall, then fewest false alarms.

## Reported once (never tuned)
- **Held-out sets:**
  - held-out own clips (flame, screen, lamps, orange object, torch, vehicle)
  - `onfire_test`
  - **extra clips:** corridor diya 1 / 4 / 10 m (fire) and the owner's **LED candle** (fake flame, must not alarm)
- **Columns:**
  - `1_no_temporal_.40` (baseline)
  - `2_no_temporal_gate`
  - `3_old_v10_rule` (if its file exists)
  - `4_final_rule`
- Results at 10 and 5 fps, **time to alarm per clip**, and **false alarms per hour by nuisance type**.
- **V12 with the same locked rule** (no gate), for a fair comparison.

**Outputs:** `temporal_final/rule.json` (the locked rule), `results.json`, `REPORT_TEMPORAL.md`, `onfire_split.json`.

## Bug fix, run 2 (7 Nov, after run 1)
- **Found in run 1:** some ONFIRE fire and no-fire videos share a file name in different folders.
  - V10's `bench.py` saved each video's detections under the file name only, so one log overwrote the other.
  - Those entries were scored on the wrong video: invented false alarms and corrupted tuning (7 names held out, 5 in tuning).
- **Fix:** every name that maps to more than one video is **dropped** from tuning and test.
  - All other videos keep their frozen run-1 split. The code checks it is a subset and lists the dropped names in `onfire_split_v2.json`.
  - Nothing else changes (grid, thresholds, selection order).
- **Disclosure:** run 2 re-tunes on corrected data after run 1 was seen. That is a correction of broken input, not a change of rule.
- The same collision also slightly affected the earlier V10/V11/V12 ONFIRE video numbers (`temporal.py`, `bench.json`).

## Limits
- Few own clips; ONFIRE is CCTV-style footage.
- The board frame rate is not measured yet (5 and 10 fps cover the likely range).
- The 6 Nov recordings other than the corridor clips were used to train V12, so they are not test data for V12. They never touched V10.

Run 3 (7 Nov): 12 ONFIRE videos are listed twice in videos.json (same file, fire and no-fire). Every name listed more than once is dropped from tuning and test; all other videos keep the frozen run-1 split (subset check). Correction of broken input, not a rule change. Earlier V10/V11/V12 ONFIRE video numbers counted these 12 twice.
