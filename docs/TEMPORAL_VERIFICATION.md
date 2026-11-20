# Temporal and appearance-based alarm verification (live viewer)

**Status (3 November 2026):** implemented, unit-tested, and tuned on six live webcam sessions; results are under
"Measured so far". Temporal filtering is the core of it: a box alarms only when its track is matched now and in
4 of the last 6 frames. Flicker, colour and box statistics are then measured over that window. The thresholds
were chosen on the same sessions they are reported on, so **no held-out validation exists yet**, and no
false-alarm rate is claimed. The YOLOv9 ONNX model, preprocessing, output decoding and NMS are unchanged.

## Layers

`scripts/live_flame.py` always draws and logs every raw detector box exactly as before. Raw boxes stay thin
red. Verified alarms get a thick white/magenta double frame and a magenta status line.

| Option | Decision | Notes |
|---|---|---|
| *(defaults)* | none | Original behaviour; the new CSV columns are empty |
| `--persistence K N` | Legacy: a current box overlaps (IoU >= 0.1) any box in K of the last N frames | Unchanged; kept for comparison |
| `--verification persistence` | Tracked K-of-N: track matched **now** and in `--min-hits` of the last `--window` frames | |
| `--verification diagnostic` | Same alarm as `persistence`; all features and rule results are computed, shown and logged | Rules **never** reject in this mode |
| `--verification fused` | Tracked persistence **AND** every given rule | Needs at least one rule threshold |

`--persistence` and `--verification` can run together, so one recording compares them on identical detections.

### Tracker (`scripts/flame_verification.py`)

- Greedy one-to-one IoU association, highest IoU first. A detection continues at most one track, and a track
  takes at most one detection per frame. Association is chained to each track's last box.
- **Distance fallback** (`--track-distance`, default 3.0; 0 means IoU only): this runs only for tracks and
  detections that are still unmatched after IoU. A detection continues a track when its centre moved at most
  3 × the track box's longer side and the two areas are within 4×, nearest first. A waved match moves 20–40 px
  per processed frame with a box about 20 px wide. Its IoU is 0, so IoU-only tracking started a new track
  every frame and never confirmed it.
- Unmatched detections start new tracks. IDs are never reused.
- A missed frame is recorded as a miss, never a hit. A track survives `--max-missed` consecutive misses
  (default `window − min_hits`, at least 1), keeping its identity. Past that it can't reach K hits without
  fresh matches, so it's dropped.
- **A missed or held track is never persistent or fused.** Only tracks matched in the current frame can
  raise an alarm.
- Histories are bounded to the window. Statistics use only observations from the last N processed frames.

### Measurements per track (window statistics)

Each matched box's region of the **undrawn** BGR frame is resized to 32 × 32. Empty, zero-size,
non-finite or off-frame boxes give no appearance features; tracking still works.

| Group | Statistics |
|---|---|
| Confidence | `score_mean`, `score_min` |
| Brightness and flicker | `grey_mean`; `grey_var_mean` (spatial variance); `flicker` = mean absolute grey change between consecutive matched ROIs (0–1); `intensity_cv` = temporal coefficient of variation of the mean grey level |
| Box dynamics | `area_mean`, `aspect_mean`, `area_cv`, `aspect_cv`; `motion_px` = mean centre step in pixels; `motion_relative` = the same divided by √area; `displacement_px` = first-to-last centre distance in the window |
| Colour | `hue_mean` and `hue_std` (circular over time, so red wraps at 0/180); `saturation_mean`/`std` and `value_mean`/`std` over time; `bright_fraction` = share of pixels with V >= 220 |

Frequency analysis isn't implemented. Live processing runs at about 6 FPS on this laptop, so changes faster
than about 3 Hz are aliased, and a spectral measure couldn't be justified at that rate.

### Rules (decide only in `fused` mode)

| Option | Passes when |
|---|---|
| `--min-score X` | `score_mean >= X` |
| `--min-flicker X` | `flicker >= X` |
| `--min-intensity-cv X` | `intensity_cv >= X` |
| `--min-area-cv X` | `area_cv >= X` |
| `--min-aspect-cv X` | `aspect_cv >= X` |
| `--max-motion-relative X` | `motion_relative <= X` |
| `--min-bright-fraction X` | `bright_fraction >= X` (luminance, not hue) |
| `--max-saturation X` | `saturation_mean <= X` (0–255 OpenCV HSV saturation, mean over the window) |

Rules are combined with logical AND, with no weights. A rule whose statistic can't be measured yet **fails**.
There's no hue rule. `--max-saturation` is the only colour statistic used, and it's off by default. It was
added because three live sessions showed it: small match flames have a near-white core (ROI saturation 47–97),
while the orange sock, jar lid and cap were 94–153. It can reject a real fire that looks deeply saturated
orange (large or distant fires, or fires behind smoke or glass). That hasn't been tested. No single feature is proof of fire: a real flame can look steady,
and a lamp, LED, screen or reflection can flicker. Every threshold must be chosen on labelled tuning clips.

`--track-iou` (default 0.1) is the minimum IoU for continuing a track. `--track-distance` is described above.

## Logging

- **`frames.csv`:** the original columns are unchanged, followed by `persistence_support`, `alarm` (legacy
  `--persistence`) and:
  - `verification_mode`, `verify_ms`, `current_tracks`;
  - `alarm_track_id`, `alarm_track_support`, `alarm_track_persistent`, `alarm_track_score_mean`,
    `alarm_track_flicker`, `alarm_track_area_cv`, `alarm_track_aspect_cv`, `alarm_track_motion_px`;
  - `persistent_alarm`, `fused_alarm`, `verified_alarm`.
  The `alarm_track_*` values describe the deciding track (or the strongest current track). `verified_alarm`
  is the decision of the selected mode. All of these are empty when verification is off.
- **`tracks.csv`:** one row per live track per frame: `matched`, `misses`, `box_xyxy`, `persistent`, `fused`,
  `checks` (JSON pass/fail per rule) and every statistic in its own column.
- **`session.json`:** a `verification` block with the mode, min_hits, window, track_iou, max_missed, rules,
  whether the rules decide, and the window unit.

## Evaluation

1. **Record clips** with a fixed camera, one scenario each: real flame, orange/red fabric, stationary lamp or
   LED, moving light or reflection, empty room, and smoke without flame if available. Note manually verified
   flame intervals in seconds. Keep these clips out of any training data.
2. **Tune and report on separate clips.** Choose thresholds on tuning clips by reading `tracks.csv` (flame
   tracks against nuisance tracks), and enable a rule only where the two clearly separate. Report on
   different clips.
3. **Copy and edit** `configs/verification_eval_example.json`, then run:

```bash
.venv-camera/bin/python scripts/evaluate_clips.py --manifest configs/my_eval.json
```

Each clip is processed once, so `raw`, `legacy_persistence`, `tracker_persistence` and `fused` are judged on
identical detections. The report covers:
- frame-level precision and recall;
- alarm events, merged across gaps of at most `event_merge_gap_seconds`, so repeated boxes aren't separate
  alarms;
- genuine and false-alarm events, missed flame intervals and event recall;
- detection delay (raw) and verification delay;
- inference and verification time per clip.

Counts only: no false-alarm rate is derived.

## Measured so far

- **Unit tests:**
  - `tests/test_flame_verification.py` (23 tests) covers consecutive frames, temporary disappearance and
    return, empty frames, expiry, two simultaneous objects, several detections competing for one track,
    movement across the frame, a different object at the same place, K-hit confirmation, no alarms from
    stale tracks, invalid ROIs, uniform-colour ROIs, the saturation rule and the distance fallback for fast
    small objects.
  - `tests/test_evaluate_clips.py` (4 tests) covers event merging and the frame/event metrics.
- **Backward compatibility** (headless, same clips): with the defaults, the detection columns match the
  committed viewer frame for frame and the original columns come first, unchanged. `--persistence` output is
  identical to the previous implementation. All modes produce identical detections.
- **Overhead:** verification adds about 0.8 ms median, 1.3 ms p95 per frame with a track present, and about
  0.03 ms with none. YOLOv9 inference plus decoding took 100–170 ms on the same laptop. These are not AM62A
  numbers.
- **First live tuning session** (`results/webcam/20261103_131310_938344`, 3 Nov 2026, laptop webcam, about 6–9
  processed FPS, `--verification diagnostic`; labels were assigned afterwards from the screenshots):
  - **Nuisance tracks** (13:13:11–13:13:51): an orange jar lid, an orange sock and a small orange backlit cap,
    giving 59 persistent track-frames. **Flame tracks** (13:15:39–13:15:54): hand-held lit matches, giving 67.
    T21 at 13:15:35 wasn't labelled.
  - The replay below counts persistent track-frames that pass each rule:

    | Rule | Flame kept | Nuisance kept | Nuisance tracks left |
    |---|---:|---:|---|
    | persistence only | 67/67 | 59/59 | jar lid, sock, cap |
    | `--min-flicker 0.05` | 63/67 | 12/59 | 4 tracks |
    | `--min-flicker 0.07` | 60/67 | 2/59 | sock as it entered the frame (T11) |
    | `saturation_mean <= 80` (not implemented) | 14/67 | 9/59 | |

  - All three flame alarm periods still alarmed at `--min-flicker 0.07`, with no added delay on this session.
  - A maximum-saturation rule would have rejected most of the real match flame (median saturation 90), so it
    wasn't added.
  - These thresholds come from **one tuning session**: one person, one room, one webcam, a hand-held moving
    match. Hand motion and box jitter add to the flicker statistic, so it isn't proof of fire. Confirm on a
    separate recording before relying on it. No false-alarm rate is derived.
- **Second live session, fused mode** (`results/webcam/20261103_131830_965275`, `--verification fused
  --min-flicker 0.07`). The sock was waved, the jar lid held up, then lit matches shown; labels were again
  assigned from the screenshots.
  - With `--min-flicker 0.07` set beforehand, so this part is held out:
    - **Jar lid:** persistence alone raised 2 alarm events (45 frames); fused raised 0.
    - **Waved sock:** persistence raised 4 events (50 frames); fused still raised 3 short events (8 frames).
      Shaking the sock raised its flicker (0.07–0.13).
    - **Match flame:** all 5 events kept (37 of 42 persistent frames). The first fused alarm came 0.5 s after
      the first raw detection.
  - The sock frames that passed had low box-area change (`area_cv` 0.02–0.10, median 0.03). The flame frames
    over both sessions had p5 0.075 and median 0.20. A replay of `--min-flicker 0.07 --min-area-cv 0.05` on
    this session gives 37/42 flame frames and 5/5 flame events against 2/95 nuisance frames and 1 event, and
    on the first session it gives the same result as flicker alone. Any `area_cv` value from 0.04 to 0.06
    gives the same counts.
  - **This session tuned `--min-area-cv`, so it no longer counts as held-out for that rule.** Validate the
    combination on a new recording.
- **Third live session, fused mode** (`results/webcam/20261103_132057_917250`, `--min-flicker 0.07
  --min-area-cv 0.05`): the jar lid/cap, a hard-waved sock, then lit matches, some held steady.
  - These rules were **held out** for this session, and they failed:
    - **Jar lid/cap:** 0 fused frames.
    - **Waved sock:** 34 fused frames, because its flicker (median 0.062) and area_cv (0.050) reached flame
      levels.
    - **Steady match (T13):** flicker 0.065, so it didn't alarm from 60.3 s to 64.7 s. Overall only 40% of
      persistent flame frames alarmed.
  - Flicker and area change mostly measure motion, so a moving cloth and a steady flame overlap.
- **Replay over all three sessions** (persistent track-frames; fused = persistent AND rules, so the replay is
  exact):

  | Rules | Flame frames alarmed s1 / s2 / s3 | Nuisance frames s1 / s2 / s3 |
  |---|---|---|
  | persistence only | 100 / 100 / 100 % | 59/59, 95/95, 181/181 |
  | `--min-flicker 0.07 --min-area-cv 0.05` | 90 / 88 / 40 % | 2, 2, 34 |
  | `--min-flicker 0.05` | 94 / 98 / 97 % | 12, 18, 74 |
  | `--max-saturation 100` | 100 / 100 / 100 % | 10, 24, 1 |
  | **`--max-saturation 100 --min-flicker 0.04`** | **100 / 100 / 100 %** | **6, 0, 0** |

  The 6 remaining frames in s1 are the small backlit cap (T19, saturation 57).
- **All three sessions were used to choose `--max-saturation 100 --min-flicker 0.04`, so none is held out.**
  The tests so far cover one person, one room, one webcam and small match flames only. Validate on a new
  recording that includes a larger flame (candle, or paper in a metal tray) before relying on it.
- **Fourth live session, held out** (`results/webcam/20261103_132502_876903`, `--verification fused
  --max-saturation 100 --min-flicker 0.04`, 131 s):
  - **Match flames** (T13, T15, T16, 36–57 s): the first fused alarm came in the same frame as persistence
    (about 0.5 s after the first box), and 95% of persistent flame frames alarmed.
  - **Orange sock** (T3, T7, T20–T22): 0 fused frames. It had saturation 128–147.
  - **Unidentified large reddish object** (T2, 6.5–12.7 s, about 210 × 350 px, hue 5): 23 fused frames in
    about 3 short alarm events. Its saturation was 97–101, right at the cut.
  - Its spatial grey variance (max 2182) was below every flame frame's p5 (2406), but adding
    `grey_var_mean >= 2200` would fit this one object and costs 10% of run-2 flame frames, so it wasn't
    added.
- **Fifth live session** (`results/webcam/20261103_132909_001199`, same rules, IoU-only tracker):
  - **Sock:** 0 fused frames.
  - **Matches held fairly still** (27–41 s): they alarmed at the first persistent frame.
  - **Match waved side to side** (41–52 s): it was detected in every frame but split into 46 one- or
    two-frame tracks, so it raised almost no alarm. That's a tracker failure, not a rule failure.
- **Tracker replay** from the logged detections, persistence only. Fused can't be replayed, because ROI
  images aren't stored. IoU-only reproduces the logged persistent frames exactly.

  | Session | Persistent frames, IoU only → with `--track-distance 3` | Tracks |
  |---|---|---|
  | 131310 | 121 → 136 | 34 → 24 |
  | 131830 | 137 → 165 | 29 → 13 |
  | 132057 | 256 → 262 | 18 → 14 |
  | 132502 | 239 → 245 | 23 → 20 |
  | 132909 | 129 → 197 | 66 → 16 |

  All added persistent frames fall inside match-flame periods, except 1 frame in 131310 at 23 s. The rules
  still decide fused on those frames.
- **Sixth live session** (`results/webcam/20261103_133222_753220`, `--max-saturation 100 --min-flicker 0.04`,
  `--track-distance 3`), held out for the tracker change:
  - **Matches, including waved** (13.5–30.7 s, 51–55 s): one track each, fused from the first persistent frame.
  - **Sock:** 0 fused.
  - **Phone torch** (T15/T16, 65–80 s, plus probably T12 at 59 s): 6 short fused events of 0.3–1.3 s. White
    light has low saturation (median 47), and its flicker (median 0.038) sits at the cut.
- **Not fixed, and why:** replays over all six sessions found no rule that removes the torch without losing
  real flame frames:
  - **Cap on bright fraction** (`<= 0.6`): 75% coverage in session 6.
  - **Cap on mean brightness:** 24–62% coverage.
  - **`--min-flicker 0.05`:** torch 28 → 17 frames, coverage 88–100%.
  - **Requiring a track to stay fused for 8 of 12 frames:** misses several short real flame events and adds
    0.8–2.5 s of delay.
  - **The phone torch remains a known false-alarm source.** The detector scores it 0.53–0.81. The proper fix
    is hard-negative training data (torches, LEDs, lamps), not more post-filters.
- **False negatives, all six live sessions.** Flame periods are the frames where a screenshot-identified
  flame track was matched, merged across gaps of at most 1 s. That makes this a **lower bound on detector
  misses**: a flame the detector never boxed for over 1 s, or never boxed at all, isn't counted. The sessions
  have 636 flame-period frames in total.

  | Layer | Missed frames (as logged) | Note |
  |---|---:|---|
  | No detector box | 23 | Detector gaps inside a flame period |
  | Box, but track not confirmed | 169 | Tracker split (sessions 1–5 ran IoU-only) plus the 4-of-6 warm-up |
  | Confirmed, rules rejected | 5 | `--max-saturation 100 --min-flicker 0.04`, replayed |
  | **Alarmed** | **439 (69%)** | Session 5: 40%, the waved match |

  A tracker-only replay (persistence, no rules) gives **63% IoU-only and 89% with `--track-distance 3`**. Of
  the 70 frames still unconfirmed, 31 fall in the first 0.6 s of a flame period (warm-up). The rest follow
  detector gaps.
- **Not measured:** any held-out accuracy, false-alarm or delay figure for the recommended rules, and AM62A frame
  rate.

## Limitations and unverified assumptions

- Flicker is undersampled live (about 6 FPS here). Clips are replayed at their recorded rate, so live and
  replayed statistics differ. Measure the AM62A frame rate, then choose window and thresholds for it.
- Windows count processed frames, not seconds.
- ROI change also comes from hand shake, box jitter, auto-exposure, moving cloth, LEDs, screens and
  reflections.
- Association uses IoU, then centre distance. Within `max_missed` frames, a different object of similar size
  appearing in the same place, or within 3 box lengths, continues the old track. Fast motion or camera pans split tracks. Overlapping flames can merge.
- The alarm can drop during detector gaps, because held tracks never alarm. The evaluation merges events
  across gaps, but a production alarm would need explicit latching and reset logic, which isn't implemented.
- There's no camera-fault, tamper or obstruction detection.
- Results on a few clips don't establish field accuracy.

## Backups and migration

- `~/Desktop/flame_backups/live_flame_persistence_20261103.py`: the viewer with `--persistence` only.
- `~/Desktop/flame_backups/20261103_before_verification_modes/`: the previous, uncommitted `--verify` version
  and its diff.

That `--verify K N` option was never committed. It's replaced by `--verification persistence|diagnostic|fused
--min-hits K --window N`. Its old `--diagnostics` flag is now `--verification diagnostic`.
