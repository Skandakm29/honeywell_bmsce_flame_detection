# Week 4: obstruction detection, blind-zone mapping and occlusion monitoring

**Last updated:** 12 Nov 2026

## Requirement (mentor, 9 Nov; `WEEK2_SUMMARY.md` section 12)
When the camera's view is partly blocked by an added object (a steel sheet, a box…), the system must detect the obstruction and report how much of the view is blind.

Week 4 has two independent parts. They can run together:

| Part | What it detects | Needs training? | State |
|---|---|---|---|
| **A. `obstruction` class in the YOLOv9 model** | objects **close to the lens** that block part of the whole view | yes (lab GPU) | code ready, not trained |
| **B. Zone monitor** (`src/obstruction/`) | people, vehicles or objects blocking **configured zones** in the scene, and for how long; blind-zone map | no | implemented, checked on generated clips; real-scene tests pending |

---

## Part A: `obstruction` class in the flame model

**Decision (week 4):** add a class `obstruction` to the **same** YOLOv9-S-TI-Lite flame model. One model and one pass per frame on the AM62A. There is no second model and no extra pass.

### How it works

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

### Status

| Item | State |
|---|---|
| Model growth, data, metrics, selection, export, viewer | Done; unit tests in `tests/test_obstruction_class.py`; CPU smoke run of every step passed |
| Training on the lab GPU | **Not run yet** (`experiments/obstruction_w4/START_HERE.md`) |
| Real-camera test | **Not run yet**: needs the trained model and the recordings below |
| Board (raw-head export for 2 classes, int8, AM62A speed) | Not done |

### Real-camera test (after training; reported once)
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

---

## Part B: zone monitor (blind-zone mapping and occlusion monitoring)

**Status:** implemented and unit-tested. Checked end to end on generated clips. Real-camera checks so far cover
frame rate, latency and logging only. **No real-scene accuracy has been measured yet** (see "Measured so far").

### What it does
It watches **configured zones** (polygons on the image) and reports when people, vehicles or objects block them,
and for how long. For each zone it reports:
- how much is blocked **now** (any duration, including someone walking through);
- how much is **persistently** blocked (each cell blocked at least `persist_s` seconds);
- a **zone alarm**: ON when the persistent share reaches `min_blocked_pct`, OFF after it stays below for
  `clear_s`;
- a **blind-zone map** per session: how long each part of the image was blocked.

It needs **no model and no training**. It compares each frame with a learned picture of the empty scene, so it
reacts to anything that stays changed: people, vehicles, boxes, sheets. It doesn't say *what* is blocking.
It does not handle objects right at the lens; that is Part A.

### How it fits with flame detection
```
camera frame ──► YOLOv9 flame detection ──► flame alarm (unchanged)
      │                      │ flame boxes (excluded from obstruction, plus a 50% glow margin)
      └──► zone monitor (src/obstruction) ──► zone %, zone alarm, zone_events.csv, blind-zone map
```
- `scripts/live_flame.py --zones <config>` runs both on the same frames. Without `--zones`, every output is
  identical to before (checked frame by frame).
- **A flame never counts as an obstruction.** Flame boxes, grown by half their size for glow, are excluded and
  never learned into the empty-scene reference.
- During a zone alarm the screen shows *"OBSTRUCTED - flame detection blind here"*, because a fire behind the
  obstruction can't be seen.

### Method (`src/obstruction/`)
| Module | Role |
|---|---|
| `zones.py` | Zone config: polygons in normalised 0..1 coordinates, per-zone `min_blocked_pct`, `persist_s`, `clear_s`, optional ignore polygons |
| `detector.py` | Learns the empty scene from `learn_s` seconds of **still** view; learning restarts if anything moves. Then it marks pixels whose grey level or edge strength differs from the reference. Whole-frame brightness changes are corrected (median gain). The reference adapts very slowly (time constant 600 s), and only where nothing is blocked, so a stationary obstruction is never learned away |
| `grid.py` | 32 × 24 grid. A cell is blocked when at least 50% of it changed. Zone % uses the union of all changes, so overlapping objects count once. Per-cell timers in seconds tolerate detection gaps up to 1.5 s |
| `confirm.py` | Zone alarm state machine with hysteresis; events with start, end, peak and mean |
| `blindmap.py` | Session map: seconds blocked and obstructed per cell; summary per zone |
| `monitor.py` | Ties it together; flame exclusion; `relearn()` |
| `overlay.py` | Zone outlines, yellow cells = blocked now, red cells = persistent |
| `ground.py` | Floor area interface. It returns **None** until Week 3 calibration exists, so no m² is claimed |

All timing is in **seconds**, so the same settings behave the same at any frame rate. The tests check this at 5
and 10 FPS.

### How to use it
1. **Draw zones** on a frame from the camera you will use:
   `.venv-camera/bin/python scripts/draw_zones.py --source 0 --output configs/zones/my_site.json`
   (left click to add points, Enter to close a zone, I for an ignore area, S to save).
2. **Run.** Keep the zones empty until the status says `ready`:
   - obstruction only: `.venv-camera/bin/python scripts/test_zone_monitor.py --zones configs/zones/my_site.json`
     (keys: R re-learns the empty scene; A–G write test markers);
   - with flame detection: `.venv-camera/bin/python scripts/live_flame.py --model <flame.onnx> --source 0
     --zones configs/zones/my_site.json`.
3. **Evaluate recorded clips.** Copy `configs/zone_eval_example.json`, enter the real obstruction times, then run
   `.venv-camera/bin/python scripts/evaluate_zone_clips.py --manifest <file>`.
   The report covers alarm events, false alarms, missed obstructions, delay and end-time error.

Zone configs: `configs/zones/webcam_test.json` (alarm at 30% of the zone blocked for 10 s, clear after 5 s) and
`short_clip_test.json` (2 s / 1 s, for short generated clips only). Both are **starting values, not tuned**.

### Outputs (per session folder)
| File | Content |
|---|---|
| `frames.csv` | Per frame: state (`learning`/`ready`), monitor ms, foreground %, gain, and per zone `blocked_pct`, `persistent_pct`, `alarm`. In `live_flame.py` these columns are prefixed `zone_` and follow the unchanged flame columns |
| `events.csv` / `zone_events.csv` | One row per zone alarm: start, end, duration, peak and mean persistent % |
| `fire_hidden_events.csv` | `live_flame.py --zones` only: one row per FIRE HIDDEN spell (track, start, end, reason, last box). Per-frame columns `zone_unknown_pct`, `zone_fire_hidden`, `zone_fire_hidden_unknown`, `zone_view_changed` follow the zone columns |
| `blind_map.json` / `.png` | Session blind-zone map, per-cell seconds and a per-zone summary |
| `session.json` | Zones, grid, detector settings, versions, start and end, latency, status (`finished`/`error`) |

A camera or monitor failure ends the run with `status: error`. It is never logged as a clear scene.

### Measured so far
- **Unit tests:** `tests/test_obstruction_zones.py`, part of the full suite (85/85 pass, including `test_fire_hidden.py` and `test_deep_change.py`). They cover:
  - config validation; overlap without double counting; ignore areas;
  - gap tolerance; hysteresis; frame-rate independence;
  - walking past never alarms; an object left in the zone alarms and clears with an accurate end time;
  - a 30 s stationary object is not learned away; lighting gain; learning restarts on motion;
  - a flame inside the zone is not an obstruction; the blind-map accounting; re-learning.
- **Synthetic 60 s clip** (generated textured scene):
  - walking: no alarm;
  - box placed at 11 s: alarm at 21.0 s, logged end 31.0 s (removed at 31 s);
  - two overlapping boxes: 75%, the union (the sum would be about 93%).
- **Generated (AI) clips, 12 Nov** (original YOLOv9 model, laptop CPU; results in
  `results/obstruction_eval/gemini_20261112/`). These are pipeline checks only, not accuracy results:
  - `week_4_test1` (forklift): the worker outside the zone gave 0%; the forklift and pallet in the zone gave
    an alarm at 8.08 s, peak 64%.
  - `week_4_test` (fire only): flame detected from 3.5 s; zone stayed at 0% with the flame excluded (11% without the
    exclusion, still below the 25% alarm level).
  - `week_4_test_3`, 11.6–30 s (fire, then a forklift passes in front of it), plain flame detection with no
    verification rules (`week_4_test_3_plain_collage.jpg`, `week_4_test_3_seg3_plain.mp4`):

    | Clip time | Flame | Zone |
    |---|---|---|
    | 20.1 s | fire detected on its first frame | clear |
    | 20.1–24.9 s | detected continuously | forklift drives in |
    | 24.9–26.8 s | mostly lost (0.6 s of brief hits): the forklift is in front | up to 48% blocked |
    | 26.75–27.4 s | partly seen | **OBSTRUCTED - flame detection blind here** |
    | 26.8–30 s | detected again | clear |

  - `week_4_test_3`, full clip: the first fire (5–10 s) was detected throughout. **Steam raised a false zone alarm**
    (4.7–11.3 s). The scene cut at 10 s changed 75% of the frame, and after it about 21% of the zone stayed
    "blocked" because the reference had been learned in the other scene.
  - **Flame rule finding:** with `--verification fused --max-saturation 100`, this large deep-orange fire
    (saturation 105–128) raised the fused alarm on only 7 frames, against 189 frames for persistence alone. The
    saturation rule, tuned on small match flames, rejects big orange fires, so it should not be used. Plain
    detection or persistence (4 of 6) caught the fire.
- **Real webcam, laptop CPU** (12 Nov): 640 × 480 at about 30 FPS. Zone monitor 8.4–9.8 ms median and 10–12 ms
  p95 per frame; inside `live_flame.py`, about 6 ms. These are not AM62A numbers.
- **Real-scene test A** (empty room, 6.5 min): no false alarm. But the zone read a constant 21% blocked, because
  something in view during learning had left. The test is to be repeated with the zone empty during learning. The
  on-screen check after learning was added because of this.
- **Not measured:** accuracy on real scenes (tests B–G), shadows and auto-exposure, real fire glow on walls, AM62A
  speed, the V10 model.

### Upgrade 17 Nov: fire seen, now hidden; flame at start-up; view changed
Added to `src/obstruction/` (`hidden.py`, `detector.py`, `monitor.py`). Flame detection, flame boxes and the flame
alarm are not changed: these are extra outputs only.
- **Fire seen, now hidden** (`HiddenFireTracker`):
  - each flame box is tracked;
  - a track is confirmed after 5 hits over at least 0.5 s;
  - if a confirmed fire is lost for 0.5 s and at least 30% of the known area around its last box is foreground,
    the output is **FIRE HIDDEN**;
  - it stays flagged for as long as that holds, with no time limit, and ends when the fire is seen again or the
    area has been clear for 2 s (the fire went out);
  - boxes that split and merge are one fire, so no ghost tracks are left;
  - a small flame that vanishes in or right next to a fire that is still visible is not hidden (a lick of flame
    above the fire in test 3 had caused a 0.5 s false FIRE HIDDEN with the DINOv2 detector).
- **Flame at start-up:**
  - pixels inside flame boxes in at least 20% of the learning frames are **unknown**, not learned as normal;
  - a change there counts only while the known ring around them is also blocked: a forklift in front counts, the
    fire simply going out does not;
  - the area is learned once it has been flame-free, still and surrounded by background for 5 s, so an object
    parked in front is never learned.
- **View changed:** if at least 60% of the frame changes at once (camera moved, scene cut), the output is VIEW
  CHANGED, and lost fire tracks are dropped instead of being called hidden. Re-learning and tamper are Week 5.
- **Unit tests:** `tests/test_fire_hidden.py` (11 tests). With the start-up fix switched off,
  the "fire goes out after start-up" test shows the old false block (about 14% of the frame) and a false FIRE
  HIDDEN. With the fix, both are 0.
- **Generated clips, 17 Nov** (same settings as 12 Nov, original YOLOv9 model; results in
  `results/obstruction_eval/hidden_20261117_v4/`):
  - the flame boxes are identical to the 12 Nov plain runs in every frame (test 3 segment 441/441; tests 4, 5
    and 6 240/240 each);
  - zone alarms are unchanged on all six clips;
  - test 3 segment: FIRE HIDDEN in the two longest flame drop-outs while the forklift was in front (clip time
    about 25.4 s and 25.9 s), before the zone alarm at 26.75 s;
  - test 3 full clip: the scene cut at 10 s is VIEW CHANGED (10.0–11.3 s), with no false FIRE HIDDEN;
  - test 5 (fire from frame 1): 28% of the frame was unknown at start-up, and the forklift alarm is unchanged
    (5.29–6.83 s, 33.5%);
  - tests 4 and 6 and the fire-only clip: no FIRE HIDDEN;
  - monitor time is about 3–6 ms median per frame.
- **Not shown yet:** a fire hidden for a long time (none of the generated clips has one) and a fire that goes out
  after start-up on a real camera. Both are in the real-camera tests.

### DINOv2 deep-feature change detection (17 Nov, optional)
`src/obstruction/deep.py`, chosen with `--zone-method deep` in `live_flame.py` (default stays `pixel`).
- **Model:** DINOv2 ViT-S/14 exported to ONNX at `models/dinov2/dinov2_vits14_252x448.onnx` (88 MB; git-ignored;
  matches PyTorch within 0.0003). 448 x 252 input, 32 x 18 patches of 14 px, 384-d feature per patch.
- **Method (PatchCore/PaDiM style, no training):**
  - a memory bank holds a feature map every 0.25 s while the scene is learned;
  - patch score = 1 - the best cosine similarity to the bank at the same patch or its 8 neighbours;
  - threshold = max(0.2, 1.5 x the 99.5th percentile of leave-one-out scores of the learning samples);
  - every 30 s, unchanged, unfrozen, non-flame patches refresh the bank.
  The changed patches feed the same grid, zones, alarm and fire tracking. A `fused` option (pixel AND deep) exists
  but behaved like the pixel method and is not recommended.
- **Speed (laptop CPU):**
  - about 100-110 ms per feature map, plus about 12 ms for the bank comparison;
  - 5 maps per second, set with `--zone-deep-every-s`;
  - live camera: a background thread runs the maps. In a simulated 24 fps loop the monitor's main-loop time was a
    median of 6.8 ms and at most 20.9 ms, against a 95th percentile of 185 ms without the thread. That loop ran
    without the flame model;
  - video files run in the main loop so results are reproducible.
- **Generated clips** (`results/obstruction_eval/deep_20261117/`; flame boxes identical in every run):
  - test 3 full clip: the steam false alarm (pixel: from 4.7 s) is gone;
  - forklift hiding the fire: OBSTRUCTED 0.8 s earlier (clip 25.9 s against 26.75 s);
  - test 1: 7.7 s against 8.1 s;
  - test 5: unchanged;
  - fire-only and test 4: no alarm;
  - test 6: alarm 4.5-7.3 s (34%) where the pixel method had none (21%). A worker in dark clothing stood inside
    the zone; this is borderline, not a fault.
  - Thresholds 0.1, 0.2 and 0.3 give nearly the same results: higher alarms slightly later; no steam alarm at any.
- **Report:** `Week4_Report.pdf` (team copy) has the full method, maths, figures and tables.

### Limitations
- **Learning:** anything in view and still during learning becomes part of the "empty" scene. Learn with the zone
  clear, or press R. On a real installation, learning should be part of commissioning.
- **Leftovers from learning, and permanent changes:** blocked cells never adapt, by design. A moved shelf stays
  "blocked" until someone re-learns.
- **FIRE HIDDEN is image-based:** it says the fire's area is covered now, not that the fire is still burning.
  It needs the fire to have been confirmed first, and it cannot see a fire that started while hidden.
- **Steam and smoke** change the image, so they count as blocked (seen in `week_4_test_3`). The monitor can't tell
  them apart from a solid object.
- **Lighting:** a whole-room brightness change is corrected; local shadows, spotlights, screens and auto-exposure
  in part of the image are not. Use ignore areas for screens and windows.
- **Camera movement** (or a scene cut in a video) makes most cells "blocked". Detecting it explicitly is Week 5
  (tamper); the learned reference here is the basis for that.
- **Image space only:** the % is of the zone's area in the image, not the floor. m² needs the Week 3 ground-plane
  calibration (`ground.py` is the hand-off point).
- **No object type:** person, vehicle and box all look the same. Labels would need a separate detector (a second
  network on the AM62A).
- **Generated (AI) video** is a pipeline check only: camera drift and flicker in generated clips cause changes a
  real fixed camera wouldn't.

### Still to do
1. Real-scene tests: redo A with the zone clear during learning, then B–G (walk past, person standing, box, two
   boxes, candle, candle blocked). Score them with `evaluate_zone_clips.py`.
2. Tune `min_blocked_pct`, `persist_s` and `clear_s` on half of those clips; report on the other half.
3. Repeat with the team V10 model, and time it on the AM62A.
4. Real-camera checks for the 17 Nov upgrade: a candle hidden by a box for at least 30 s, and a candle burning
   at start-up and then put out.
5. DINOv2 on real recordings: compare with the pixel method; check its threshold on real noise and lighting.
