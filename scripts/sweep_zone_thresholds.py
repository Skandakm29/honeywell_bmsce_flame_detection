"""Choose zone-obstruction settings from LABELLED recordings by the cost of misses and false alarms.

For each detector setting (pixel grey threshold; optionally the DINOv2 minimum threshold) every clip is run ONCE
and the blocked grid cells of every frame are kept. The alarm part is then replayed offline for every combination
of zone alarm level x cell persistence x clear time, exactly as SceneMonitor does it (CellTimers + ZoneAlarm).

Each combination is scored on the TUNE clips with
    cost = miss_cost x missed obstruction intervals + false_cost x false alarm events + delay_cost x total delay s
and the cheapest one is reported on the separate TEST clips (never used for the choice).

Manifest (same as evaluate_zone_clips.py, plus "split"): settings: zones, learn_s, model (optional: flame boxes are
excluded from obstruction; the flame model is unchanged), arch; clips[]: video, scenario, zone, split ("tune" or
"test"; default alternates), obstruction_intervals ([] = none; null clips are skipped).
Results only mean something on real labelled recordings; on generated clips they are a pipeline check.
  python scripts/sweep_zone_thresholds.py --manifest configs/zone_sweep_example.json
"""
import argparse
import csv
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from evaluate_clips import judge  # noqa: E402
from obstruction import CellTimers, ReferenceDetector, SceneMonitor, ZoneAlarm, load_zones  # noqa: E402

GRID = dict(grey_threshold=(15., 25., 35.), min_blocked_pct=(10., 15., 20., 25., 30., 40.),
            persist_s=(.5, 1., 1.5, 2., 3.), clear_s=(.5, 1., 2.))


def flame_boxes(video, settings):
    """Flame boxes per frame from the unchanged flame model (or none without a model)."""
    if not settings.get('model'):
        return None
    import onnxruntime as ort
    from live_flame import decode, prepare
    session = ort.InferenceSession(str(Path(settings['model']).expanduser()), providers=['CPUExecutionProvider'])
    cap, out = cv2.VideoCapture(str(video)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tensor, scale = prepare(frame, 640, settings.get('arch', 'yolov9'))
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        out.append(decode(raw, frame, scale, .40, settings.get('arch', 'yolov9')))
    return out


def record(video, zones_file, learn_s, detector_settings, boxes):
    """Run the monitor once; return times, per-frame blocked cells (None when not judged) and the grid."""
    monitor = SceneMonitor(load_zones(zones_file), ReferenceDetector(learn_s=learn_s, **detector_settings))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    times, cells, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = monitor.update(frame, i / fps, exclude_boxes=None if boxes is None else boxes[i])
        times.append(i / fps)
        cells.append(None if out['cells'] is None else out['cells'].copy())
        i += 1
    if not times:
        raise SystemExit(f'{video}: no frames')
    return times, cells, monitor.grid


def replay(times, cells, grid, zone, min_pct, persist_s, clear_s, gap_s=1.5):
    """SceneMonitor's alarm part for one zone with other settings -> per-frame alarm flags."""
    timers = CellTimers((grid.rows, grid.cols), gap_s)
    alarm = ZoneAlarm(zone, min_pct, clear_s)
    weights = grid.weights[zone]
    flags = []
    for t, c in zip(times, cells):
        if c is None:                                # learning / view changed: nothing judged
            timers.reset()
            flags.append(alarm.update(0., t)[0])
            continue
        dwell = timers.update(c, t)
        persistent = c & (dwell >= max(persist_s, 1e-9))
        flags.append(alarm.update(float(100 * (weights * persistent).sum() / weights.sum()), t)[0])
    return flags


def score(results, costs):
    missed = sum(r['missed_flame_intervals'] for r in results)
    false = sum(r['false_alarm_events'] for r in results)
    delay = sum(d for r in results for d in r['delay_seconds'] if d is not None)
    return dict(missed=missed, false_alarms=false, total_delay_s=round(delay, 2),
                cost=round(costs['miss'] * missed + costs['false'] * false + costs['delay'] * delay, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--miss-cost', type=float, default=10., help='Cost of one missed obstruction interval')
    parser.add_argument('--false-cost', type=float, default=1., help='Cost of one false alarm event')
    parser.add_argument('--delay-cost', type=float, default=.1, help='Cost per second of alarm delay')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    base, settings = args.manifest.resolve().parent, manifest['settings']
    resolve = lambda p: Path(p).expanduser() if Path(p).expanduser().is_absolute() else base / p
    zones_file = resolve(settings['zones'])
    zone_names = [z.name for z in load_zones(zones_file).zones]
    clips = [c for c in manifest['clips'] if c.get('obstruction_intervals') is not None]
    for i, c in enumerate(clips):
        c.setdefault('split', 'tune' if i % 2 == 0 else 'test')
    if not any(c['split'] == 'tune' for c in clips) or not any(c['split'] == 'test' for c in clips):
        parser.error('need labelled clips in both the tune and the test split')
    costs = dict(miss=args.miss_cost, false=args.false_cost, delay=args.delay_cost)
    output = args.output or ROOT / 'results/zone_sweep' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    gap = float(settings.get('event_merge_gap_seconds', 1.))
    boxes = {c['video']: flame_boxes(resolve(c['video']), settings) for c in clips}
    rows = []
    for grey in GRID['grey_threshold']:
        recorded = {c['video']: record(resolve(c['video']), zones_file, float(settings.get('learn_s', 3.)),
                                       dict(grey_threshold=grey), boxes[c['video']]) for c in clips}
        print(f'grey threshold {grey}: recorded {len(clips)} clips', flush=True)
        for pct, persist, clear in itertools.product(GRID['min_blocked_pct'], GRID['persist_s'], GRID['clear_s']):
            per = {}
            for c in clips:
                times, cells, grid = recorded[c['video']]
                flags = replay(times, cells, grid, c.get('zone', zone_names[0]), pct, persist, clear)
                per[c['video']] = judge(times, flags, c['obstruction_intervals'], gap)
            row = dict(grey_threshold=grey, min_blocked_pct=pct, persist_s=persist, clear_s=clear)
            for split in ('tune', 'test'):
                s = score([per[c['video']] for c in clips if c['split'] == split], costs)
                row.update({f'{split}_{k}': v for k, v in s.items()})
            rows.append(row)
    rows.sort(key=lambda r: (r['tune_cost'], r['tune_missed'], r['tune_false_alarms']))
    with (output / 'sweep.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    current = next(r for r in rows if (r['grey_threshold'], r['min_blocked_pct'], r['persist_s'], r['clear_s'])
                   == (25., 25., 1.5, 1.))
    lines = ['# Zone threshold sweep', '',
             f'Manifest `{args.manifest}` | {len(clips)} labelled clips '
             f"({sum(c['split'] == 'tune' for c in clips)} tune, {sum(c['split'] == 'test' for c in clips)} test) | "
             f"cost = {costs['miss']} x missed + {costs['false']} x false alarms + {costs['delay']} x delay s", '',
             'Chosen on the TUNE clips only; the TEST columns are the honest estimate.', '',
             '| Setting | grey | zone % | persist s | clear s | tune cost | tune missed | tune false | test cost | '
             'test missed | test false | test delay s |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, r in (('best on tune', best), ('current default', current)):
        lines.append(f"| {label} | {r['grey_threshold']:.0f} | {r['min_blocked_pct']:.0f} | {r['persist_s']} | "
                     f"{r['clear_s']} | {r['tune_cost']} | {r['tune_missed']} | {r['tune_false_alarms']} | "
                     f"{r['test_cost']} | {r['test_missed']} | {r['test_false_alarms']} | {r['test_total_delay_s']} |")
    lines += ['', f'All {len(rows)} combinations: sweep.csv (sorted by tune cost).']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
