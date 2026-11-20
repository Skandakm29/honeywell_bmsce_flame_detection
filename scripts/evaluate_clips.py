"""Before/after alarm evaluation on recorded clips with manually verified flame intervals.

Each clip is processed ONCE by scripts/live_flame.py --headless, so every alarm source below is judged on
identical detections:
  raw                  any detector box in the frame (the baseline)
  legacy_persistence   original --persistence K N filter (if settings.legacy_persistence is given)
  tracker_persistence  tracked K-of-N, current match required (persistent_alarm)
  fused                tracker persistence AND rules (fused_alarm; only with --verification fused)

Manifest (example: configs/verification_eval_example.json):
  settings.model / arch / confidence, settings.verification_args (must include --verification),
  settings.legacy_persistence [K, N] (optional), settings.event_merge_gap_seconds (default 1.0)
  clips[]: video, scenario, flame_intervals
flame_intervals: [[start_s, end_s], ...] when a real flame is visible; [] = verified no flame;
null = not labelled (counts only, no accuracy judgement).

Definitions:
- Frame level: a frame is positive when its video time lies inside a flame interval. precision = alarm
  frames inside intervals / alarm frames; recall = alarm frames inside intervals / positive frames.
- Alarm event: alarm frames merged when separated by at most event_merge_gap_seconds, so repeated boxes
  on one object are one event. An event overlapping a flame interval is genuine, otherwise a false alarm.
- Event recall: flame intervals overlapped by at least one event / flame intervals. Missed = the rest.
- Delay: first alarm frame inside an interval minus the interval start ('raw' = detection delay,
  others = verification delay). None when the interval was missed.
Counts and durations only: no false-alarm rate is derived, and a few clips do not establish field accuracy.
"""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = {'raw': None, 'legacy_persistence': 'alarm', 'tracker_persistence': 'persistent_alarm', 'fused': 'fused_alarm'}


def events(times, flags, gap):
    """[(start, end)] alarm events: alarm frames merged across gaps of at most `gap` seconds."""
    runs = []
    for t, flag in zip(times, flags):
        if not flag:
            continue
        if runs and t - runs[-1][1] <= gap:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return [tuple(r) for r in runs]


def judge(times, flags, intervals, gap):
    runs = events(times, flags, gap)
    result = {'alarm_frames': sum(flags), 'alarm_events': len(runs)}
    if intervals is None:
        return result | {'ground_truth': 'not labelled'}
    inside = [any(a <= t <= b for a, b in intervals) for t in times]
    true_frames = sum(f and i for f, i in zip(flags, inside))
    overlaps = lambda run, interval: run[0] <= interval[1] and run[1] >= interval[0]
    genuine = sum(any(overlaps(r, i) for i in intervals) for r in runs)
    found = [any(overlaps(r, i) for r in runs) for i in intervals]
    delays = []
    for start, end in intervals:
        first = next((t for t, f in zip(times, flags) if f and start <= t <= end), None)
        delays.append(None if first is None else round(first - start, 3))
    return result | {'positive_frames': sum(inside), 'true_alarm_frames': true_frames,
                     'frame_precision': true_frames / sum(flags) if sum(flags) else None,
                     'frame_recall': true_frames / sum(inside) if sum(inside) else None,
                     'genuine_events': genuine, 'false_alarm_events': len(runs) - genuine,
                     'flame_intervals': len(intervals), 'missed_flame_intervals': found.count(False),
                     'event_recall': found.count(True) / len(intervals) if intervals else None,
                     'delay_seconds': delays}


def summary(values):
    values = sorted(float(v) for v in values if v != '')
    if not values:
        return None
    return {'median': round(statistics.median(values), 3), 'p95': round(values[int(.95 * (len(values)-1))], 3),
            'max': round(values[-1], 3)}


def totals(clips, source):
    labelled = [c[source] for c in clips if source in c and 'frame_precision' in c[source]]
    if not labelled:
        return None
    add = lambda key: sum(r[key] for r in labelled)
    alarms, positives, true = add('alarm_frames'), add('positive_frames'), add('true_alarm_frames')
    intervals = add('flame_intervals')
    return {'labelled_clips': len(labelled), 'alarm_frames': alarms,
            'frame_precision': true / alarms if alarms else None, 'frame_recall': true / positives if positives else None,
            'alarm_events': add('alarm_events'), 'genuine_events': add('genuine_events'),
            'false_alarm_events': add('false_alarm_events'), 'flame_intervals': intervals,
            'missed_flame_intervals': add('missed_flame_intervals'),
            'event_recall': (intervals - add('missed_flame_intervals')) / intervals if intervals else None}


def fmt(value):
    return 'n/a' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: results/camera_comparison/verify_<timestamp>')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    settings = manifest['settings']
    verification = settings['verification_args']
    if '--verification' not in verification:
        parser.error('settings.verification_args must include --verification <mode>')
    mode = verification[verification.index('--verification') + 1]
    gap = float(settings.get('event_merge_gap_seconds', 1.0))
    sources = ['raw'] + (['legacy_persistence'] if settings.get('legacy_persistence') else []) + \
        ['tracker_persistence'] + (['fused'] if mode == 'fused' else [])
    output = args.output or ROOT / 'results/camera_comparison' / datetime.now().strftime('verify_%Y%m%d_%H%M%S')
    if output.exists():
        parser.error(f'{output} exists; choose a new output folder')
    model = Path(settings['model']).expanduser()
    base = args.manifest.resolve().parent
    report = {'manifest': str(args.manifest.resolve()), 'settings': settings, 'event_merge_gap_seconds': gap,
              'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(), 'sources': sources,
              'definitions': __doc__.split('Definitions:')[1].strip(), 'clips': []}
    for index, clip in enumerate(manifest['clips']):
        video = Path(clip['video']).expanduser()
        video = video if video.is_absolute() else base / video
        run = output / f'{index:02d}_{video.stem}'
        command = [sys.executable, str(ROOT/'scripts/live_flame.py'), '--headless', '--source', str(video),
                   '--model', str(model), '--arch', settings.get('arch', 'yolov9'),
                   '--confidence', str(settings.get('confidence', .40)), '--output', str(run), *verification]
        if settings.get('legacy_persistence'):
            command += ['--persistence', *map(str, settings['legacy_persistence'])]
        print('running', video.name, flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        rows = list(csv.DictReader((run/'frames.csv').open()))
        if not rows or rows[0]['video_seconds'] == '':
            raise SystemExit(f'{video}: no frames or no video timestamps')
        times = [float(r['video_seconds']) for r in rows]
        entry = {'video': str(video), 'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
                 'scenario': clip.get('scenario', ''), 'flame_intervals': clip.get('flame_intervals'),
                 'frames': len(rows), 'duration_seconds': round(times[-1], 3), 'output': str(run),
                 'processing_ms': summary(r['processing_ms'] for r in rows),
                 'verify_ms': summary(r['verify_ms'] for r in rows)}
        for source in sources:
            column = COLUMNS[source]
            flags = [r['detections_xyxy_score'] != '[]' if column is None else r[column] == '1' for r in rows]
            entry[source] = judge(times, flags, entry['flame_intervals'], gap)
        report['clips'].append(entry)
    report['totals_over_labelled_clips'] = {s: totals(report['clips'], s) for s in sources}
    (output/'report.json').write_text(json.dumps(report, indent=2))
    lines = ['# Alarm verification: before/after on recorded clips', '',
             f"Model sha256 `{report['model_sha256'][:12]}` | confidence {settings.get('confidence', .40)} | "
             f"verification `{' '.join(verification)}` | legacy persistence {settings.get('legacy_persistence')} | "
             f'events merged across gaps <= {gap} s', '',
             'All sources judged on identical detections. Counts on these clips only; not a field false-alarm rate.', '',
             '## Totals over labelled clips', '',
             '| Source | Alarm frames | Frame precision | Frame recall | Events | Genuine | False alarm | Missed flame | Event recall |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for source, t in report['totals_over_labelled_clips'].items():
        if t:
            lines.append(f"| {source} | {t['alarm_frames']} | {fmt(t['frame_precision'])} | {fmt(t['frame_recall'])} | "
                         f"{t['alarm_events']} | {t['genuine_events']} | {t['false_alarm_events']} | "
                         f"{t['missed_flame_intervals']}/{t['flame_intervals']} | {fmt(t['event_recall'])} |")
    lines += ['', '## Per clip', '',
              '| Clip | Scenario | s | Source | Alarm frames | Frame P | Frame R | Events | False | Missed | Delay s |',
              '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for c in report['clips']:
        for source in sources:
            r = c[source]
            lines.append(f"| {Path(c['video']).name} | {c['scenario']} | {c['duration_seconds']} | {source} | "
                         f"{r['alarm_frames']} | {fmt(r.get('frame_precision'))} | {fmt(r.get('frame_recall'))} | "
                         f"{r['alarm_events']} | {fmt(r.get('false_alarm_events'))} | {fmt(r.get('missed_flame_intervals'))} | "
                         f"{r.get('delay_seconds', 'n/a')} |")
    lines += ['', '## Processing time (this machine)', '',
              '| Clip | Inference+decode ms median / p95 | Verification ms median / p95 |', '|---|---|---|']
    for c in report['clips']:
        p, v = c['processing_ms'], c['verify_ms']
        lines.append(f"| {Path(c['video']).name} | {p['median']} / {p['p95']} | "
                     f"{v['median'] if v else 'n/a'} / {v['p95'] if v else 'n/a'} |")
    lines += ['', 'Laptop CPU timings, not AM62A performance. Clips are processed frame by frame at their recorded '
              'rate; a live camera drops frames, so the same K-of-N window spans a different time live.']
    (output/'report.md').write_text('\n'.join(lines) + '\n')
    print(output/'report.md')


if __name__ == '__main__':
    main()
