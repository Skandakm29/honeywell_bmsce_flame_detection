"""Score the zone-obstruction monitor on recorded clips with manually noted obstruction times.

Each clip runs ONCE headless:
  - settings.model given -> scripts/live_flame.py --zones (flame boxes are excluded from obstruction)
  - otherwise            -> scripts/test_zone_monitor.py (obstruction only, no model)

Manifest (example: configs/zone_eval_example.json; paths relative to the manifest):
  settings: zones, learn_s (default 3), event_merge_gap_seconds (default 1.0), model (optional), arch
  clips[]: video, scenario, zone (default: first zone), obstruction_intervals
obstruction_intervals: [[start_s, end_s], ...] when the zone is really blocked by something that stays
(start = when it stops moving); [] = verified no obstruction (empty, walking past, lighting, flame only);
null = not labelled (counts only).

Per clip and zone, from the logged zone alarm:
- alarm events (alarm frames merged across gaps <= event_merge_gap_seconds), genuine vs false alarm events
- missed obstruction intervals; delay = first alarm - interval start (includes the zone's persist_s by design)
- end error = the monitor's logged event end_s - interval end (end_s is when the zone fell below the
  threshold; the alarm flag itself stays on for clear_s after that by design)
Counts on these clips only; not a field false-alarm rate. Generated (AI) video is a pipeline check, not evidence.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_clips import judge  # noqa: E402
from obstruction import load_zones  # noqa: E402


def score(times, alarms, intervals, gap, logged=()):
    """judge() on the alarm flags, plus the end error from the monitor's logged events [(start_s, end_s)]."""
    result = judge(times, alarms, intervals, gap)
    if intervals:
        result['end_error_seconds'] = []
        for start, end in intervals:
            hit = [e for e in logged if e[0] <= end and e[1] >= start]
            result['end_error_seconds'].append(round(hit[-1][1] - end, 3) if hit else None)
    return result


def run_clip(video, settings, zones_file, output):
    if settings.get('model'):
        command = [sys.executable, str(ROOT / 'scripts/live_flame.py'), '--headless', '--source', str(video),
                   '--model', str(Path(settings['model']).expanduser()), '--arch', settings.get('arch', 'yolov9'),
                   '--zones', str(zones_file), '--zone-learn-s', str(settings.get('learn_s', 3.)),
                   '--output', str(output)]
        time_column, prefix, event_file = 'video_seconds', 'zone_', 'zone_events.csv'
    else:
        command = [sys.executable, str(ROOT / 'scripts/test_zone_monitor.py'), '--headless', '--source', str(video),
                   '--zones', str(zones_file), '--learn-s', str(settings.get('learn_s', 3.)), '--output', str(output)]
        time_column, prefix, event_file = 't_s', '', 'events.csv'
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    rows = list(csv.DictReader((output / 'frames.csv').open(encoding='utf-8')))
    if not rows:
        raise SystemExit(f'{video}: no frames processed')
    logged = list(csv.DictReader((output / event_file).open(encoding='utf-8')))
    return rows, time_column, prefix, logged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: results/obstruction_eval/<timestamp>')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    base, settings = args.manifest.resolve().parent, manifest['settings']
    resolve = lambda p: Path(p).expanduser() if Path(p).expanduser().is_absolute() else base / p
    zones_file = resolve(settings['zones'])
    config = load_zones(zones_file)
    gap = float(settings.get('event_merge_gap_seconds', 1.))
    output = args.output or ROOT / 'results/obstruction_eval' / datetime.now().strftime('%Y%m%d_%H%M%S')
    if output.exists():
        parser.error(f'{output} exists; choose a new output folder')
    report = dict(manifest=str(args.manifest.resolve()), settings=settings, zones_file=str(zones_file),
                  definitions=__doc__.split('Per clip and zone')[1].strip(), clips=[])
    for index, clip in enumerate(manifest['clips']):
        video = resolve(clip['video'])
        zone = clip.get('zone', config.zones[0].name)
        print('running', video.name, flush=True)
        rows, time_column, prefix, logged = run_clip(video, settings, zones_file, output / f'{index:02d}_{video.stem}')
        times = [float(r[time_column]) for r in rows]
        alarm_column = f'{prefix}{zone}_alarm'
        if alarm_column not in rows[0]:
            raise SystemExit(f'{video}: no column {alarm_column}; zone {zone!r} not in {zones_file}')
        alarms = [r[alarm_column] == '1' for r in rows]
        entry = dict(video=str(video), scenario=clip.get('scenario', ''), zone=zone,
                     obstruction_intervals=clip.get('obstruction_intervals'), frames=len(rows),
                     duration_s=round(times[-1], 3),
                     learned_at_s=next((t for t, r in zip(times, rows)
                                        if r.get(f'{prefix}state', r.get('state')) == 'ready'), None),
                     result=score(times, alarms, clip.get('obstruction_intervals'), gap,
                                  [(float(e['start_s']), float(e['end_s'])) for e in logged if e['zone'] == zone]))
        report['clips'].append(entry)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fmt = lambda v: 'n/a' if v is None else f'{v:.2f}' if isinstance(v, float) else str(v)
    lines = ['# Zone obstruction: clip results', '',
             f"Zones `{zones_file.name}` | flame exclusion: {'yes (' + settings['model'] + ')' if settings.get('model') else 'no (no model)'}"
             f' | events merged across gaps <= {gap} s', '',
             'Counts on these clips only; not a field false-alarm rate. Generated video is a pipeline check only.', '',
             '| Clip | Scenario | s | Learned at s | Alarm events | False alarms | Missed | Delay s | End error s |',
             '|---|---|---:|---:|---:|---:|---:|---|---|']
    for c in report['clips']:
        r = c['result']
        lines.append(f"| {Path(c['video']).name} | {c['scenario']} | {c['duration_s']} | {fmt(c['learned_at_s'])} | "
                     f"{r['alarm_events']} | {fmt(r.get('false_alarm_events'))} | "
                     f"{fmt(r.get('missed_flame_intervals'))} | {r.get('delay_seconds', 'n/a')} | "
                     f"{r.get('end_error_seconds', 'n/a')} |")
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(output / 'report.md')


if __name__ == '__main__':
    main()
