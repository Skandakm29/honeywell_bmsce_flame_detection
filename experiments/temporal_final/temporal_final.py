"""Final temporal alarm rule for V10 + small-box threshold (protocol: TEMPORAL_PROTOCOL.md, fixed before running).

Reads (never writes) bench/<model>/videos/*.jsonl (10 fps, score floor .25) and sources/videos.json from the V10 work
folder. Runs the detector itself only on the extra test clips (corridor recordings + LED candle). Writes only
<work>/temporal_final/.  CPU is enough; minutes.
  python temporal_final.py
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from common import read_json, work_dir, write_json   # V10 code, read-only
import temporal as T                                  # V10 code, read-only: iou, events, in_fire, video_signals, apply

BASE, SMALL_SIDE = .40, 16                           # small = longest side < 16 px at the 640 input
SMALL_THRESHOLDS = (.25, .30, .35)                    # lowest one that adds no false alarms is kept; else none
GRID = dict(W=(1., 2., 3.), h=(.3, .5, .7, .9), m=(0., .45, .55, .65), hold=(0., 1., 2.))
OFF_S, TRACK_GAP_S, REFRACTORY_S = 3., 1., 30.
MATCH_IOU, MATCH_CENTRE = .1, 1.5
RECALL_FLOOR, MAX_DELAY_S = .90, 30.
FPS_VARIANTS = {'10fps': 1, '5fps': 2}
SPLIT_SALT = 'onfire-split-v1'


# ---------------------------------------------------------------- per-frame gate and location tracker
def gate(dets, to640, small_t):
    out = []
    for d in dets:
        small = max(d[2] - d[0], d[3] - d[1]) * to640 < SMALL_SIDE
        if d[4] >= BASE or (small_t is not None and small and d[4] >= small_t):
            out.append(d)
    return out


def track(frames, to640, small_t):
    tracks, next_id, per = {}, 0, []
    for fr in frames:
        t = fr['t']
        tracks = {k: v for k, v in tracks.items() if t - v['last'] <= TRACK_GAP_S}
        used, assigned = set(), []
        for d in sorted(gate(fr['dets'], to640, small_t), key=lambda d: -d[4]):
            best = None
            for k, v in tracks.items():
                if k in used:
                    continue
                b = v['box']
                overlap = T.iou(b, d)
                side = max(d[2] - d[0], d[3] - d[1], b[2] - b[0], b[3] - b[1])
                dist = math.hypot((b[0] + b[2] - d[0] - d[2]) / 2, (b[1] + b[3] - d[1] - d[3]) / 2)
                if overlap >= MATCH_IOU or dist <= MATCH_CENTRE * side:
                    key = (overlap, -dist)
                    if best is None or key > best[0]:
                        best = (key, k)
            if best is None:
                k, next_id = next_id, next_id + 1
                tracks[k] = {}
            else:
                k = best[1]
            used.add(k)
            tracks[k].update(box=d[:4], last=t)
            assigned.append((k, d[4]))
        per.append(assigned)
    return per


# ---------------------------------------------------------------- alarm rule (time based)
def raw_flags(times, per, W, h, m):
    dt = float(np.median(np.diff(times))) if len(times) > 1 else .1
    expected = max(1, round(W / dt))                  # a part-filled window never counts as 100%
    window, flags = deque(), []
    for t, a in zip(times, per):
        window.append((t, a))
        while window and window[0][0] <= t - W + 1e-9:
            window.popleft()
        n = max(len(window), expected)
        hits = defaultdict(list)
        for _, aa in window:
            for k, s in dict(aa).items():
                hits[k].append(s)
        flags.append(any(len(v) / n >= h and float(np.mean(v)) >= m for v in hits.values()))
    return flags


def alarm(times, flags, hold):
    on, run_start, last_true, out = False, None, None, []
    for t, f in zip(times, flags):
        if f:
            run_start = t if run_start is None else run_start
            last_true = t
        else:
            run_start = None
        if not on and f and t - run_start >= hold - 1e-9:
            on = True
        elif on and (last_true is None or t - last_true >= OFF_S - 1e-9):
            on = False
        out.append(on)
    return out


def rule_policy(W, h, m, hold):
    return lambda times, per, frames: alarm(times, raw_flags(times, per, W, h, m), hold)


def frame_policy(times, per, frames):
    return [bool(x) for x in per]


def old_policy(rule):
    return lambda times, per, frames: T.apply(T.video_signals(frames), 'persist_tuned', rule)


# ---------------------------------------------------------------- data and evaluation
def load_frames(work, model, v):
    path = Path(v['jsonl'][model]) if 'jsonl' in v else work / 'bench' / model / 'videos' / f'{v["name"]}.jsonl'
    return T.load_video(path)


def frame_size(v, cache={}):
    if v['path'] not in cache:
        import cv2
        cap = cv2.VideoCapture(v['path'])
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not w or not h:
            raise RuntimeError(f'cannot read frame size of {v["path"]} (needed for the 640-px size gate)')
        cache[v['path']] = (w, h)
    return cache[v['path']]


def prepare(work, model, videos, small_t, step):
    data = {}
    for v in videos:
        frames = load_frames(work, model, v)[::step]
        to640 = 640 / max(frame_size(v))
        data[v['name']] = ([f['t'] for f in frames], track(frames, to640, small_t), frames)
    return data


def evaluate(videos, data, policy):
    rows, hours = [], 0.
    for v in videos:
        times, per, frames = data[v['name']]
        alarms = policy(times, per, frames)
        detected, delay, fa = T.events(times, alarms, v, REFRACTORY_S)
        inside = [on for t, on in zip(times, alarms) if T.in_fire(t, v)] if v['fire'] else []
        rows.append(dict(video=v['name'], source=v['source'], category=v['category'], fire=v['fire'],
                         detected=detected, time_to_alarm_s=delay, false_alarms=fa, hours=v['duration_s'] / 3600,
                         coverage=sum(inside) / len(inside) if inside else None))
        hours += v['duration_s'] / 3600
    fire = [r for r in rows if r['fire']]
    delays = [r['time_to_alarm_s'] for r in fire if r['time_to_alarm_s'] is not None]
    cov = [r['coverage'] for r in fire if r['coverage'] is not None]
    false = sum(r['false_alarms'] for r in rows)
    return dict(fire_videos=len(fire), event_recall=sum(bool(r['detected']) for r in fire) / max(1, len(fire)) if fire else None,
                alarm_coverage=float(np.mean(cov)) if cov else None,
                median_time_to_alarm_s=float(np.median(delays)) if delays else None,
                max_time_to_alarm_s=float(max(delays)) if delays else None,
                false_alarms=false, hours=hours, false_alarms_per_hour=false / max(hours, 1e-9), videos=rows)


# ---------------------------------------------------------------- robust tuning
KEYS = [(W, h, m, hold) for W in GRID['W'] for h in GRID['h'] for m in GRID['m'] for hold in GRID['hold']]
AXES = [GRID['W'], GRID['h'], GRID['m'], GRID['hold']]


def neighbours(k):
    idx = [AXES[i].index(k[i]) for i in range(4)]
    out = [k]
    for i in range(4):
        for d in (-1, 1):
            j = idx[i] + d
            if 0 <= j < len(AXES[i]):
                out.append(tuple(AXES[a][j if a == i else idx[a]] for a in range(4)))
    return out


_prep_cache = {}


def cache_prepare(work, model, videos, small_t, step):
    key = (model, tuple(v['name'] for v in videos), small_t, step)
    if key not in _prep_cache:
        _prep_cache[key] = prepare(work, model, videos, small_t, step)
    return _prep_cache[key]


def tune(per_fps):
    table = []
    for k in KEYS:
        cells = [per_fps[f][n] for f in per_fps for n in neighbours(k)]
        worst = dict(recall=min(c['event_recall'] or 0. for c in cells), fa=max(c['false_alarms_per_hour'] for c in cells),
                     median=max((c['median_time_to_alarm_s'] or 0.) for c in cells),
                     max_delay=max((c['max_time_to_alarm_s'] or 0.) for c in cells),
                     coverage=min((c['alarm_coverage'] or 0.) for c in cells))
        feasible = worst['recall'] >= RECALL_FLOOR and worst['max_delay'] <= MAX_DELAY_S
        table.append(dict(W=k[0], h=k[1], m=k[2], hold=k[3], feasible=feasible, worst=worst))
    simple = lambda r: (r['m'] == 0., -r['hold'], r['W'])          # on a tie the simpler rule wins
    pool = [r for r in table if r['feasible']]
    if pool:
        best = max(pool, key=lambda r: (-round(r['worst']['fa'], 3), round(r['worst']['coverage'], 2),
                                        -round(r['worst']['median'], 1), *simple(r)))
    else:
        best = max(table, key=lambda r: (r['worst']['recall'], -round(r['worst']['fa'], 3),
                                         round(r['worst']['coverage'], 2), *simple(r)))
    return best, table


def worst_fa(work, model, tune_set, small_t, rule):
    return max(evaluate(tune_set, cache_prepare(work, model, tune_set, small_t, step), rule_policy(**rule))['false_alarms_per_hour']
               for step in FPS_VARIANTS.values())


def tune_with_gate(work, model, tune_set):
    """Lowest small-box threshold whose tuned rule adds NO false alarms over the same rule without the small-box gate."""
    tried = []
    for small_t in SMALL_THRESHOLDS + (None,):
        per_fps = {label: {k: evaluate(tune_set, cache_prepare(work, model, tune_set, small_t, step), rule_policy(*k))
                           for k in KEYS} for label, step in FPS_VARIANTS.items()}
        best, table = tune(per_fps)
        rule = dict(W=best['W'], h=best['h'], m=best['m'], hold=best['hold'])
        fa_gate = worst_fa(work, model, tune_set, small_t, rule)
        fa_plain = worst_fa(work, model, tune_set, None, rule)
        ok = small_t is None or fa_gate <= fa_plain + 1e-9
        tried.append(dict(small_threshold=small_t, rule=rule, feasible=best['feasible'], worst=best['worst'],
                          fa_with_gate=fa_gate, fa_without_gate=fa_plain, accepted=ok))
        print('small threshold', small_t, 'rule', rule, 'fa gate/plain', round(fa_gate, 2), round(fa_plain, 2), 'accepted', ok, flush=True)
        if ok:
            return small_t, rule, best, table, tried


# ---------------------------------------------------------------- extra test clips (detector run here)
def extra_clips(work, out):
    """Corridor recordings (fire, distance test) + LED candle (not fire). Logged at 10 fps, score floor .25."""
    import cv2
    from common import detect_frames, load_detector
    clips = []
    for p in sorted((work / 'recordings_v12').rglob('corridor_*.mp4')):
        clips.append((p, True, 'corridor_' + p.stem.split('_')[-1]))
    led = Path(__file__).resolve().parent / 'test_clips' / 'led_candle.mp4'
    if led.exists():
        clips.append((led, False, 'led_fake_flame'))
    checkpoints = {'v10': work / 'FROZEN_V10' / 'selected.pth', 'v12r': work / 'select_v12_rec' / 'best.pth'}
    videos = []
    for path, fire, category in clips:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n / fps
        name = 'extra_' + path.stem
        v = dict(name=name, path=str(path), source='extra', category=category, fire=fire,
                 fire_intervals=[[0., duration]] if fire else [], fps=fps, duration_s=duration, split='eval', jsonl={})
        for model, ckpt in checkpoints.items():
            if not ckpt.exists():
                continue
            target = out / 'extra_logs' / model / f'{name}.jsonl'
            if not target.exists():
                det = load_detector(ckpt)
                target.parent.mkdir(parents=True, exist_ok=True)
                cap = cv2.VideoCapture(str(path))
                lines, index, next_t = [], 0, 0.
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    t = index / fps
                    if t + 1e-6 >= next_t:
                        dets = detect_frames(det, [frame], floor=.25)[0]
                        lines.append(json.dumps(dict(t=round(t, 3), dets=[[round(x, 4) for x in d[:5]] + [0., 0., 0., 0.] for d in dets])))
                        next_t += .1
                    index += 1
                target.write_text('\n'.join(lines) + '\n')
            v['jsonl'][model] = str(target)
        videos.append(v)
    return videos


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.parse_args()
    work = work_dir()
    out = work / 'temporal_final'
    out.mkdir(exist_ok=True)
    videos = read_json(work / 'sources' / 'videos.json')['videos']
    # BUG FIX (7 Nov, run 3): 12 ONFIRE videos are listed twice in videos.json (same file, once as fire, once as
    # no-fire). Which label is right cannot be told from the files, so every name listed more than once is dropped.
    counts = defaultdict(int)
    for v in videos:
        counts[v['name']] += 1
    colliding = sorted(n for n, c in counts.items() if c > 1)
    videos = [v for v in videos if v['name'] not in colliding]

    def half(v):
        return int(hashlib.md5((SPLIT_SALT + v['name']).encode()).hexdigest(), 16) % 2
    onfire = [v for v in videos if v['source'] == 'onfire' and v['split'] == 'eval']
    tune_half = [dict(v, source='onfire_tune') for v in onfire if half(v) == 0]
    test_half = [dict(v, source='onfire_test') for v in onfire if half(v) == 1]
    own_train = [v for v in videos if v['source'] == 'own' and v['split'] in ('train', 'hnval')]
    split = dict(salt=SPLIT_SALT, rule='md5(salt + name) % 2: 0 = tuning, 1 = held-out test',
                 tune=sorted(v['name'] for v in tune_half), test=sorted(v['name'] for v in test_half))
    frozen = out / 'onfire_split.json'                 # run 1 (before the fix): the remaining split must be a subset
    if frozen.exists():
        old = read_json(frozen)
        if not (set(split['tune']) <= set(old['tune']) and set(split['test']) <= set(old['test'])):
            raise RuntimeError('ONFIRE split changed beyond dropping colliding names; never change it')
    else:
        write_json(split, frozen)
    write_json(dict(split, dropped_colliding_names=colliding), out / 'onfire_split_v2.json')
    tune_a = own_train + tune_half                     # option A (owner decision 7 Nov)
    tune_b = own_train                                 # clips only, shown side by side
    held = [v for v in videos if v['split'] == 'eval' and v['source'] != 'onfire'] + test_half + extra_clips(work, out)

    results = dict(protocol='TEMPORAL_PROTOCOL.md', dropped_colliding_names=colliding, run='3 (duplicate entries dropped)', base_threshold=BASE, small_side_px=SMALL_SIDE, grid=GRID,
                   off_s=OFF_S, track_gap_s=TRACK_GAP_S, match=dict(iou=MATCH_IOU, centre_x_side=MATCH_CENTRE),
                   recall_floor=RECALL_FLOOR, max_delay_s=MAX_DELAY_S, onfire_split=dict(tune=len(tune_half), test=len(test_half)),
                   tuning_a=[v['name'] for v in tune_a], held_out=[v['name'] for v in held], setups={})
    small_t, rule, best, table, tried = tune_with_gate(work, 'v10', tune_a)
    small_b, rule_b, best_b, _, tried_b = tune_with_gate(work, 'v10', tune_b)
    results['gate_search'] = tried
    results['gate_search_clips_only'] = tried_b
    setups = [('V10 + gate + locked rule (option A)', 'v10', small_t, rule, best),
              ('V10 + gate + rule tuned on own clips only (option B, comparison)', 'v10', small_b, rule_b, best_b)]
    if (work / 'bench' / 'v12r' / 'videos').exists():
        setups.append(('V12 + same locked rule, no gate', 'v12r', None, rule, None))
    old = work / 'temporal' / 'v10' / 'persistence.json'
    old_rule = None
    if old.exists():
        o = read_json(old)
        old_rule = T.Persistence(o['min_hit'], o['min_mean'], o['hold_s'])
    for name, model, gt, r, b in setups:
        entry = dict(model=model, small_threshold=gt, rule=r, training_worst_case=b['worst'] if b else None,
                     feasible=b['feasible'] if b else None, held_out={})
        usable = [v for v in held if 'jsonl' not in v or model in v['jsonl']]
        for label, step in FPS_VARIANTS.items():
            gated = prepare(work, model, usable, gt, step)
            plain = prepare(work, model, usable, None, step)
            pols = {'1_no_temporal_.40': (plain, frame_policy)}
            if gt is not None:
                pols['2_no_temporal_gate'] = (gated, frame_policy)
            if old_rule is not None and label == '10fps' and model == 'v10':
                std = [v for v in usable if 'jsonl' not in v]
                pols['3_old_v10_rule'] = (plain, old_policy(old_rule), std)
            pols['4_final_rule'] = (gated, rule_policy(**r))
            for pol, spec in pols.items():
                d, fn = spec[0], spec[1]
                group_all = spec[2] if len(spec) > 2 else usable
                for src in sorted({v['source'] for v in group_all}):
                    group = [v for v in group_all if v['source'] == src]
                    entry['held_out'].setdefault(src, {}).setdefault(label, {})[pol] = evaluate(group, d, fn)
        results['setups'][name] = entry
    write_json(results, out / 'results.json')
    lock = dict(model='V10 (FROZEN_V10/selected.pth, sha256 d046dab6...)', protocol='TEMPORAL_PROTOCOL.md',
                score_threshold=BASE, small_box=dict(longest_side_px_at_640_below=SMALL_SIDE, threshold=small_t),
                tracker=dict(match_iou=MATCH_IOU, match_centre_x_longest_side=MATCH_CENTRE, drop_after_s=TRACK_GAP_S),
                window_s=rule['W'], min_hit_ratio=rule['h'], min_mean_score=rule['m'], alarm_on_after_s=rule['hold'],
                alarm_off_after_s=OFF_S, notes='time based; tested at 10 and 5 fps')
    write_json(lock, out / 'rule.json')
    report(results, lock, out)


def pct(x):
    return '-' if x is None else f'{100 * x:.1f}%'


def sec(x):
    return '-' if x is None else f'{x:.1f}'


def report(results, lock, out):
    L = ['# Temporal alarm rule for V10 (measured on the lab PC; protocol TEMPORAL_PROTOCOL.md)', '',
         '## Locked rule', '', '```', json.dumps(lock, indent=1), '```', '',
         'Small-box threshold search (kept only if it adds no false alarms on the tuning no-fire video):', '']
    for t in results['gate_search']:
        L.append(f'- small threshold {t["small_threshold"]}: false alarms/h with gate {t["fa_with_gate"]:.2f}, '
                 f'without {t["fa_without_gate"]:.2f} -> {"KEPT" if t["accepted"] else "rejected"}')
    L.append('')
    for name, e in results['setups'].items():
        L += [f'## {name}', '', f'Rule {e["rule"]}, small threshold {e["small_threshold"]}.', '',
              '| set | fps | policy | fires caught | coverage | median s | max s | false alarms / h |', '|---|---|---|---|---|---|---|---|']
        for src, by in e['held_out'].items():
            for fps, pols in by.items():
                for pol, s in sorted(pols.items()):
                    L.append(f'| {src} | {fps} | {pol} | {pct(s["event_recall"])} ({s["fire_videos"]}) | {pct(s["alarm_coverage"])} | '
                             f'{sec(s["median_time_to_alarm_s"])} | {sec(s["max_time_to_alarm_s"])} | '
                             f'{s["false_alarms_per_hour"]:.2f} ({s["false_alarms"]} in {s["hours"]:.2f} h) |')
        L.append('')
        # per clip (10 fps): time to alarm for fire clips, false alarms for others, baseline vs final
        L += ['Per clip, 10 fps (baseline .40 / final rule):', '', '| clip | kind | category | baseline | final |', '|---|---|---|---|---|']
        for src, by in e['held_out'].items():
            base = {r['video']: r for r in by['10fps']['1_no_temporal_.40']['videos']}
            final = {r['video']: r for r in by['10fps']['4_final_rule']['videos']}
            for vname, f in final.items():
                b = base[vname]
                if f['fire']:
                    L.append(f'| {vname} | fire | {f["category"]} | {"alarm " + sec(b["time_to_alarm_s"]) + " s" if b["detected"] else "MISSED"} | '
                             f'{"alarm " + sec(f["time_to_alarm_s"]) + " s" if f["detected"] else "MISSED"} |')
                else:
                    L.append(f'| {vname} | no fire | {f["category"]} | {b["false_alarms"]} false alarms | {f["false_alarms"]} false alarms |')
        L.append('')
        # false alarms per hour by nuisance category (10 fps, final rule vs baseline)
        cats = defaultdict(lambda: [0, 0, 0.])
        for src, by in e['held_out'].items():
            base = {r['video']: r for r in by['10fps']['1_no_temporal_.40']['videos']}
            for r in by['10fps']['4_final_rule']['videos']:
                if not r['fire']:
                    c = cats[r['category']]
                    c[0] += base[r['video']]['false_alarms']
                    c[1] += r['false_alarms']
                    c[2] += r['hours']
        L += ['False alarms per hour by nuisance type (10 fps):', '', '| type | hours | baseline .40 | final rule |', '|---|---|---|---|']
        for k, (b, f, h) in sorted(cats.items()):
            L.append(f'| {k} | {h:.2f} | {b / max(h, 1e-9):.1f} | {f / max(h, 1e-9):.1f} |')
        L.append('')
    (out / 'REPORT_TEMPORAL.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
