"""Temporal alarm layer: IoU tracker + per-track window features + verifier + alarm hysteresis.

Pure numpy at run time (portable to the AM62A ARM cores); scikit-learn only fits the logistic verifier.
Input per video: frames sampled at FPS (10) with detections [x1,y1,x2,y2,score,bright,sat,hue_spread,diff] (bench.py).

Three alarm policies are compared on the same detections:
  frame     any detection >= .40 in a frame raises the alarm (what the raw detector does)
  persist   some track has >= 60% hits at >= .40 over the last 2 s (fixed reference rule)
  persist_tuned  same rule with hit ratio, mean score and hold time tuned on the training clips
  learned   logistic verifier on track features over the last 2 s, held for HOLD_S
Event metrics: fire clips -> detected? time to alarm; all clips -> false alarm events per hour
(alarm rising edges outside the fire interval, 30 s refractory).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import read_json, work_dir, write_json

FPS = 10
WINDOW = 2 * FPS
FEATURES = ['hit_ratio', 'mean_score', 'max_score', 'area_cv', 'centroid_jitter', 'diff_energy',
            'flicker_power', 'hue_spread', 'saturation', 'brightness']


def iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.


class Tracker:
    """Greedy IoU association; a track survives MAX_GAP missed frames."""
    MAX_GAP = 5

    def __init__(self, min_iou=.3):
        self.min_iou, self.tracks, self.next_id = min_iou, {}, 0

    def update(self, frame_index, detections):
        live = {k: t for k, t in self.tracks.items() if frame_index - t['last'] <= self.MAX_GAP}
        self.tracks = live
        pairs = sorted(((iou(t['box'], d), k, j) for k, t in live.items() for j, d in enumerate(detections)), reverse=True)
        used_t, used_d = set(), set()
        for overlap, k, j in pairs:
            if overlap < self.min_iou or k in used_t or j in used_d:
                continue
            used_t.add(k)
            used_d.add(j)
            self._add(self.tracks[k], frame_index, detections[j])
        for j, d in enumerate(detections):
            if j not in used_d:
                self.tracks[self.next_id] = dict(id=self.next_id, history={}, box=d[:4], last=frame_index)
                self._add(self.tracks[self.next_id], frame_index, d)
                self.next_id += 1
        return self.tracks

    @staticmethod
    def _add(track, frame_index, detection):
        track['history'][frame_index] = detection
        track['box'], track['last'] = detection[:4], frame_index
        for old in [f for f in track['history'] if f <= frame_index - WINDOW]:
            del track['history'][old]


def track_features(track, frame_index, threshold=.4):
    h = [track['history'][f] for f in range(frame_index - WINDOW + 1, frame_index + 1) if f in track['history']]
    if not h:
        return None
    a = np.asarray(h, dtype=np.float64)
    scores = a[:, 4]
    areas = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    cx, cy = (a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2
    side = np.sqrt(max(areas.mean(), 1.))
    series = np.zeros(WINDOW)                    # brightness series (missing frames = mean) for flicker
    for f in range(frame_index - WINDOW + 1, frame_index + 1):
        series[f - frame_index + WINDOW - 1] = track['history'][f][5] if f in track['history'] else np.nan
    series = np.where(np.isnan(series), np.nanmean(series), series)
    spectrum = np.abs(np.fft.rfft(series - series.mean())) ** 2
    frequencies = np.fft.rfftfreq(WINDOW, 1 / FPS)
    band = (frequencies >= 1) & (frequencies <= 5)
    flicker = spectrum[band].sum() / (spectrum.sum() + 1e-6)
    return np.array([(scores >= threshold).sum() / WINDOW, scores.mean(), scores.max(),
                     areas.std() / (areas.mean() + 1e-6), np.hypot(cx.std(), cy.std()) / side,
                     a[:, 8].mean(), flicker, a[:, 7].mean(), a[:, 6].mean(), a[:, 5].mean()])


class Verifier:
    """Standardised logistic regression stored as plain JSON (no sklearn needed at run time)."""

    def __init__(self, mean, scale, coef, intercept, threshold, hold_s):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.coef, self.intercept = np.asarray(coef), float(intercept)
        self.threshold, self.hold_s = float(threshold), float(hold_s)

    def probability(self, x):
        z = (np.asarray(x) - self.mean) / self.scale
        return 1 / (1 + np.exp(-(z @ self.coef + self.intercept)))

    def to_json(self):
        return dict(features=FEATURES, mean=self.mean.tolist(), scale=self.scale.tolist(), coef=self.coef.tolist(),
                    intercept=self.intercept, threshold=self.threshold, hold_s=self.hold_s, fps=FPS, window_s=WINDOW / FPS)

    @classmethod
    def from_json(cls, d):
        return cls(d['mean'], d['scale'], d['coef'], d['intercept'], d['threshold'], d['hold_s'])


def load_video(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def video_signals(frames):
    """Per frame: frame-policy flag, persistence-policy flag, and per-track features."""
    tracker = Tracker()
    signals = []
    for index, frame in enumerate(frames):
        tracks = tracker.update(index, [d for d in frame['dets'] if d[4] >= .25])
        features = [f for f in (track_features(t, index) for t in tracks.values() if t['last'] == index) if f is not None]
        signals.append(dict(t=frame['t'], frame=any(d[4] >= .4 for d in frame['dets']),
                            persist=any(f[0] >= .6 for f in features), features=features))
    return signals


def in_fire(t, video):
    return any(a <= t <= b for a, b in video['fire_intervals'])


def events(times, alarms, video, refractory=30.):
    """Rising edges -> (detected, time_to_alarm, false_alarm_events)."""
    first, false, last_false, previous = None, 0, -1e9, False
    for t, on in zip(times, alarms):
        if on and in_fire(t, video):
            first = t if first is None else first     # alarm active while the flame is present
        elif on and not previous and t - last_false >= refractory:
            false += 1                                 # new alarm outside any fire interval
            last_false = t
        previous = on
    detected = first is not None if video['fire'] else None
    delay = first - video['fire_intervals'][0][0] if video['fire'] and first is not None else None
    return detected, delay, false


def hold(flags, seconds):
    """Alarm only after the flag has been continuously on for `seconds`."""
    need, run, out = max(1, int(round(seconds * FPS))), 0, []
    for f in flags:
        run = run + 1 if f else 0
        out.append(run >= need)
    return out


class Persistence:
    """Tunable debounce: some track has >= min_hit of the last 2 s with score >= .40 and mean score >= min_mean,
    continuously for hold_s. The fixed 'persist' policy is min_hit .6, min_mean 0, hold 0."""

    def __init__(self, min_hit, min_mean, hold_s):
        self.min_hit, self.min_mean, self.hold_s = float(min_hit), float(min_mean), float(hold_s)

    def to_json(self):
        return dict(min_hit=self.min_hit, min_mean=self.min_mean, hold_s=self.hold_s, fps=FPS, window_s=WINDOW / FPS)


def apply(signals, policy, rule=None):
    if policy in ('frame', 'persist'):
        return [s[policy] for s in signals]
    if policy == 'persist_tuned':
        raw = [any(f[0] >= rule.min_hit and f[1] >= rule.min_mean for f in s['features']) for s in signals]
        return hold(raw, rule.hold_s)
    raw = [bool(s['features']) and rule.probability(np.asarray(s['features'])).max() >= rule.threshold for s in signals]
    return hold(raw, rule.hold_s)


def best_rule(videos, signals_by_video, policy, candidates, max_false_per_hour):
    """Highest event recall within the false-alarm budget, then fastest alarm, then fewest false alarms."""
    best = None
    for rule in candidates:
        s = summarise(videos, signals_by_video, policy, rule)
        key = (s['false_alarms_per_hour'] <= max_false_per_hour, s['event_recall'],
               -(s['median_time_to_alarm_s'] or 1e9), -s['false_alarms_per_hour'])
        if best is None or key > best[0]:
            best = (key, rule, s)
    return best[1], best[2]


def tune_persistence(videos, signals_by_video, max_false_per_hour=1.0):
    candidates = [Persistence(h, m, d) for h in (.2, .3, .4, .5, .6, .7, .8) for m in (0., .4, .5, .6)
                  for d in (0., .5, 1., 2.)]
    return best_rule(videos, signals_by_video, 'persist_tuned', candidates, max_false_per_hour)


def summarise(videos, signals_by_video, policy, verifier=None):
    rows, hours, false = [], 0., 0
    for v in videos:
        s = signals_by_video[v['name']]
        detected, delay, fa = events([x['t'] for x in s], apply(s, policy, verifier), v)
        rows.append(dict(video=v['name'], source=v['source'], category=v['category'], fire=v['fire'],
                         detected=detected, time_to_alarm_s=delay, false_alarms=fa))
        hours += v['duration_s'] / 3600
        false += fa
    fire = [r for r in rows if r['fire']]
    delays = [r['time_to_alarm_s'] for r in fire if r['time_to_alarm_s'] is not None]
    return dict(policy=policy, fire_videos=len(fire), event_recall=sum(r['detected'] for r in fire) / max(1, len(fire)),
                median_time_to_alarm_s=float(np.median(delays)) if delays else None,
                false_alarms=false, hours=hours, false_alarms_per_hour=false / max(hours, 1e-9), videos=rows)


def training_rows(videos, signals_by_video):
    """Window features labelled by clip: fire clips inside the fire interval -> 1, non-fire clips -> 0.
    Mixed 'flame next to nuisance' clips are excluded (their nuisance tracks would be mislabelled)."""
    X, y, groups = [], [], []
    for v in videos:
        if 'nuisance' in v['category'].lower() or 'mixed' in v['category'].lower():
            continue
        for s in signals_by_video[v['name']]:
            if v['fire'] and not in_fire(s['t'], v):
                continue
            if v['fire'] and s['features']:
                X.append(max(s['features'], key=lambda f: f[1]))   # strongest track in a fire clip
                y.append(1)
                groups.append(v['name'])
            elif not v['fire']:
                for f in s['features']:
                    X.append(f)
                    y.append(0)
                    groups.append(v['name'])
    return np.asarray(X), np.asarray(y), groups


def fit(videos, signals_by_video, max_false_per_hour=1.0):
    from sklearn.linear_model import LogisticRegression
    X, y, _ = training_rows(videos, signals_by_video)
    if len(set(y.tolist())) < 2:   # e.g. the detector finds nothing at all on the non-fire training clips
        return None, None, dict(rows=len(y), positives=int(y.sum()), negatives=int(len(y) - y.sum()),
                                note='verifier not fitted: training windows of only one class; persistence rule only')
    mean, scale = X.mean(0), X.std(0) + 1e-6
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit((X - mean) / scale, y)
    candidates = [Verifier(mean, scale, model.coef_[0], model.intercept_[0], threshold, hold_s)
                  for threshold in np.linspace(.3, .95, 14) for hold_s in (0., .5, 1., 2.)]
    verifier, fitted = best_rule(videos, signals_by_video, 'learned', candidates, max_false_per_hour)
    return verifier, fitted, dict(rows=len(y), positives=int(y.sum()), negatives=int(len(y) - y.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='bench name, e.g. v9 or v10')
    p.add_argument('--max-false-per-hour', type=float, default=1.0)
    a = p.parse_args()
    work = work_dir()
    videos = read_json(work / 'sources' / 'videos.json')['videos']
    folder = work / 'bench' / a.model / 'videos'
    signals = {v['name']: video_signals(load_video(folder / f'{v["name"]}.jsonl')) for v in videos}
    train = [v for v in videos if v['source'] == 'own' and v['split'] in ('train', 'hnval')]
    evaluation = [v for v in videos if v['split'] == 'eval']
    verifier, fitted, sizes = fit(train, signals, a.max_false_per_hour)
    persistence, persistence_fitted = tune_persistence(train, signals, a.max_false_per_hour)
    rules = {'frame': None, 'persist': None, 'persist_tuned': persistence}
    if verifier:
        rules['learned'] = verifier
    result = dict(model=a.model, verifier=verifier.to_json() if verifier else None, training_rows=sizes,
                  fitted_on_train_clips=fitted, persistence=persistence.to_json(),
                  persistence_fitted_on_train_clips=persistence_fitted)
    sets = sorted({v['source'] for v in evaluation})          # own, mivia, onfire (test only)
    for source in sets:
        group = [v for v in evaluation if v['source'] == source]
        result[f'{source}_eval'] = {policy: summarise(group, signals, policy, rule) for policy, rule in rules.items()}
    out = work / 'temporal' / a.model
    write_json(result, out / 'temporal.json')
    write_json(persistence.to_json(), out / 'persistence.json')
    if verifier:
        write_json(verifier.to_json(), out / 'verifier.json')
    print('training windows:', sizes, '| tuned persistence:', persistence.to_json())
    for source in sets:
        for policy, s in result[f'{source}_eval'].items():
            print(f'{source:7s} {policy:13s} recall {s["event_recall"]:.2f}  median delay {s["median_time_to_alarm_s"]}  '
                  f'false/h {s["false_alarms_per_hour"]:.2f}')


if __name__ == '__main__':
    main()
