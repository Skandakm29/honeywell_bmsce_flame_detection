"""Week 2 locked flame alarm rule, frame by frame for the live viewer (no new rule, no new parameters).

A streaming port of experiments/temporal_final/temporal_final.py (gate -> track -> raw_flags -> alarm) with the
numbers from results/week2/rule.json (ALARM_SPEC.md). tests/test_flame_alarm.py checks that it gives exactly the
same per-frame tracks and ALARM as those offline functions. The flame model, its boxes and NMS are untouched:
this only reads the decoded boxes.

Per frame:
1. Gate: a box counts if score >= 0.40, or its longest side at the 640 input is < 16 px and score >= 0.25.
2. Track by location: highest score first, a counted box continues the best existing track with IoU >= 0.1 or
   centre distance <= 1.5 x the longest side of the two boxes; otherwise a new track. A track is dropped after
   1.0 s without a counted box.
3. flag = some track was counted in >= 30% of the frames of the last 2 s (and its mean score >= 0).
4. ALARM on when flag; off after 3 s without flag. WARNING = any counted box in this frame.
A track that has satisfied the flag condition is CONFIRMED: the Week 4 visibility check only follows confirmed fires.
"""
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

RULE_FILE = Path(__file__).resolve().parents[1] / 'results/week2/rule.json'


def iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


class FlameAlarm:
    def __init__(self, rule=None, frame_dt=None):
        """rule: dict as in results/week2/rule.json (default: that file). frame_dt: nominal seconds per processed
        frame (video: 1/fps). Without it the median of the intervals seen so far is used, as offline."""
        rule = rule or json.loads(RULE_FILE.read_text(encoding='utf-8'))
        self.base = rule['score_threshold']
        self.small_side = rule['small_box']['longest_side_px_at_640_below']
        self.small_t = rule['small_box']['threshold']
        self.match_iou = rule['tracker']['match_iou']
        self.match_centre = rule['tracker']['match_centre_x_longest_side']
        self.drop_s = rule['tracker']['drop_after_s']
        self.window_s, self.min_hit, self.min_mean = rule['window_s'], rule['min_hit_ratio'], rule['min_mean_score']
        self.on_after_s, self.off_after_s = rule['alarm_on_after_s'], rule['alarm_off_after_s']
        self.frame_dt = frame_dt
        self.tracks, self.next_id = {}, 0
        self.window, self.times = deque(), []
        self.alarm, self._run_start, self._last_flag = False, None, None
        self.confirmed = set()

    def gate(self, boxes, to640):
        out = []
        for d in boxes:
            small = max(d[2] - d[0], d[3] - d[1]) * to640 < self.small_side
            if d[4] >= self.base or (self.small_t is not None and small and d[4] >= self.small_t):
                out.append(tuple(float(v) for v in d[:5]))
        return out

    def update(self, boxes, t, frame_shape):
        """boxes: [N, 5] x1, y1, x2, y2, score in frame pixels (decoded at a floor <= the small-box threshold to use
        the small-box gate). Returns dict(warning, alarm, flag, tracks=[dict(id, box, score, last_seen, confirmed,
        hit_ratio)])."""
        to640 = 640 / max(frame_shape[:2])
        self.tracks = {k: v for k, v in self.tracks.items() if t - v['last'] <= self.drop_s}
        used, assigned = set(), []
        for d in sorted(self.gate(np.asarray(boxes, float).reshape(-1, 5), to640), key=lambda d: -d[4]):
            best = None
            for k, v in self.tracks.items():
                if k in used:
                    continue
                b = v['box']
                overlap = iou(b, d)
                side = max(d[2] - d[0], d[3] - d[1], b[2] - b[0], b[3] - b[1])
                dist = math.hypot((b[0] + b[2] - d[0] - d[2]) / 2, (b[1] + b[3] - d[1] - d[3]) / 2)
                if overlap >= self.match_iou or dist <= self.match_centre * side:
                    key = (overlap, -dist)
                    if best is None or key > best[0]:
                        best = (key, k)
            if best is None:
                k, self.next_id = self.next_id, self.next_id + 1
                self.tracks[k] = {}
            else:
                k = best[1]
            used.add(k)
            self.tracks[k].update(box=d[:4], last=t, score=d[4])
            assigned.append((k, d[4]))
        # raw flag over the last window_s seconds (time based; a part-filled window never counts as 100%)
        self.times.append(t)
        if self.frame_dt:
            dt = self.frame_dt
        else:
            dt = float(np.median(np.diff(self.times[-200:]))) if len(self.times) > 1 else .1
        expected = max(1, round(self.window_s / dt))
        self.window.append((t, assigned))
        while self.window and self.window[0][0] <= t - self.window_s + 1e-9:
            self.window.popleft()
        n = max(len(self.window), expected)
        hits = defaultdict(list)
        for _, aa in self.window:
            for k, s in dict(aa).items():
                hits[k].append(s)
        ratio = {k: len(v) / n for k, v in hits.items()}
        passing = {k for k, v in hits.items() if len(v) / n >= self.min_hit and float(np.mean(v)) >= self.min_mean}
        flag = bool(passing)
        self.confirmed |= passing
        # alarm with hysteresis
        if flag:
            self._run_start = t if self._run_start is None else self._run_start
            self._last_flag = t
        else:
            self._run_start = None
        if not self.alarm and flag and t - self._run_start >= self.on_after_s - 1e-9:
            self.alarm = True
        elif self.alarm and (self._last_flag is None or t - self._last_flag >= self.off_after_s - 1e-9):
            self.alarm = False
        self.confirmed &= set(self.tracks) | passing
        tracks = [dict(id=k, box=v['box'], score=v['score'], last_seen=v['last'], confirmed=k in self.confirmed,
                       hit_ratio=ratio.get(k, 0.)) for k, v in self.tracks.items()]
        return dict(warning=bool(assigned), alarm=self.alarm, flag=flag, tracks=tracks, assigned=assigned)
