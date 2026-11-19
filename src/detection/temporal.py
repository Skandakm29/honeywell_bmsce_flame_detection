"""The locked Week 2 temporal alarm rule, fed one frame at a time (for live cameras).

Same rule and same numbers as experiments/temporal_final/temporal_final.py (track / raw_flags / alarm), which
works on a whole recorded clip at once. Nothing is re-tuned; the settings come from rule_v12.json:
  - tracking by location: a box joins a track if IoU >= 0.1 or its centre is within 1.5 x the larger box side;
    a track is dropped after 1 s without a box
  - window: for each track, hit ratio = frames with a box in the last 2 s / max(frames in window, 2 s x fps)
  - ALARM ON when some track has hit ratio >= 0.30 (and mean score >= min_mean_score)
  - ALARM OFF after 3 s with no qualifying track
  - WARNING = any (already thresholded) box in this frame
Boxes passed in must already be thresholded by the detection stage (full frame 0.40, zoom 0.20).
tests/test_temporal_stream.py and the W3 Phase A check show it gives the same ALARM frames as the batch code.
"""
import json
import math
from collections import defaultdict, deque
from statistics import median

TRACK_GAP_S, OFF_S, MATCH_IOU, MATCH_CENTRE = 1.0, 3.0, 0.1, 1.5


def _iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


class AlarmRule:
    def __init__(self, window_s=2.0, min_hit_ratio=0.30, min_mean_score=0.0, alarm_on_after_s=0.0,
                 alarm_off_after_s=OFF_S, track_gap_s=TRACK_GAP_S, match_iou=MATCH_IOU, match_centre=MATCH_CENTRE):
        self.W, self.h, self.m, self.hold = window_s, min_hit_ratio, min_mean_score, alarm_on_after_s
        self.off_s, self.gap, self.match_iou, self.match_centre = alarm_off_after_s, track_gap_s, match_iou, match_centre
        self.tracks, self.next_id = {}, 0
        self.window = deque()
        self.dts = deque(maxlen=50)
        self.last_t = None
        self.on, self.run_start, self.last_true = False, None, None

    @classmethod
    def from_json(cls, path):
        r = json.load(open(path, encoding='utf-8'))
        return cls(window_s=r['window_s'], min_hit_ratio=r['min_hit_ratio'], min_mean_score=r['min_mean_score'],
                   alarm_on_after_s=r['alarm_on_after_s'], alarm_off_after_s=r.get('alarm_off_after_s', OFF_S),
                   track_gap_s=r.get('tracker', {}).get('drop_after_s', TRACK_GAP_S),
                   match_iou=r.get('tracker', {}).get('match_iou', MATCH_IOU),
                   match_centre=r.get('tracker', {}).get('match_centre_x_longest_side', MATCH_CENTRE))

    def _track(self, t, boxes):
        self.tracks = {k: v for k, v in self.tracks.items() if t - v['last'] <= self.gap}
        used, assigned = set(), []
        for d in sorted(boxes, key=lambda d: -d[4]):
            best = None
            for k, v in self.tracks.items():
                if k in used:
                    continue
                b = v['box']
                overlap = _iou(b, d)
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
            self.tracks[k].update(box=list(d[:4]), last=t)
            assigned.append((k, float(d[4])))
        return assigned

    def update(self, t, boxes):
        """t: seconds (video time or a monotonic clock); boxes: [[x1, y1, x2, y2, score], ...] already thresholded.
        Returns dict(state='CLEAR'|'WARNING'|'ALARM', alarm=bool, warning=bool, best_hit_ratio=float)."""
        boxes = [list(map(float, b[:5])) for b in boxes]
        if self.last_t is not None:
            self.dts.append(t - self.last_t)
        self.last_t = t
        assigned = self._track(t, boxes)
        dt = median(self.dts) if self.dts else .1
        expected = max(1, round(self.W / dt)) if dt > 0 else 1
        self.window.append((t, assigned))
        while self.window and self.window[0][0] <= t - self.W + 1e-9:
            self.window.popleft()
        n = max(len(self.window), expected)
        hits = defaultdict(list)
        for _, aa in self.window:
            for k, s in dict(aa).items():
                hits[k].append(s)
        ratios = [len(v) / n for v in hits.values() if sum(v) / len(v) >= self.m]
        flag = any(r >= self.h for r in ratios)
        if flag:
            self.run_start = t if self.run_start is None else self.run_start
            self.last_true = t
        else:
            self.run_start = None
        if not self.on and flag and t - self.run_start >= self.hold - 1e-9:
            self.on = True
        elif self.on and (self.last_true is None or t - self.last_true >= self.off_s - 1e-9):
            self.on = False
        warning = bool(boxes)
        return dict(state='ALARM' if self.on else ('WARNING' if warning else 'CLEAR'), alarm=self.on,
                    warning=warning, best_hit_ratio=max(ratios, default=0.0))
