"""Lightweight tracking and multi-signal alarm verification around frame-level flame detections.

NumPy/OpenCV only. Raw detector boxes are never changed. For every frame the verifier reports, per track:
- persistent: matched in the CURRENT frame and in at least K of the last N processed frames;
- fused: persistent AND every enabled rule passes (rules are off unless a threshold is given).
A held track (temporarily missed) keeps its identity but can never be persistent or fused on its own.
No measurement here establishes that a region is real fire; thresholds must come from labelled recordings.
"""
import collections
import math

import cv2
import numpy as np

ROI_SIZE = 32  # every ROI is resized to ROI_SIZE x ROI_SIZE before any statistic
BRIGHT_VALUE = 220  # HSV value at or above which a pixel counts as bright
AREA_RATIO = 4.0  # distance fallback: the larger box may be at most this many times the smaller one's area


def iou_matrix(a, b):
    """IoU between every xyxy row of a and every xyxy row of b."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None, :4], np.asarray(b, float)[None, :, :4]
    width = (np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])).clip(0)
    height = (np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])).clip(0)
    intersection = width * height
    area = lambda x: (x[..., 2]-x[..., 0]) * (x[..., 3]-x[..., 1])
    return intersection / np.maximum(area(a) + area(b) - intersection, 1e-9)


def match(track_boxes, detections, threshold, distance=0.0):
    """Greedy one-to-one association; returns [(track index, detection index)].

    First pass: descending IoU, pairs with IoU >= threshold. Second pass (distance > 0), only for what is left:
    ascending centre distance, measured in units of the track box's longer side, pairs within `distance` whose
    areas differ by at most AREA_RATIO. This keeps a small, fast-moving object (a waved match moves more than
    its own width between processed frames, so IoU is 0) on one track.
    """
    overlap = iou_matrix(track_boxes, detections)
    pairs, used_tracks, used_detections = [], set(), set()
    for flat in np.argsort(-overlap, axis=None):
        t, d = np.unravel_index(flat, overlap.shape)
        if overlap[t, d] < threshold:
            break
        if t not in used_tracks and d not in used_detections:
            pairs.append((int(t), int(d)))
            used_tracks.add(t)
            used_detections.add(d)
    if distance > 0 and overlap.size:
        a, b = np.asarray(track_boxes, float)[:, None, :4], np.asarray(detections, float)[None, :, :4]
        centre = lambda x: np.stack([(x[..., 0]+x[..., 2])/2, (x[..., 1]+x[..., 3])/2], -1)
        size = np.maximum(a[..., 2]-a[..., 0], a[..., 3]-a[..., 1]).clip(1e-9)
        gap = np.linalg.norm(centre(a) - centre(b), axis=-1) / size
        area = lambda x: ((x[..., 2]-x[..., 0]) * (x[..., 3]-x[..., 1])).clip(1e-9)
        ratio = np.maximum(area(a) / area(b), area(b) / area(a))
        gap = np.where(ratio <= AREA_RATIO, gap, np.inf)
        for flat in np.argsort(gap, axis=None):
            t, d = np.unravel_index(flat, gap.shape)
            if gap[t, d] > distance:
                break
            if t not in used_tracks and d not in used_detections:
                pairs.append((int(t), int(d)))
                used_tracks.add(t)
                used_detections.add(d)
    return pairs


def circular_hue(hues):
    """Circular mean and standard deviation of OpenCV hues (0-179), in hue units; red wraps around 0."""
    vector = np.exp(1j * np.asarray(hues, float) * (2 * math.pi / 180)).mean()
    mean = (math.degrees(math.atan2(vector.imag, vector.real)) / 2) % 180
    length = min(max(abs(vector), 1e-12), 1.0)  # rounding can push a uniform hue's length just above 1
    std = math.sqrt(-2 * math.log(length)) * 180 / (2 * math.pi)
    return float(mean), float(std)


def roi_features(frame, box):
    """Appearance of one box from the undrawn BGR frame; None for an empty, invalid or off-frame ROI."""
    if frame is None or not np.isfinite(box[:4]).all():
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    x2, y2 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = cv2.resize(frame[y1:y2, x1:x2], (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return dict(grey=grey, grey_mean=float(grey.mean()), grey_var=float(grey.var()),
                hue_mean=circular_hue(hsv[..., 0].ravel())[0], saturation_mean=float(hsv[..., 1].mean()),
                value_mean=float(hsv[..., 2].mean()), bright_fraction=float((hsv[..., 2] >= BRIGHT_VALUE).mean()))


def relative_spread(values):
    values = np.asarray(values, float)
    return float(values.std() / max(abs(values.mean()), 1e-9))


class Track:
    def __init__(self, track_id, window):
        self.id, self.window = track_id, window
        self.hits = collections.deque(maxlen=window)  # True where matched, one entry per processed frame
        self.observations = collections.deque(maxlen=window)  # matched frames only, bounded
        self.frame = 0  # processed frames since the track started
        self.misses = 0  # consecutive frames without a match
        self.box = None

    @property
    def matched(self):
        return self.misses == 0

    def observe(self, box, frame_image, time):
        features = roi_features(frame_image, box)
        previous = self.observations[-1]['features'] if self.observations else None
        change = None
        if features is not None and previous is not None:
            # Mean absolute grey change between this ROI and the track's previous matched ROI, 0..1.
            change = float(np.abs(features['grey'].astype(np.int16) - previous['grey'].astype(np.int16)).mean() / 255)
        self.frame += 1
        self.observations.append(dict(box=np.asarray(box[:4], float), score=float(box[4]), features=features,
                                      change=change, time=time, frame=self.frame))
        self.hits.append(True)
        self.misses = 0
        self.box = np.asarray(box[:4], float)

    def miss(self):
        self.frame += 1
        self.hits.append(False)
        self.misses += 1

    def statistics(self):
        """Statistics over matched observations inside the last N processed frames; None where not computable."""
        obs = [o for o in self.observations if o['frame'] > self.frame - self.window]
        stats = dict.fromkeys(STATISTICS)
        stats.update(hits=int(sum(self.hits)), window=len(self.hits), observations=len(obs))
        if not obs:
            return stats
        boxes = np.array([o['box'] for o in obs])
        widths, heights = boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]
        areas, aspects = widths * heights, widths / np.maximum(heights, 1e-9)
        centres = (boxes[:, :2] + boxes[:, 2:]) / 2
        stats.update(score_mean=float(np.mean([o['score'] for o in obs])),
                     score_min=float(np.min([o['score'] for o in obs])),
                     area_mean=float(areas.mean()), aspect_mean=float(aspects.mean()))
        if len(obs) > 1:
            steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
            stats.update(area_cv=relative_spread(areas), aspect_cv=relative_spread(aspects),
                         motion_px=float(steps.mean()),
                         motion_relative=float(steps.mean() / max(math.sqrt(areas.mean()), 1e-9)),
                         displacement_px=float(np.linalg.norm(centres[-1] - centres[0])))
        changes = [o['change'] for o in obs if o['change'] is not None]
        if changes:
            stats['flicker'] = float(np.mean(changes))
        features = [o['features'] for o in obs if o['features'] is not None]
        if features:
            grey_means = [f['grey_mean'] for f in features]
            hue_mean, hue_std = circular_hue([f['hue_mean'] for f in features])
            stats.update(grey_mean=float(np.mean(grey_means)), grey_var_mean=float(np.mean([f['grey_var'] for f in features])),
                         hue_mean=hue_mean, saturation_mean=float(np.mean([f['saturation_mean'] for f in features])),
                         value_mean=float(np.mean([f['value_mean'] for f in features])),
                         bright_fraction=float(np.mean([f['bright_fraction'] for f in features])))
            if len(features) > 1:
                stats.update(intensity_cv=relative_spread(grey_means), hue_std=hue_std,
                             saturation_std=float(np.std([f['saturation_mean'] for f in features])),
                             value_std=float(np.std([f['value_mean'] for f in features])))
        return stats


STATISTICS = ['hits', 'window', 'observations', 'score_mean', 'score_min', 'area_mean', 'aspect_mean', 'area_cv',
              'aspect_cv', 'motion_px', 'motion_relative', 'displacement_px', 'flicker', 'intensity_cv', 'grey_mean',
              'grey_var_mean', 'hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean', 'value_std',
              'bright_fraction']

# Rule name -> (statistic, comparison). Every threshold comes from the caller; None disables a rule.
RULES = {'min_score': ('score_mean', '>='), 'min_flicker': ('flicker', '>='),
         'min_intensity_cv': ('intensity_cv', '>='), 'min_area_cv': ('area_cv', '>='),
         'min_aspect_cv': ('aspect_cv', '>='), 'max_motion_relative': ('motion_relative', '<='),
         'min_bright_fraction': ('bright_fraction', '>='), 'max_saturation': ('saturation_mean', '<=')}


class Verifier:
    """Track detections; report persistence (K-of-N, current match) and fused (persistence AND rules) per track.

    A rule whose statistic cannot be computed yet fails, so a track is only fused on evidence actually measured.
    """
    def __init__(self, min_hits, window, iou=0.1, max_missed=None, rules=None, distance=0.0):
        if not 1 <= min_hits <= window:
            raise ValueError('Verifier needs 1 <= min_hits <= window')
        self.min_hits, self.window, self.iou, self.distance = min_hits, window, iou, distance
        # Default: drop a track once it cannot reach min_hits without fresh matches.
        self.max_missed = max(1, window - min_hits) if max_missed is None else max_missed
        if self.max_missed < 0:
            raise ValueError('max_missed must be >= 0')
        self.rules = {name: value for name, value in (rules or {}).items() if value is not None}
        unknown = set(self.rules) - set(RULES)
        if unknown:
            raise ValueError(f'Unknown verification rules: {sorted(unknown)}')
        self.tracks = []
        self.next_id = 1

    def update(self, boxes, frame=None, time=None):
        """boxes: [M, 5] xyxy+score for one processed frame. Returns one report per live track."""
        boxes = np.asarray(boxes, float).reshape(-1, 5)
        pairs = match([t.box for t in self.tracks], boxes[:, :4], self.iou, self.distance)
        matched = dict(pairs)
        for index, track in enumerate(self.tracks):
            if index in matched:
                track.observe(boxes[matched[index]], frame, time)
            else:
                track.miss()
        for d in sorted(set(range(len(boxes))) - set(matched.values())):
            track = Track(self.next_id, self.window)
            self.next_id += 1
            track.observe(boxes[d], frame, time)
            self.tracks.append(track)
        self.tracks = [t for t in self.tracks if t.misses <= self.max_missed]
        reports = []
        for track in self.tracks:
            stats = track.statistics()
            persistent = track.matched and stats['hits'] >= self.min_hits
            checks = {}
            for name, threshold in self.rules.items():
                value = stats[RULES[name][0]]
                checks[name] = value is not None and (value >= threshold if RULES[name][1] == '>=' else value <= threshold)
            reports.append(dict(track_id=track.id, box=track.box.tolist(), matched=track.matched, misses=track.misses,
                                persistent=persistent, fused=persistent and all(checks.values()),
                                checks=checks, stats=stats))
        return reports
