"""Camera/video viewer for our one-class TI YOLOX and YOLOv9 ONNX exports."""
import argparse
import collections
import contextlib
import csv
import hashlib
import json
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from flame_verification import RULES, STATISTICS, Verifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'results/lab_backup_20261031/outputs/validation/export/flame.onnx'


def prepare(frame, size=640, architecture='yolov9'):
    h, w = frame.shape[:2]
    ratio = min(size / w, size / h)
    rounding = round if architecture == 'yolov9' else int
    width, height = max(1, rounding(w * ratio)), max(1, rounding(h * ratio))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:height, :width] = cv2.resize(frame, (width, height))
    if architecture == 'yolov9':
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        scale = np.array([width/w, height/h, width/w, height/h])
    else:
        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        scale = np.full(4, ratio)
    return np.ascontiguousarray(tensor), scale


def decode(output, frame, scale, confidence, architecture='yolov9'):
    rows = output[0]
    fields = 5 if architecture == 'yolov9' else 6
    if rows.ndim != 2 or rows.shape[1] != fields or not np.isfinite(rows).all():
        raise ValueError(f'Expected finite [1, N, {fields}] output for {architecture}; check the model contract.')
    if architecture != 'yolov9':
        rows = np.column_stack((rows[:, :2] - rows[:, 2:4]/2,
                                rows[:, :2] + rows[:, 2:4]/2, rows[:, 4]*rows[:, 5]))
    rows = rows[rows[:, 4] >= confidence].copy()
    if not len(rows):
        return rows
    h, w = frame.shape[:2]
    if architecture == 'yolov9':
        rows[:, :4] /= scale
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, w)
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, h)
    rows = rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]
    if not len(rows):
        return rows
    xywh = rows[:, :4].copy()
    xywh[:, 2:] -= xywh[:, :2]
    selected = cv2.dnn.NMSBoxes(xywh.tolist(), rows[:, 4].tolist(), 0.0, 0.65)
    rows = rows[np.asarray(selected, dtype=int).reshape(-1)[:100]]
    if architecture != 'yolov9':
        rows[:, :4] /= scale
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, w)
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, h)
    return rows


def decode_classes(output, frame, scale, confidences):
    """Week 4 two-class YOLOv9 export [1, N, 4 + C] (flame, obstruction): per-class threshold and NMS.
    Returns {class index: [M, 5] array of x1, y1, x2, y2, score} in frame pixels."""
    rows = output[0]
    if rows.ndim != 2 or rows.shape[1] != 4 + len(confidences) or not np.isfinite(rows).all():
        raise ValueError(f'Expected finite [1, N, {4 + len(confidences)}] output; check the model contract.')
    return {c: decode(np.concatenate([rows[:, :4], rows[:, 4 + c:5 + c]], 1)[None], frame, scale, threshold)
            for c, threshold in enumerate(confidences)}


def blind_percent(boxes, frame, grid=160):
    """Share of the frame covered by the union of the obstruction boxes, in percent."""
    h, w = frame.shape[:2]
    gh = max(1, round(grid * h / w))
    mask = np.zeros((gh, grid), bool)
    for x1, y1, x2, y2 in boxes[:, :4]:
        mask[int(y1 / h * gh):int(np.ceil(y2 / h * gh)), int(x1 / w * grid):int(np.ceil(x2 / w * grid))] = True
    return 100. * mask.mean()


def overlaps(box, boxes):
    """IoU of one xyxy box with each row of boxes."""
    width = (np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])).clip(0)
    height = (np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])).clip(0)
    intersection = width * height
    union = (box[2]-box[0])*(box[3]-box[1]) + (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) - intersection
    return intersection / np.maximum(union, 1e-9)


class Persistence:
    """Alarm only when a current box recurs (IoU >= 0.1) in at least k of the last n processed frames.

    Counts processed frames, not seconds: the live camera keeps only the latest frame, so the window
    lasts n / processing FPS. Suppresses brief detections; a static look-alike that is detected in
    every frame still alarms.
    """
    def __init__(self, k, n, overlap=0.1):
        self.k, self.overlap = k, overlap
        self.history = collections.deque(maxlen=n)

    def update(self, boxes):
        self.history.append(boxes[:, :4].copy())
        support = max((sum(bool(len(past)) and overlaps(box, past).max() >= self.overlap
                           for past in self.history) for box in boxes[:, :4]), default=0)
        return bool(support >= self.k), int(support)


VERIFICATION_COLUMNS = ['verification_mode', 'verify_ms', 'current_tracks', 'alarm_track_id', 'alarm_track_support',
                        'alarm_track_persistent', 'alarm_track_score_mean', 'alarm_track_flicker',
                        'alarm_track_area_cv', 'alarm_track_aspect_cv', 'alarm_track_motion_px',
                        'persistent_alarm', 'fused_alarm', 'verified_alarm']
TRACK_COLUMNS = ['camera_sequence', 'video_seconds', 'track_id', 'matched', 'misses', 'box_xyxy',
                 'persistent', 'fused', 'checks', *STATISTICS]


def decision(report, mode):
    """The alarm decision of one track: fused mode adds the rules; persistence/diagnostic do not."""
    return report['fused'] if mode == 'fused' else report['persistent']


def verification_row(reports, mode, milliseconds):
    """Frame-level summary appended to frames.csv; describes the deciding (or strongest current) track."""
    current = [r for r in reports if r['matched']]
    rounded = lambda v: '' if v is None else round(v, 4)
    if not current:
        return [mode, milliseconds, 0, *[''] * 8, 0, 0, 0]
    top = max(current, key=lambda r: (decision(r, mode), r['persistent'], r['stats']['hits'], r['stats']['score_mean']))
    s = top['stats']
    return [mode, milliseconds, len(current), top['track_id'], s['hits'], int(top['persistent']), rounded(s['score_mean']),
            rounded(s['flicker']), rounded(s['area_cv']), rounded(s['aspect_cv']), rounded(s['motion_px']),
            int(any(r['persistent'] for r in current)), int(any(r['fused'] for r in current)),
            int(any(decision(r, mode) for r in current))]


def draw_verification(frame, reports, mode, args, milliseconds, line_y=96):
    """Verified alarms: thick double white/magenta frame. Raw detector boxes stay thin red, drawn separately."""
    alarms = [r for r in reports if r['matched'] and decision(r, mode)]
    for r in reports:
        x1, y1, x2, y2 = map(int, r['box'])
        if r in alarms:
            cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (255, 255, 255), 4)
            cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (255, 0, 255), 2)
        s = r['stats']
        text = f"T{r['track_id']} {s['hits']}/{args.window}" + ('' if r['matched'] else ' (missed)')
        if mode in ('diagnostic', 'fused'):
            value = lambda v, f='.2f': '-' if v is None else format(v, f)
            text += f" fl {value(s['flicker'])} mv {value(s['motion_px'], '.0f')}px"
            failed = [name for name, ok in r['checks'].items() if not ok]
            text += (' rules OK' if not failed else ' fail:' + ','.join(failed)) if r['checks'] else ''
        colour = (255, 0, 255) if r in alarms else (200, 200, 200)
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
        position = (max(0, min(x1, frame.shape[1] - width)), min(frame.shape[0]-6, y2+18))
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1)
    line = (f'VERIFIED ALARM ({mode}) | tracks {[r["track_id"] for r in alarms]}' if alarms
            else f'no verified alarm ({mode}) | needs {args.min_hits}/{args.window} | verify {milliseconds} ms')
    cv2.putText(frame, line, (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 0, 255) if alarms else (0, 255, 0), 2)


class Video:
    """Read every recorded frame in order; replay speed follows inference speed."""
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f'Cannot open video {path}')
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

    def next(self, previous):
        ok, frame = self.capture.read()
        if not ok:
            return None
        return previous + 1, frame

    def close(self):
        self.capture.release()


class Camera:
    """Continuously drain the camera and keep only the latest frame."""
    def __init__(self, source, backend):
        self.capture = cv2.VideoCapture(source, backend)
        if not self.capture.isOpened():
            raise RuntimeError(f'Cannot open camera {source}. Close other camera apps, allow camera '
                               'access for desktop apps, or try --source 1 / --backend dshow.')
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.latest = None
        self.sequence = 0
        self.failed = False
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.stopping.is_set():
            ok, frame = self.capture.read()
            with self.condition:
                if not ok:
                    self.failed = True
                    self.condition.notify_all()
                    return
                self.latest = frame
                self.sequence += 1
                self.condition.notify_all()

    def next(self, previous):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence != previous or self.failed, timeout=15)
            if not ready or self.latest is None or self.failed:
                raise RuntimeError('No camera frames. Check the webcam/phone connection and permissions.')
            return self.sequence, self.latest.copy()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=1)
        self.capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0', help='Camera index, or a camera stream URL')
    parser.add_argument('--arch', choices=['yolov9', 'nano', 's'], default='yolov9')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--confidence', type=float, default=0.40)
    parser.add_argument('--obstruction-confidence', type=float, default=0.40,
                        help='Only for a two-class (flame + obstruction) YOLOv9 export')
    parser.add_argument('--backend', choices=['auto', 'dshow', 'msmf'], default='auto')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--headless', action='store_true', help='Process a saved video without a display')
    parser.add_argument('--persistence', type=int, nargs=2, metavar=('K', 'N'),
                        help='Alarm only if a box recurs in K of the last N processed frames (off by default)')
    verify = parser.add_argument_group('track verification (see docs/TEMPORAL_VERIFICATION.md)')
    verify.add_argument('--verification', choices=['off', 'persistence', 'diagnostic', 'fused'], default='off',
                        help='off: original behaviour. persistence: tracked K-of-N alarm. diagnostic: persistence '
                             'alarm plus all features and rule results shown, never used to reject. fused: '
                             'persistence AND every given rule')
    verify.add_argument('--window', type=int, default=6, help='N: processed frames in the persistence window')
    verify.add_argument('--min-hits', type=int, default=4, help='K: matched frames required within the window')
    verify.add_argument('--track-iou', type=float, default=0.1, help='Minimum IoU to continue a track')
    verify.add_argument('--track-distance', type=float, default=3.0,
                        help='If no IoU match: continue a track when the centre moved at most this many track-box '
                             'lengths (areas within 4x). Keeps a waved match on one track. 0 = IoU only')
    verify.add_argument('--max-missed', type=int,
                        help='Drop a track after this many consecutive misses (default window - min-hits, at least 1)')
    for name, (statistic, comparison) in RULES.items():
        verify.add_argument('--' + name.replace('_', '-'), type=float,
                            help=f'Rule: track {statistic} {comparison} value. Uncalibrated; decides only in fused '
                                 'mode, shown only in diagnostic mode')
    zone = parser.add_argument_group('zone obstruction monitor (src/obstruction; image-space %, no floor m2)')
    zone.add_argument('--zones', type=Path, help='Zone config JSON, e.g. configs/zones/webcam_test.json. Off by '
                      'default; flame detection, logging and alarms are unchanged either way')
    zone.add_argument('--zone-learn-s', type=float, default=3., help='Seconds of still, empty scene to learn')
    zone.add_argument('--zone-confirm', action='store_true',
                      help='Live: arm the zone monitor only after the operator checks the learned empty scene (C '
                           'confirms, R re-learns). The learned scene is always saved as zone_reference_<n>.png')
    zone.add_argument('--zone-method', choices=('pixel', 'deep', 'fused'), default='pixel',
                      help='Change detection: pixel (grey + edges, default), deep (DINOv2 patch features), fused '
                           '(pixel AND deep). deep/fused need models/dinov2/dinov2_vits14_252x448.onnx')
    zone.add_argument('--zone-deep-every-s', type=float, default=1.,
                      help='Seconds between DINOv2 feature maps (about 100 ms each on a laptop CPU). Measured with '
                           'V12: 1 s and 1 thread slow flame detection by ~27%%; 0.2 s on all cores by ~3x')
    zone.add_argument('--zone-deep-threads', type=int, default=1, help='CPU threads for DINOv2 (0 = all)')
    zone.add_argument('--zone-deep-threshold', type=float, default=.2,
                      help='Minimum patch change score (1 - cosine similarity); uncalibrated on real scenes')
    zone.add_argument('--zone-deep-sync', action='store_true',
                      help='Run DINOv2 in the main loop (reproducible video runs). Default for a live camera: '
                           'a background thread, so flame detection never waits')
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error('--confidence must be between 0 and 1')
    if args.persistence and not 1 <= args.persistence[0] <= args.persistence[1]:
        parser.error('--persistence needs 1 <= K <= N')
    rules = {name: getattr(args, name) for name in RULES if getattr(args, name) is not None}
    if args.verification != 'off' and not 1 <= args.min_hits <= args.window:
        parser.error('--verification needs 1 <= --min-hits <= --window')
    if args.track_distance < 0:
        parser.error('--track-distance must be >= 0')
    if rules and args.verification not in ('diagnostic', 'fused'):
        parser.error('rule thresholds need --verification diagnostic or fused')
    if args.verification == 'fused' and not rules:
        parser.error('--verification fused needs at least one rule threshold (e.g. --min-flicker); '
                     'use --verification persistence for persistence alone')
    if not args.model.is_file():
        parser.error(f'Model missing: {args.model}. Supply --model with the actual flame.onnx path.')
    cv2.setNumThreads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    size = 416 if args.arch == 'nano' else 640
    if session.get_inputs()[0].shape != [1, 3, size, size]:
        raise ValueError(f'{args.arch} requires our fixed {size}x{size} flame export.')
    # Week 4: a YOLOv9 export with 6 output fields is the two-class model (flame, obstruction).
    two_class = args.arch == 'yolov9' and session.get_outputs()[0].shape[-1] == 6
    if two_class and not 0 < args.obstruction_confidence < 1:
        parser.error('--obstruction-confidence must be between 0 and 1')
    confidences = [args.confidence, args.obstruction_confidence]

    def detect(raw, frame, scale):
        if not two_class:
            return decode(raw, frame, scale, args.confidence, args.arch), np.zeros((0, 5))
        found = decode_classes(raw, frame, scale, confidences)
        return found[0], found[1]

    tensor, scale = prepare(np.zeros((480, 640, 3), dtype=np.uint8), size, args.arch)
    warmup = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    detect(warmup, np.zeros((480, 640, 3), dtype=np.uint8), scale)
    source = int(args.source) if args.source.isdecimal() else args.source
    backend = {'auto': cv2.CAP_ANY, 'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF}[args.backend]
    is_video = isinstance(source, str) and Path(source).is_file()
    if args.headless and not is_video:
        parser.error('--headless requires a saved video file')
    output = args.output or ROOT / 'results/webcam' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'frames.csv').exists():
        parser.error('Output already contains frames.csv; choose a new output folder.')
    metadata = dict(model=str(args.model.resolve()), confidence=args.confidence,
                    architecture=args.arch, model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    classes=['flame', 'obstruction'] if two_class else ['flame'],
                    obstruction_confidence=args.obstruction_confidence if two_class else None,
                    source_kind='video' if is_video else 'camera',
                    source=str(source), backend=args.backend, provider='CPUExecutionProvider',
                    platform=platform.platform(), onnxruntime=ort.__version__,
                    persistence=dict(k=args.persistence[0], n=args.persistence[1], iou=0.1,
                                     window='processed frames, not seconds') if args.persistence else None,
                    note='Frame detections only; no alarm verification or ground-truth accuracy metrics. '
                         'CPU processing FPS is not camera FPS or AM62A performance.')
    persistence = Persistence(*args.persistence) if args.persistence else None
    verifier = None
    if args.verification != 'off':
        verifier = Verifier(args.min_hits, args.window, iou=args.track_iou, max_missed=args.max_missed, rules=rules,
                            distance=args.track_distance)
        metadata['verification'] = dict(mode=args.verification, min_hits=args.min_hits, window=args.window,
                                        track_iou=args.track_iou, track_distance=args.track_distance,
                                        max_missed=verifier.max_missed, rules=rules,
                                        rules_decide=args.verification == 'fused',
                                        window_unit='processed frames, not seconds')
    monitor = None
    if args.zones:
        # Imported only when asked for, so the default viewer does not depend on the zone code.
        sys.path.insert(0, str(ROOT / 'src'))
        from obstruction import ReferenceDetector, SceneMonitor, load_zones
        from obstruction.deep import DEFAULT_MODEL as DEEP_MODEL, DeepChangeDetector, OnnxPatchFeatures
        from obstruction.overlay import draw_lines, draw_visibility, draw_zones, zone_lines
        from flame_alarm import RULE_FILE, FlameAlarm
        zone_config = load_zones(args.zones)
        pixel = ReferenceDetector(learn_s=args.zone_learn_s)
        detector = pixel if args.zone_method == 'pixel' else DeepChangeDetector(
            OnnxPatchFeatures(threads=args.zone_deep_threads),
            method=args.zone_method, pixel=pixel, every_s=args.zone_deep_every_s,
            min_threshold=args.zone_deep_threshold, background=not (args.zone_deep_sync or is_video))
        if args.zone_confirm and args.headless:
            parser.error('--zone-confirm needs the live window (C to confirm)')
        monitor = SceneMonitor(zone_config, detector, require_confirmation=args.zone_confirm)
        # The Week 2 locked flame alarm, live: the visibility check follows the fires it confirms. It reads boxes
        # decoded at the small-box floor (0.25); the boxes drawn and logged above stay at --confidence.
        week2 = FlameAlarm()
        week2_floor = min(args.confidence, week2.small_t if week2.small_t is not None else week2.base)
        d = pixel
        metadata['zones'] = dict(
            file=str(args.zones.resolve()), grid=list(zone_config.grid),
            zones=[dict(name=z.name, polygon=z.polygon, min_blocked_pct=z.min_blocked_pct, persist_s=z.persist_s,
                        clear_s=z.clear_s) for z in zone_config.zones], ignore=zone_config.ignore,
            detector=dict(analysis_width=d.width, learn_s=d.learn_s, adapt_s=d.adapt_s,
                          grey_threshold=d.grey_threshold, edge_threshold=d.edge_threshold),
            method=args.zone_method,
            deep=None if args.zone_method == 'pixel' else dict(
                model=str(DEEP_MODEL), every_s=detector.every_s, min_threshold=detector.min_threshold,
                factor=detector.factor, sample_s=detector.sample_s, bank_update_s=detector.bank_update_s,
                threads=args.zone_deep_threads,
                background=detector.worker is not None),
            camera_motion=None if not monitor.motion else dict(
                shift_px_analysis=monitor.motion.shift_px, angle_deg=monitor.motion.angle_deg,
                scale_change=monitor.motion.scale_change, check_s=monitor.motion.check_s,
                note='ORB + RANSAC similarity vs the learned scene; moved -> VIEW CHANGED, obstruction paused'),
            ghost_correction=dict(ghost_s=d.ghost_s, outline_ratio=d.ghost_ratio, texture_match=d.ghost_texture),
            learn_timeout_s=d.learn_timeout_s, require_confirmation=args.zone_confirm,
            gap_s=monitor.gap_s, min_cell_fraction=monitor.min_cell_fraction,
            flame_exclusion='raw flame boxes grown by {:.0%} of their longer side'.format(monitor.exclude_margin),
            start_up_flame='flame areas while learning are unknown, filled after {:.0f} s flame-free and still'.format(
                d.fill_s),
            week2_alarm=dict(rule=str(RULE_FILE), decode_floor=week2_floor,
                             note='Week 2 locked rule, streaming; flame boxes drawn/logged are unchanged'),
            fire_visibility={k: getattr(monitor.visibility, k) for k in ('lost_s', 'cover_share', 'clear_s', 'grow')}
            if monitor.visibility else None,
            time_unit='seconds (video time, or since start live)', floor_area='not available (no Week 3 calibration)')
    camera = Video(source) if is_video else Camera(source, backend)
    if is_video:
        metadata['source_fps'] = camera.fps
        metadata['source_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    (output / 'session.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Press Q/Esc to quit; S to save the displayed screenshot.')
    print(f'Logs: {output}\nNo video is recorded automatically. Running on CPU.')
    sequence = 0
    window = f'Honeywell {args.arch} flame demo - Q quits, S saves'
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 720)
    try:
        with (output / 'frames.csv').open('w', newline='', encoding='utf-8') as file, \
                ((output / 'tracks.csv').open('w', newline='', encoding='utf-8') if verifier
                 else contextlib.nullcontext()) as track_file, \
                ((output / 'zone_events.csv').open('w', newline='', encoding='utf-8') if monitor
                 else contextlib.nullcontext()) as zone_file:
            writer = csv.writer(file)
            header = ['time_local', 'camera_sequence', 'processing_ms', 'detections_xyxy_score', 'video_seconds']
            writer.writerow(header + (['obstructions_xyxy_score', 'blind_percent'] if two_class else [])
                            + ['persistence_support', 'alarm', *VERIFICATION_COLUMNS]
                            + (['zone_state', 'zone_ms', 'zone_flame_excluded_pct', 'zone_frame_blocked_pct',
                                'zone_unknown_pct', 'week2_warning', 'week2_alarm', 'zone_fire_visibility_compromised',
                                'zone_fire_visibility_unknown', 'zone_fire_area_covered_pct',
                                'zone_view_changed', 'zone_camera_shift_px', 'zone_camera_angle_deg',
                                'zone_calibration_failed', 'zone_ghost_healed_pct',
                                'zone_deep_ms', 'zone_deep_max_score']
                               + [f'zone_{z.name}_{k}' for z in monitor.config.zones
                                  for k in ('blocked_pct', 'persistent_pct', 'alarm')] if monitor else []))
            if monitor:
                zone_writer = csv.writer(zone_file)
                zone_writer.writerow(['zone', 'start_s', 'end_s', 'duration_s', 'peak_pct', 'mean_pct'])
                write_zone_events = lambda events: [zone_writer.writerow(
                    [e['zone'], round(e['start_s'], 3), round(e['end_s'], 3), round(e['end_s'] - e['start_s'], 3),
                     round(e['peak_pct'], 2), round(e['mean_pct'], 2)]) for e in events]
                ready_s, zone_t, zone_frame, references = None, 0., None, 0
                hidden_file = (output / 'fire_visibility_events.csv').open('w', newline='', encoding='utf-8')
                hidden_writer = csv.writer(hidden_file)
                hidden_writer.writerow(['week2_track_id', 'start_s', 'end_s', 'duration_s', 'reason',
                                        'fire_last_seen_s', 'last_box_xyxy'])
                write_hidden = lambda events: [hidden_writer.writerow(
                    [e['track_id'], round(e['start_s'], 3), round(e['end_s'], 3), round(e['end_s'] - e['start_s'], 3),
                     e['reason'], round(e['last_seen_s'], 3), json.dumps(e['box'])]) for e in events]
            if verifier:
                track_writer = csv.writer(track_file)
                track_writer.writerow(TRACK_COLUMNS)
            clock = time.monotonic()
            while True:
                item = camera.next(sequence)
                if item is None:
                    break
                sequence, frame = item
                started = time.perf_counter()
                tensor, scale = prepare(frame, size, args.arch)
                raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                boxes, obstructions = detect(raw, frame, scale)
                milliseconds = (time.perf_counter() - started) * 1000
                blind = blind_percent(obstructions, frame) if two_class else 0.
                alarm, support = persistence.update(boxes) if persistence else (None, '')
                video_seconds = (sequence-1)/camera.fps if is_video and camera.fps > 0 else ''
                reports, verify_ms, summary = [], '', [''] * len(VERIFICATION_COLUMNS)
                if verifier:
                    checked = time.perf_counter()
                    # ROI statistics use the undrawn frame; time is video time, or seconds since start live.
                    reports = verifier.update(boxes, frame, video_seconds if is_video else time.monotonic()-clock)
                    verify_ms = round((time.perf_counter() - checked) * 1000, 3)
                    summary = verification_row(reports, args.verification, verify_ms)
                    for r in reports:
                        track_writer.writerow([sequence, video_seconds, r['track_id'], int(r['matched']), r['misses'],
                                               json.dumps([round(v, 1) for v in r['box']]), int(r['persistent']),
                                               int(r['fused']), json.dumps(r['checks']),
                                               *[r['stats'][k] for k in STATISTICS]])
                zone_row = []
                if monitor:
                    # Undrawn frame; flame boxes are excluded so a flame never counts as an obstruction.
                    zone_t = video_seconds if is_video else time.monotonic() - clock
                    alarm_boxes = (decode(raw, frame, scale, week2_floor, args.arch) if not two_class else boxes)
                    week2_out = week2.update(alarm_boxes, zone_t, frame.shape)
                    checked = time.perf_counter()
                    zone_out = monitor.update(frame, zone_t, exclude_boxes=boxes, flame=week2_out)
                    zone_frame = frame             # undrawn; kept for the blind-zone map image
                    zone_ms = round((time.perf_counter() - checked) * 1000, 3)
                    write_zone_events(zone_out['events'])
                    write_hidden(zone_out['visibility']['events'])
                    zone_file.flush()
                    hidden_file.flush()
                    learned = zone_out['state'] in ('ready', 'awaiting_confirmation', 'view_changed')
                    if learned and ready_s is None:     # just learned: save what was learned, for the record
                        references += 1
                        reference = monitor.reference_image()
                        if reference is not None:
                            cv2.imwrite(str(output / f'zone_reference_{references}.png'),
                                        cv2.resize(reference, frame.shape[1::-1], interpolation=cv2.INTER_NEAREST))
                    ready_s = (ready_s if ready_s is not None else zone_t) if learned else None
                    zone_row = [zone_out['state'], zone_ms, round(zone_out['excluded_pct'], 3),
                                round(zone_out['frame_pct'], 3), round(zone_out.get('unknown_pct', 0.), 3),
                                int(week2_out['warning']), int(week2_out['alarm']),
                                int(zone_out['visibility']['visibility_compromised']),
                                int(zone_out['visibility']['compromised_unknown']),
                                round(zone_out['visibility']['fire_area_covered_pct'], 2),
                                int(zone_out.get('view_changed', False)),
                                *[('' if not (zone_out.get('camera_motion') or {}).get('known') else
                                   round(zone_out['camera_motion'][k], 2)) for k in ('shift_px', 'angle_deg')],
                                int(zone_out.get('calibration_failed', False)),
                                round(zone_out.get('healed_pct', 0.), 3),
                                '' if zone_out.get('deep_ms') is None else round(zone_out['deep_ms'], 1),
                                '' if zone_out.get('deep_max') is None else round(zone_out['deep_max'], 4)]
                    for z in monitor.config.zones:
                        r = zone_out['zones'][z.name]
                        zone_row += [round(r['blocked_pct'], 3), round(r['persistent_pct'], 3), int(r['active'])]
                writer.writerow([datetime.now().isoformat(), sequence, round(milliseconds, 2),
                                 json.dumps(boxes.tolist()), video_seconds]
                                + ([json.dumps(obstructions.tolist()), round(blind, 2)] if two_class else [])
                                + [support, '' if alarm is None else int(alarm), *summary] + zone_row)
                if verifier:
                    track_file.flush()
                file.flush()
                if args.headless:
                    continue
                if monitor:
                    frame = draw_zones(frame, monitor, zone_out)   # under the flame boxes, which stay on top
                    frame = draw_visibility(frame, zone_out['visibility'])
                for x1, y1, x2, y2, score in obstructions:
                    a, b = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(frame, a, b, (255, 128, 0), 2)
                    cv2.putText(frame, f'obstruction {score:.2f}', (a[0] + 4, min(frame.shape[0] - 8, b[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 128, 0), 2)
                for x1, y1, x2, y2, score in boxes:
                    a, b = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(frame, a, b, (0, 0, 255), 2)
                    cv2.putText(frame, f'flame {score:.2f}', (a[0], max(22, a[1]-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
                title = f'CPU | processed {1000/max(milliseconds, .001):.1f} FPS | conf {args.confidence:.2f}'
                cv2.putText(frame, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
                if len(boxes):
                    status = f'FLAME DETECTED | boxes {len(boxes)} | score {boxes[:, 4].max():.2f}'
                    color = (0, 0, 255)
                else:
                    status = 'NO FLAME DETECTED above threshold'
                    color = (0, 255, 0)
                cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                if persistence:
                    k, n = args.persistence
                    check = f'ALARM | box in {support}/{n} frames (needs {k})' if alarm else \
                        f'no alarm | box in {support}/{n} frames (needs {k})'
                    cv2.putText(frame, check, (10, 73), cv2.FONT_HERSHEY_SIMPLEX, .55,
                                (0, 0, 255) if alarm else (0, 255, 0), 2)
                elif not verifier:
                    cv2.putText(frame, 'Frame result only - no alarm verification', (10, 73),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                if two_class:
                    view = (f'VIEW BLOCKED ~{blind:.0f}% of frame | boxes {len(obstructions)}' if len(obstructions)
                            else 'View clear (no obstruction above threshold)')
                    cv2.putText(frame, view, (10, 96), cv2.FONT_HERSHEY_SIMPLEX, .55,
                                (255, 128, 0) if len(obstructions) else (0, 255, 0), 2)
                if verifier:
                    draw_verification(frame, reports, args.verification, args, verify_ms, 119 if two_class else 96)
                if monitor:
                    lines = [('Week 2 flame ALARM: ON' if week2_out['alarm'] else
                              'Week 2 flame alarm: off' + (' (WARNING: flame box)' if week2_out['warning'] else ''),
                              (0, 0, 255) if week2_out['alarm'] else (0, 255, 0))]
                    lines += zone_lines(monitor, zone_out, ready_s, zone_t)
                    lines.append((f'zones {zone_ms:.1f} ms | flame areas excluded from obstruction | '
                                  'image-space % only', (200, 200, 200)))
                    frame = draw_lines(frame, lines, frame.shape[0] - 10 - 22 * len(lines))
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if monitor and key == ord('r'):
                    write_zone_events(monitor.relearn(zone_t))
                    print(f'Re-learning the empty scene at {zone_t:.1f} s')
                if monitor and key == ord('c') and zone_out['state'] == 'awaiting_confirmation':
                    monitor.confirm()
                    print(f'Learned empty scene confirmed by the operator at {zone_t:.1f} s')
                if monitor and args.zone_confirm:
                    reference = monitor.reference_image()
                    if reference is not None and zone_out['state'] == 'awaiting_confirmation':
                        cv2.imshow('Learned empty scene - C confirms, R re-learns', reference)
                if key == ord('s'):
                    path = output / f'capture_{sequence}.jpg'
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError(f'Could not save screenshot to {path}')
                    print(f'Saved {path}')
            if monitor:
                write_zone_events(monitor.close(zone_t))   # an alarm still open at the end is logged too
                if monitor.visibility:
                    write_hidden(monitor.visibility.close(zone_t))
                hidden_file.close()
                if monitor.blind_map is not None:
                    monitor.blind_map.save(output, zone_frame)   # zone blind-zone map: blind_map.json/.png
    finally:
        if monitor and hasattr(monitor.detector, 'close'):
            monitor.detector.close()         # finish the DINOv2 worker cleanly
        camera.close()
        cv2.destroyAllWindows()
    print(f'Session saved: {output}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt) as error:
        raise SystemExit(str(error)) from error
