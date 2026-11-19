"""Camera/video viewer for our one-class TI YOLOX and YOLOv9 ONNX exports."""
import argparse
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
    parser.add_argument('--backend', choices=['auto', 'dshow', 'msmf'], default='auto')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--headless', action='store_true', help='Process a saved video without a display')
    parser.add_argument('--camera', type=Path, help='camera.json: add distance_m/bearing_deg/x_m/y_m per detection '
                        '(frames.csv column detections_geo) and draw range rings')
    parser.add_argument('--contact-height-m', type=float, help='override camera.json contact_height_m')
    parser.add_argument('--max-frames', type=int, default=0, help='stop after N frames (0 = no limit)')
    parser.add_argument('--zoom', action='store_true',
                        help='second look at weak full-frame boxes: crop around them, rerun the model, add new boxes '
                             '(see src/detection/zoom.py); --confidence stays the full-frame threshold')
    parser.add_argument('--zoom-confidence', type=float, default=0.20, help='threshold for boxes found in a crop')
    parser.add_argument('--zoom-candidate-min', type=float, default=0.10,
                        help='weak full-frame boxes from this score up to --confidence are zoom candidates')
    parser.add_argument('--zoom-crop', type=int, default=320, help='crop size in original-frame pixels')
    parser.add_argument('--zoom-max-crops', type=int, default=3, help='at most this many crops per frame')
    parser.add_argument('--alarm-rule', type=Path,
                        help='rule_v12.json: apply the locked temporal rule live (CLEAR / WARNING / ALARM); '
                             'see src/detection/temporal.py')
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error('--confidence must be between 0 and 1')
    if not args.model.is_file():
        parser.error(f'Model missing: {args.model}. Supply --model with the actual flame.onnx path.')
    cv2.setNumThreads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    zoom_cfg = None
    if args.zoom:
        sys.path.insert(0, str(ROOT / 'src'))
        from detection import ZoomConfig, detect_with_zoom
        try:
            zoom_cfg = ZoomConfig(full_confidence=args.confidence, zoom_confidence=args.zoom_confidence,
                                  candidate_min=args.zoom_candidate_min, max_crops=args.zoom_max_crops,
                                  crop=args.zoom_crop)
        except ValueError as error:
            parser.error(f'zoom settings: {error}')
    alarm_rule = None
    if args.alarm_rule:
        if not args.alarm_rule.is_file():
            parser.error(f'Alarm rule missing: {args.alarm_rule}')
        sys.path.insert(0, str(ROOT / 'src'))
        from detection import AlarmRule
        alarm_rule = AlarmRule.from_json(args.alarm_rule)
    geo_cam = overlay = None
    if args.camera:
        sys.path.insert(0, str(ROOT / 'src'))
        from geometry import Camera
        from geometry.overlay import RingOverlay, fit_to_frame
        geo_cam = Camera.load(args.camera)
    size = 416 if args.arch == 'nano' else 640
    if session.get_inputs()[0].shape != [1, 3, size, size]:
        raise ValueError(f'{args.arch} requires our fixed {size}x{size} flame export.')
    tensor, scale = prepare(np.zeros((480, 640, 3), dtype=np.uint8), size, args.arch)
    warmup = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    decode(warmup, np.zeros((480, 640, 3), dtype=np.uint8), scale, args.confidence, args.arch)
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
                    source_kind='video' if is_video else 'camera',
                    source=str(source), backend=args.backend, provider='CPUExecutionProvider',
                    zoom=(vars(zoom_cfg) if zoom_cfg else None),
                    alarm_rule=(str(args.alarm_rule.resolve()) if alarm_rule else None),
                    platform=platform.platform(), onnxruntime=ort.__version__,
                    note=('Detections plus the locked temporal alarm rule; no ground-truth accuracy metrics. '
                          if alarm_rule else
                          'Frame detections only; no alarm verification or ground-truth accuracy metrics. ')
                         + 
                         'CPU processing FPS is not camera FPS or AM62A performance.')
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
        with (output / 'frames.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['time_local', 'camera_sequence', 'processing_ms', 'detections_xyxy_score', 'video_seconds']
                            + (['detections_geo'] if geo_cam else [])
                            + (['detections_from_zoom', 'zoom_crops_xywh'] if zoom_cfg else [])
                            + (['alarm_state', 'best_hit_ratio'] if alarm_rule else []))
            while True:
                item = camera.next(sequence)
                if item is None:
                    break
                sequence, frame = item
                started = time.perf_counter()
                from_zoom, windows = np.zeros(0, dtype=bool), []
                if zoom_cfg is None:
                    tensor, scale = prepare(frame, size, args.arch)
                    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                    boxes = decode(raw, frame, scale, args.confidence, args.arch)
                else:
                    def infer(image):
                        t, s = prepare(image, size, args.arch)
                        return decode(session.run(None, {session.get_inputs()[0].name: t})[0], image, s,
                                      zoom_cfg.candidate_min, args.arch)
                    boxes, from_zoom, windows = detect_with_zoom(frame, infer, zoom_cfg)
                milliseconds = (time.perf_counter() - started) * 1000
                verdict = None
                if alarm_rule is not None:
                    now = (sequence - 1) / camera.fps if is_video and camera.fps > 0 else time.monotonic()
                    verdict = alarm_rule.update(now, boxes.tolist())
                geo = []
                if geo_cam is not None:
                    if overlay is None:
                        fh, fw = frame.shape[:2]
                        geo_cam = fit_to_frame(geo_cam, fw, fh)
                        overlay = RingOverlay(geo_cam, fw, fh)
                    geo = [geo_cam.box_to_floor(b, args.contact_height_m) for b in boxes]
                writer.writerow([datetime.now().isoformat(), sequence, round(milliseconds, 2),
                                 json.dumps(boxes.tolist()),
                                 (sequence-1)/camera.fps if is_video and camera.fps > 0 else '']
                                + ([json.dumps(geo)] if geo_cam is not None else [])
                                + ([json.dumps(from_zoom.tolist()), json.dumps(windows)] if zoom_cfg else [])
                                + ([verdict['state'], round(verdict['best_hit_ratio'], 3)] if verdict else []))
                file.flush()
                if args.max_frames and sequence >= args.max_frames:
                    break
                if args.headless:
                    continue
                if overlay is not None:
                    overlay.apply(frame)
                for x0, y0, w, h in windows:
                    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 0), 1)
                for i, (x1, y1, x2, y2, score) in enumerate(boxes):
                    a, b = (int(x1), int(y1)), (int(x2), int(y2))
                    zoomed = i < len(from_zoom) and from_zoom[i]
                    color = (0, 140, 255) if zoomed else (0, 0, 255)
                    cv2.rectangle(frame, a, b, color, 2)
                    dist = f" {geo[i]['distance_m']:.1f} m" if geo and geo[i]['distance_m'] is not None else ''
                    cv2.putText(frame, f"flame {score:.2f}{' zoom' if zoomed else ''}{dist}", (a[0], max(22, a[1]-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
                title = f'CPU | processed {1000/max(milliseconds, .001):.1f} FPS | conf {args.confidence:.2f}'
                if zoom_cfg:
                    title += f' | zoom {zoom_cfg.zoom_confidence:.2f}, crops {len(windows)}'
                cv2.putText(frame, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
                if verdict is not None:
                    color = {'ALARM': (0, 0, 255), 'WARNING': (0, 200, 255), 'CLEAR': (0, 200, 0)}[verdict['state']]
                    status = f"{verdict['state']} | boxes {len(boxes)} | hit ratio {verdict['best_hit_ratio']:.2f}"
                    note = 'ALARM = flame in >= 30% of frames over 2 s, off after 3 s (rule_v12)'
                elif len(boxes):
                    status = f'FLAME DETECTED | boxes {len(boxes)} | score {boxes[:, 4].max():.2f}'
                    color, note = (0, 0, 255), 'Frame result only - no alarm verification'
                else:
                    status = 'NO FLAME DETECTED above threshold'
                    color, note = (0, 255, 0), 'Frame result only - no alarm verification'
                cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                cv2.putText(frame, note, (10, 73), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('s'):
                    path = output / f'capture_{sequence}.jpg'
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError(f'Could not save screenshot to {path}')
                    print(f'Saved {path}')
    finally:
        camera.close()
        cv2.destroyAllWindows()
    print(f'Session saved: {output}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt) as error:
        raise SystemExit(str(error)) from error
