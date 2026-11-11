"""Fair 3-model comparison on the user's test videos: V9 vs V10 (frozen epoch 6) vs A1 (mean of epochs 6-10).

Every frame of every video is decoded once and the SAME pixels go to all three ONNX models, using the live
viewer's own prepare()/decode() (scripts/live_flame.py): aspect-preserving resize, conf 0.40, NMS 0.65.
Step 1 (infer): per-model per-frame detections (score floor 0.25 kept for analysis) -> logs/<video>/<model>.jsonl
Step 2 (render): side-by-side 3-panel video per clip, boxes >= 0.40 only (the operating point).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, r'E:\Projects\honeywell\honeywell_bmsce_flame_detection\scripts')
from live_flame import decode, prepare  # noqa: E402  (the exact live-viewer pre/post-processing)

PACK = Path(r'E:\Projects\honeywell\Flame_Detection_Test_Pack')
MODELS = {  # name: (path, expected SHA256 prefix)
    'V9': (PACK / 'YOLOv9' / 'flame.onnx', '332cb480'),
    'V10': (PACK / 'YOLOv9_V10' / 'flame.onnx', '6686d04a'),
    'A1': (PACK / 'YOLOv9_V10_A1' / 'flame.onnx', 'bb639b45'),
}
FLOOR, CONF = .25, .40
COLOURS = {'V9': (0, 165, 255), 'V10': (0, 220, 0), 'A1': (255, 160, 0)}
# clip time stamp -> (caption shown in the video, file name)
CAPTIONS = {
    '8.54.39 PM': ('Lighter struck 5 times in a dark room (flame)', '1_lighter_dark_room'),
    '8.54.42 PM': ('Laptop screen playing a fire video (no real flame)', '2_fire_on_laptop_screen'),
    '8.54.46 PM': ('LED lantern + orange/red objects (no flame)', '3_lantern_orange_objects'),
    '8.54.51 PM': ('Lit candle beside LED lantern, orange sock, red object (flame)', '4_candle_with_lantern'),
    '8.54.53 PM': ('Lighter, flame not visible (not scored)', '5_lighter_not_visible'),
    '8.54.57 PM': ('LED lantern + orange object, room lights switching (no flame)', '6_lantern_lights_switching'),
    '8.56.03 PM': ('Lit red candle, camera walking away: near to far (flame)', '7_candle_walk_away'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer(videos, out):
    sessions = {}
    for name, (path, prefix) in MODELS.items():
        digest = sha(path)
        if not digest.startswith(prefix):
            raise RuntimeError(f'{name}: unexpected model hash {digest}')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        sessions[name] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    meta = dict(models={k: dict(path=str(v[0]), sha256=sha(v[0])) for k, v in MODELS.items()},
                score_floor_logged=FLOOR, operating_confidence=CONF, nms=.65, onnxruntime=ort.__version__,
                preprocessing='live_flame.prepare (yolov9)', postprocessing='live_flame.decode (yolov9)')
    (out / 'run_meta.json').write_text(json.dumps(meta, indent=1))
    for video in videos:
        folder = out / 'logs' / video.stem
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'done.json').exists():
            print('skip (done)', video.name, flush=True)
            continue
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        files = {k: open(folder / f'{k}.jsonl', 'w') for k in MODELS}
        index, started = 0, time.time()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tensor, scale = prepare(frame, 640, 'yolov9')
            for name, session in sessions.items():
                raw = session.run(None, {'images': tensor})[0]
                rows = decode(raw, frame, scale, FLOOR, 'yolov9')
                files[name].write(json.dumps(dict(frame=index, t=round(index / fps, 3),
                                                  dets=[[round(float(v), 2) for v in r[:4]] + [round(float(r[4]), 4)]
                                                        for r in rows])) + '\n')
            index += 1
        for f in files.values():
            f.close()
        capture.release()
        (folder / 'done.json').write_text(json.dumps(dict(video=str(video), sha256=sha(video), frames=index, fps=fps,
                                                          seconds=round(time.time() - started, 1))))
        print(f'{video.name}: {index} frames in {time.time() - started:.0f} s', flush=True)


def render(videos, out, names=tuple(MODELS)):
    (out / 'grid').mkdir(exist_ok=True)
    for video in videos:
        folder = out / 'logs' / video.stem
        logs = {k: [json.loads(l) for l in open(folder / f'{k}.jsonl')] for k in names}
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        writer = None
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            panels = []
            for name in names:
                panel = frame.copy()
                kept = [d for d in logs[name][index]['dets'] if d[4] >= CONF]
                for x1, y1, x2, y2, s in kept:
                    cv2.rectangle(panel, (int(x1), int(y1)), (int(x2), int(y2)), COLOURS[name], 2)
                    cv2.putText(panel, f'{s:.2f}', (int(x1), max(12, int(y1) - 4)), 0, .45, COLOURS[name], 1)
                bar = np.zeros((28, panel.shape[1], 3), np.uint8)
                cv2.putText(bar, f'{name}  boxes>=0.40: {len(kept)}', (6, 19), 0, .55, COLOURS[name], 1)
                panels.append(np.vstack([bar, panel]))
            gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
            parts = []
            for panel in panels:
                parts += [panel, gap]
            grid = np.hstack(parts[:-1])
            footer = np.zeros((24, grid.shape[1], 3), np.uint8)
            caption, short = CAPTIONS.get(video.stem[-10:], (video.stem, video.stem))
            cv2.putText(footer, f'{caption}   |   t={index / fps:4.1f}s   |   conf 0.40', (6, 17), 0, .45, (255, 255, 255), 1)
            grid = np.vstack([grid, footer])
            if writer is None:
                writer = cv2.VideoWriter(str(out / 'grid' / f'{short}_{"_vs_".join(names)}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (grid.shape[1], grid.shape[0]))
            writer.write(grid)
            index += 1
        writer.release()
        capture.release()
        print('rendered', video.name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['infer', 'render'])
    p.add_argument('--videos', default=r'E:\Projects\honeywell\testing')
    p.add_argument('--out', default=r'E:\Projects\honeywell\testing\compare_out')
    p.add_argument('--models', default='V9,V10,A1', help='render: comma-separated subset, e.g. V9,V10')
    a = p.parse_args()
    # 8.54.37 PM skipped at the owner's request (flame not visible in the recording)
    videos = [v for v in sorted(Path(a.videos).glob('*.mp4')) if '8.54.37' not in v.name]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.step == 'infer':
        infer(videos, out)
    else:
        render(videos, out, tuple(a.models.split(',')))
