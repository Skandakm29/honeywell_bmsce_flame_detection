"""Does V10 / V12 sense the daylight candle at all? IMG_7936 at a score floor of 0.01, every 3rd frame (10 fps).
A detection counts as "on the flame" if its box centre lies inside the flame box (flame_contrast.daylight_box)
expanded by 50%, or IoU >= 0.1 with it. Reports the best on-flame score per frame, per distance segment.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(r'E:\Projects\honeywell')
sys.path[:0] = [str(ROOT / r'honeywell_bmsce_flame_detection\scripts'), str(ROOT / 'testing')]
from live_flame import decode, prepare  # noqa: E402
from flame_contrast import daylight_box  # noqa: E402

MODELS = {'V10': ROOT / r'Flame_Detection_Test_Pack\YOLOv9_V10\flame.onnx',
          'V12': ROOT / r'investigation_week2\lab_backup_2026-11-07\work\select_v12_rec\export\flame.onnx'}
VIDEO = ROOT / r'testing\videos_2026-11-09\IMG_7936.MOV'
SEGMENTS = ((0, 10, '1 m'), (12, 30, '3 m'), (32, 45, '4 m'))


def on_flame(d, fb):
    x1, y1, x2, y2 = fb
    w, h = x2 - x1, y2 - y1
    cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    inside = x1 - w <= cx <= x2 + w and y1 - 2 * max(h, 20) <= cy <= y2 + .5 * h   # flame sits above the stick
    ix = max(0., min(d[2], x2) - max(d[0], x1)); iy = max(0., min(d[3], y2) - max(d[1], y1))
    union = (d[2] - d[0]) * (d[3] - d[1]) + w * h - ix * iy
    return inside or (union > 0 and ix * iy / union >= .1)


def main():
    sessions = {k: ort.InferenceSession(str(p), providers=['CPUExecutionProvider']) for k, p in MODELS.items()}
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, index, best_example = [], 0, {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % 3 == 0:
            t = index / fps
            fb = daylight_box(frame)
            tensor, scale = prepare(frame, 640, 'yolov9')
            row = dict(t=round(t, 2), flame_box=fb)
            for name, s in sessions.items():
                dets = decode(s.run(None, {'images': tensor})[0], frame, scale, .01, 'yolov9')
                on = [float(d[4]) for d in dets if on_flame(d, fb)]
                off = [float(d[4]) for d in dets if not on_flame(d, fb)]
                row[name] = dict(on_flame=max(on) if on else 0., elsewhere=max(off) if off else 0.)
                if on and (name not in best_example or max(on) > best_example[name][0]):
                    best_example[name] = (max(on), t)
            rows.append(row)
        index += 1
    out = ROOT / r'testing\compare_2026-11-09\lowconf_daylight.json'
    summary = {}
    for name in MODELS:
        for a, b, lab in SEGMENTS:
            s = [r[name] for r in rows if a <= r['t'] < b]
            on = np.array([x['on_flame'] for x in s])
            summary.setdefault(name, {})[lab] = dict(frames=len(s), max_on_flame=round(float(on.max()), 3),
                                                     median_on_flame=round(float(np.median(on)), 3),
                                                     frames_any_on_flame=round(float((on > 0).mean()), 2),
                                                     frames_on_flame_ge_010=round(float((on >= .10).mean()), 2),
                                                     max_elsewhere=round(float(max(x['elsewhere'] for x in s)), 3))
    out.write_text(json.dumps(dict(summary=summary, best_example=best_example, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))
    print('best on-flame score (score, time s):', best_example)


if __name__ == '__main__':
    main()
