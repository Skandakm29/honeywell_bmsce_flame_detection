"""Why is the candle missed in daylight? Measure the flame against its surroundings in both 9 Nov videos.

Flame box per frame:
  dim video (VID20261109112922): V12's own detection (>= .40), which is on the candle in every frame.
  daylight video (IMG_7936): static camera; the flame = the brightest warm spot inside a fixed window above the cup.
Measured inside the flame box (the brightest 30% of its pixels = the core) and in a ring around it (3x box, minus box):
  core brightness V (0-1), core saturation S (0-1), surroundings brightness, contrast = core V - surroundings V,
  warm colour = mean (R - B) / 255 of the core, flame height in px at the model's 640 input.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'E:\Projects\honeywell\testing')


def stats(frame, box):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.
    f = frame.astype(np.float32)
    region = hsv[y1:y2, x1:x2].reshape(-1, 3)
    core = region[:, 2] >= np.quantile(region[:, 2], .7)
    rb = (f[y1:y2, x1:x2, 2] - f[y1:y2, x1:x2, 0]).reshape(-1)[core].mean() / 255.
    w, h = x2 - x1, y2 - y1
    X1, Y1, X2, Y2 = max(0, x1 - w), max(0, y1 - h), min(W, x2 + w), min(H, y2 + h)
    ring = np.ones((Y2 - Y1, X2 - X1), bool)
    ring[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
    bg = hsv[Y1:Y2, X1:X2, 2][ring]
    cv, cs = float(region[core, 2].mean()), float(region[core, 1].mean())
    return dict(core_V=cv, core_S=cs, warm=float(rb), surround_V=float(bg.mean()), contrast=cv - float(bg.mean()),
                height_640=h * 640 / max(W, H))


def daylight_box(frame, window=(850, 380, 1150, 820)):
    x0, y0, x1, y1 = window
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    warm = np.clip((crop[:, :, 2] + crop[:, :, 1]) / 2 - crop[:, :, 0], 0, None)
    score = cv2.GaussianBlur(crop.max(2) / 255. * (1 + warm / 64.), (0, 0), 3)
    _, _, _, (px, py) = cv2.minMaxLoc(score)
    # flame = bright region around the peak: pixels brighter than halfway between local surroundings and the peak
    v = crop.max(2)
    y_a, y_b, x_a, x_b = max(0, py - 60), min(v.shape[0], py + 30), max(0, px - 30), min(v.shape[1], px + 30)
    local = v[y_a:y_b, x_a:x_b]
    level = np.median(local) + .5 * (local.max() - np.median(local))
    n, lab, st, _ = cv2.connectedComponentsWithStats((local >= level).astype(np.uint8), 8)
    k = lab[py - y_a, px - x_a] or 1 + int(np.argmax(st[1:, 4]))
    bx, by, bw, bh = st[k][:4]
    return [int(v) for v in (bx + x_a + x0 - 2, by + y_a + y0 - 2, bx + x_a + x0 + bw + 2, by + y_a + y0 + bh + 2)]


def run():
    rows = {}
    # dim room: V12 boxes
    name = 'VID20261109112922'
    log = [json.loads(l) for l in open(ROOT / 'compare_2026-11-09' / 'logs' / name / 'V12.jsonl')]
    cap = cv2.VideoCapture(str(ROOT / 'videos_2026-11-09' / f'{name}.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows['dim room (detected)'] = []
    for s in np.arange(.5, 22, 1.):
        i = int(s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        dets = [d for d in log[i]['dets'] if d[4] >= .4]
        if ok and dets:
            rows['dim room (detected)'].append(dict(t=float(s), **stats(frame, max(dets, key=lambda d: d[4])[:4])))
    # daylight: fixed window search
    cap = cv2.VideoCapture(str(ROOT / 'videos_2026-11-09' / 'IMG_7936.MOV'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows['bright daylight (missed)'] = []
    examples = []
    for s in np.arange(.5, 44, 1.):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        box = daylight_box(frame)
        rows['bright daylight (missed)'].append(dict(t=float(s), box=box, **stats(frame, box)))
        if s in (2.5, 15.5, 30.5):
            examples.append((frame, box, s))
    summary = {}
    for k, r in rows.items():
        keys = ('core_V', 'core_S', 'warm', 'surround_V', 'contrast', 'height_640')
        summary[k] = {q: round(float(np.median([x[q] for x in r])), 3) for q in keys} | dict(frames=len(r))
    out = ROOT / 'compare_2026-11-09' / 'flame_contrast.json'
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    for k, v in summary.items():
        print(k, v)
    # check sheet: the daylight boxes found
    tiles = []
    for frame, box, s in examples:
        x1, y1, x2, y2 = [int(v) for v in box]
        c = frame[max(0, y1 - 80):y2 + 80, max(0, x1 - 80):x2 + 80].copy()
        cv2.rectangle(c, (80, 80), (80 + x2 - x1, 80 + y2 - y1), (0, 0, 255), 1)
        tiles.append(cv2.resize(c, (300, int(300 * c.shape[0] / c.shape[1]))))
    h = max(t.shape[0] for t in tiles)
    cv2.imwrite(str(ROOT / 'compare_2026-11-09' / 'daylight_flame_boxes.jpg'),
                np.hstack([np.vstack([t, np.zeros((h - t.shape[0], 300, 3), np.uint8)]) for t in tiles]))


if __name__ == '__main__':
    run()
