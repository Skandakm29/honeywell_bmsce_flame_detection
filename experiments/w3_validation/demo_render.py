"""W3 demo video: calibration + V12 detection + zoom on weak candidates + live temporal alarm + range rings.

Uses the same code as `scripts/live_flame.py --zoom --alarm-rule --camera`: prepare/decode, src/detection
(zoom + AlarmRule) and src/geometry (Camera, RingOverlay). Processes the clip at 10 fps (the rate the alarm rule
was tuned at) and writes a 10 fps MP4, so it plays in real time.

  python experiments/w3_validation/demo_render.py --video m39_1to3_lighton.mp4 \
      --camera experiments/w3_validation/calibrations/camera_1p5m_setup39.json \
      --model flame_v12_full.onnx --rule rule_v12.json --out W3_demo.mp4
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from live_flame import prepare, decode  # noqa: E402
from detection import ZoomConfig, detect_with_zoom, AlarmRule  # noqa: E402
from geometry import Camera  # noqa: E402
from geometry.overlay import RingOverlay, ring_polylines, fit_to_frame  # noqa: E402

OW, OH = 1280, 720
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, ORANGE, CYAN, YELLOW, WHITE, GREEN, GREY = (0, 0, 255), (0, 140, 255), (255, 255, 0), (0, 255, 255), \
    (255, 255, 255), (0, 200, 0), (200, 200, 200)


def text(img, s, org, scale=0.7, color=WHITE, thick=2):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def ring_labels(cam, radii, w, h):
    """Label position on each ring: the ring point nearest x = 22% of the width (away from the centre line)."""
    out = []
    for r in radii:
        pts = [p for run in ring_polylines(cam, r) for p in run
               if 10 < p[0] < w - 80 and 30 < p[1] < h - 10]
        if pts:
            p = min(pts, key=lambda p: abs(p[0] - 0.22 * w))
            out.append((r, (int(p[0]), int(p[1]))))
    return out


def card(lines, bg=None, seconds=4.0, fps=10):
    img = np.zeros((OH, OW, 3), np.uint8) if bg is None else bg.copy()
    if bg is not None:
        img[:] = (img * 0.35).astype(np.uint8)
    y = 120
    for s, scale, color in lines:
        if s:
            text(img, s, (70, y), scale, color, 2 if scale < 1 else 3)
        y += int(52 * max(scale, 0.75))
    return [img] * int(seconds * fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--rule', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tapes', default='1:844,949 1.5:853,619 2:848,392 2.5:858,217 3:840,87',
                    help='tape marks used for calibration, metres:x,y in full-res pixels (for the calibration card)')
    ap.add_argument('--radii', default='1,1.5,2,2.5,3')
    a = ap.parse_args()

    sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    cfg = ZoomConfig()
    rule = AlarmRule.from_json(a.rule)
    cap = cv2.VideoCapture(a.video)
    W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    cam = fit_to_frame(Camera.load(a.camera), W, H)
    S = OW / W
    small = fit_to_frame(Camera.load(a.camera), OW, OH)
    radii = [float(r) for r in a.radii.split(',')]
    rings = RingOverlay(small, OW, OH, radii=radii, labels=False, thickness=2)
    labels = ring_labels(small, radii, OW, OH)
    tapes = {float(k): tuple(map(int, v.split(','))) for k, v in (t.split(':') for t in a.tapes.split())}
    pose = json.load(open(a.camera))['pose']

    def infer(img):
        t, s = prepare(img, 640)
        return decode(sess.run(None, {name: t})[0], img, s, cfg.candidate_min)

    def draw_rings(v):
        rings.apply(v)
        for r, (x, y) in labels:
            text(v, f'{r:g} m', (x - 20, y - 8), 0.75, YELLOW, 2)

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'), 10, (OW, OH))
    ok, first = cap.read()
    bg = cv2.resize(first, (OW, OH))

    # --- card 1: what the demo shows
    for f in card([('Week 3 demo: flame detection with spatial calibration', 1.05, WHITE), ('', 1, WHITE),
                   ('1. Calibration: camera height + tilt -> every pixel maps to a floor position', 0.72, YELLOW),
                   ('2. V12 detects flames on the full frame (score >= 0.40)', 0.72, RED),
                   ('3. Zoom: weak boxes (0.10-0.40) are cropped, re-checked, kept if >= 0.20', 0.72, ORANGE),
                   ('4. Alarm rule: flame in >= 30% of frames over 2 s -> ALARM; off after 3 s', 0.72, (80, 80, 255)),
                   ('5. Distance + bearing of each flame; range rings on the floor', 0.72, GREEN), ('', 1, WHITE),
                   ('Model V12 unchanged (no retraining). Test clip: camera at 1.5 m, candle moved 1 m -> 3 m.',
                    0.6, GREY)], bg, 6.0):
        vw.write(f)

    # --- card 2: calibration
    v = bg.copy()
    draw_rings(v)
    for d, (x, y) in tapes.items():
        p = (int(x * S), int(y * S))
        cv2.circle(v, p, 14, (255, 0, 255), 2, cv2.LINE_AA)
        text(v, f'tape {d:g} m', (p[0] + 22, p[1] + 6), 0.6, (255, 0, 255), 2)
    cv2.rectangle(v, (0, 0), (OW, 128), (0, 0, 0), -1)
    text(v, 'STEP 1  Calibration (once, at install)', (20, 34), 0.85, YELLOW)
    text(v, f"Measured: camera height {pose['height_m']} m.   Fitted from the 5 tape marks (pink): "
            f"tilt {pose['tilt_deg']} deg, field of view 61.5 deg", (20, 70), 0.6, WHITE, 1)
    text(v, 'Yellow rings = floor distance from the point below the camera, drawn from the calibration. '
            'They pass through the tapes.', (20, 102), 0.6, WHITE, 1)
    for _ in range(80):
        vw.write(v)

    # --- card 3: legend
    for f in card([('STEP 2  How each frame is processed', 1.0, WHITE), ('', 1, WHITE),
                   ('Red box     = flame found on the full frame (score >= 0.40)', 0.72, RED),
                   ('Cyan square = zoom crop around a weak box (score 0.10-0.40)', 0.72, CYAN),
                   ('Orange box  = flame confirmed inside the zoom crop (score >= 0.20)', 0.72, ORANGE),
                   ('Top-right   = the zoom crop as the model sees it', 0.72, WHITE), ('', 1, WHITE),
                   ('Status: CLEAR (no flame) / WARNING (flame this frame) / ALARM (confirmed by the 2 s rule)',
                    0.62, YELLOW),
                   ('Each flame: distance along the floor + bearing (left -, right +)', 0.62, GREEN)], bg, 6.0):
        vw.write(f)

    # --- main: every 3rd frame (10 fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    step = max(1, round(fps / 10))
    i = -1
    last_crop = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        t = i / fps
        boxes, from_zoom, windows = detect_with_zoom(frame, infer, cfg)
        verdict = rule.update(t, boxes.tolist())
        v = cv2.resize(frame, (OW, OH))
        draw_rings(v)
        for x0, y0, w, h in windows:
            cv2.rectangle(v, (int(x0 * S), int(y0 * S)), (int((x0 + w) * S), int((y0 + h) * S)), CYAN, 2)
        if windows:
            x0, y0, w, h = windows[0]
            last_crop = (cv2.resize(frame[y0:y0 + h, x0:x0 + w], (240, 240), interpolation=cv2.INTER_LINEAR),
                         [b for b, z in zip(boxes, from_zoom) if z], (x0, y0, w))
        info = ''
        # headline: full-frame flames first (most trusted), then zoom ones; strongest first within each
        for b, z in sorted(zip(boxes, from_zoom), key=lambda bz: (bool(bz[1]), -bz[0][4])):
            col = ORANGE if z else RED
            p1, p2 = (int(b[0] * S), int(b[1] * S)), (int(b[2] * S), int(b[3] * S))
            cv2.rectangle(v, p1, p2, col, 2, cv2.LINE_AA)
            g = cam.box_to_floor(b[:4])
            if g['distance_m'] is not None:
                text(v, f"{g['distance_m']:.2f} m", (p2[0] + 8, p2[1]), 0.7, col, 2)
                if not info:
                    info = (f"Flame at {g['distance_m']:.2f} m, bearing {g['bearing_deg']:+.0f} deg  "
                            f"({'zoom' if z else 'full frame'}, score {b[4]:.2f})")
        # zoom inset (keep the last crop on screen for a moment so it can be read)
        if last_crop is not None and windows:
            crop, zb, (x0, y0, w) = last_crop
            inset = crop.copy()
            k = 240 / w
            for b in zb:
                if x0 <= b[0] <= x0 + w and y0 <= b[1] <= y0 + w:
                    cv2.rectangle(inset, (int((b[0] - x0) * k), int((b[1] - y0) * k)),
                                  (int((b[2] - x0) * k), int((b[3] - y0) * k)), ORANGE, 2)
            v[70:310, OW - 254:OW - 14] = inset
            cv2.rectangle(v, (OW - 255, 69), (OW - 13, 311), CYAN, 2)
            text(v, f'zoom crop ({w} px -> model)', (OW - 254, 330), 0.5, CYAN, 1)
        state = verdict['state']
        col = {'ALARM': RED, 'WARNING': (0, 200, 255), 'CLEAR': GREEN}[state]
        cv2.rectangle(v, (0, 0), (OW, 56), (0, 0, 0), -1)
        text(v, state, (14, 41), 1.25, col, 3)
        text(v, info or 'no flame in this frame', (240, 36), 0.7, WHITE, 2)
        cv2.rectangle(v, (0, OH - 34), (OW, OH), (0, 0, 0), -1)
        text(v, f't = {t:5.1f} s   |   hit ratio {verdict["best_hit_ratio"]:.2f} (alarm at 0.30)   |   '
                f'red = full frame, orange = zoom, cyan = crop   |   rings: floor distance', (12, OH - 11), 0.5, GREY, 1)
        vw.write(v)
    vw.release()
    print('wrote', a.out)


if __name__ == '__main__':
    main()
