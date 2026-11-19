"""Phase A: cache V12 detections at 10 fps for every test clip.
Per frame: full-frame boxes (score >= 0.05) and zoom-on-candidate boxes: up to 3 full-frame boxes with score in
[0.10, 0.40) are candidates; a 320x320 crop (centre snapped to a 32 px grid so it stays steady) is cut around each,
resized to 640, run through V12, and its boxes mapped back to full-frame pixels. Also times each pass."""
import os, sys, json, time, glob
import cv2, numpy as np, onnxruntime as ort
import os as _os
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[2]
H0 = _os.environ.get('W3_DATA', str(_REPO.parent)); R = str(_REPO)
sys.path[:0] = [R + '/scripts']
from live_flame import prepare, decode
OUT = os.path.join(os.environ.get('W3_OUT', os.path.join(H0, 'w3_demo')), 'phaseA_dets'); os.makedirs(OUT, exist_ok=True)
s = ort.InferenceSession(H0 + '/W3_starter_V12/models/flame_v12_full.onnx', providers=['CPUExecutionProvider'])
IN = s.get_inputs()[0].name; C = 320


def det(img):
    x, r = prepare(img, 640)
    return [[round(float(v), 1) for v in b[:4]] + [round(float(b[4]), 4)] for b in decode(s.run(None, {IN: x})[0], img, r, 0.05, 'yolov9')]


for path in sorted(glob.glob(H0 + '/test_videos_1p5m/*.mp4')):
    name = os.path.basename(path)[:-4]; dst = os.path.join(OUT, name + '.json')
    if os.path.exists(dst): continue
    cap = cv2.VideoCapture(path); W, H = int(cap.get(3)), int(cap.get(4)); frames = []; nxt = 0.0
    while True:
        ok, f = cap.read()
        if not ok: break
        t = cap.get(0) / 1000
        if t + 1e-6 < nxt: continue
        nxt += 0.1
        t0 = time.perf_counter(); full = det(f); t1 = time.perf_counter()
        cands = sorted([b for b in full if 0.10 <= b[4] < 0.40], key=lambda b: -b[4])[:3]
        zoom, centres = [], set()
        for b in cands:
            cx = int(round((b[0] + b[2]) / 2 / 32) * 32); cy = int(round((b[1] + b[3]) / 2 / 32) * 32)
            x0 = min(max(0, cx - C // 2), W - C); y0 = min(max(0, cy - C // 2), H - C)
            if (x0, y0) in centres: continue
            centres.add((x0, y0))
            zoom += [[d[0] + x0, d[1] + y0, d[2] + x0, d[3] + y0, d[4]] for d in det(np.ascontiguousarray(f[y0:y0 + C, x0:x0 + C]))]
        t2 = time.perf_counter()
        frames.append(dict(t=round(t, 3), full=full, zoom=zoom, ncrop=len(centres), ms_full=round((t1 - t0) * 1e3, 1), ms_zoom=round((t2 - t1) * 1e3, 1)))
    json.dump(dict(W=W, H=H, frames=frames), open(dst, 'w'))
    print(name, len(frames), 'frames', flush=True)
print('ALL DONE')
