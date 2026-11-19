
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('W3_DATA', REPO.parent))          # folder holding W3_starter_V12/ and w3_demo/
MODEL = str(DATA / 'W3_starter_V12/models/flame_v12_full.onnx')
VIDEO = os.environ.get('W3_VIDEO', str(DATA / 'w3_demo/drive_videos/fire6.mp4'))
OUT = Path(os.environ.get('W3_OUT', DATA / 'w3_demo'))
"""fire_6: side-by-side FULL FRAME vs ZOOM-320, both with the locked V12 temporal rule (WARNING per frame, ALARM = 2 s/30%/3 s off)
and distance from camera_fire6.json. Runs the detector at 10 fps (as the rule was tuned)."""
import sys, json, cv2, numpy as np, onnxruntime as ort
R = str(REPO)
sys.path[:0] = [R + '/src', R + '/scripts', R + '/experiments/temporal_final', R + '/experiments/yolov9_v10/workflow']
from live_flame import prepare, decode
import temporal_final as TF
from geometry import Camera
from geometry.overlay import RingOverlay
RULE = json.load(open(str(DATA / 'W3_starter_V12/rule_v12.json')))
cam = Camera.load(str(REPO / 'experiments/w3_validation/calibrations/camera_fire6.json')); OFF = 0.415; W, H = 1920, 1080; C = 320
s = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
cap = cv2.VideoCapture(VIDEO); fps = cap.get(5)
SEG = [(11, 25, 0.5), (26, 41, 1.0), (42, 57, 1.5), (58, 71, 2.0), (72, 86, 2.5), (88, 104, 3.0)]
def det(img):
    x, r = prepare(img, 640); return [list(map(float, b[:5])) for b in decode(s.run(None, {s.get_inputs()[0].name: x})[0], img, r, 0.05, 'yolov9')]
frames, imgs, crops, nxt = {'full': [], 'zoom': []}, [], [], 0.0
while True:
    ok, f = cap.read()
    if not ok: break
    t = cap.get(0) / 1000
    if t + 1e-6 < nxt: continue
    nxt += 0.1
    frames['full'].append(dict(t=t, dets=det(f)))
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (9, 9), 0); _, _, _, (x, y) = cv2.minMaxLoc(g)
    x0 = min(max(0, x - C // 2), W - C); y0 = min(max(0, y - C // 2), H - C)
    d = det(np.ascontiguousarray(f[y0:y0 + C, x0:x0 + C]))
    frames['zoom'].append(dict(t=t, dets=[[b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0, b[4]] for b in d]))
    imgs.append(cv2.resize(f, (W // 2, H // 2))); crops.append((x0, y0))
state, summary = {}, {}
for p, fr in frames.items():
    to640 = 640 / max(W, H) if p == 'full' else 640 / C
    per = TF.track(fr, to640, None); times = [q['t'] for q in fr]
    flags = TF.raw_flags(times, per, RULE['window_s'], RULE['min_hit_ratio'], RULE['min_mean_score'])
    state[p] = (per, TF.alarm(times, flags, RULE['alarm_on_after_s']))
    rows = []
    for a, b, dist in SEG:
        idx = [i for i, q in enumerate(fr) if a <= q['t'] < b]
        warn = sum(1 for i in idx if TF.gate(fr[i]['dets'], to640, None)); al = sum(state[p][1][i] for i in idx)
        ds = [cam.box_to_floor(d[:4])['distance_m'] - OFF for i in idx for d in TF.gate(fr[i]['dets'], to640, None)]
        rows.append(dict(mark=dist, frames=len(idx), warning=warn, alarm=al, dist=round(float(np.median(ds)), 3) if ds else None))
    nc = [i for i, q in enumerate(fr) if q['t'] < 9]
    summary[p] = dict(no_candle_alarm=sum(state[p][1][i] for i in nc), rows=rows)
json.dump(summary, open(str(OUT / 'fire6_zoom_alarm_results.json'), 'w'), indent=1)
for p in summary:
    print(p, 'no-candle alarm frames:', summary[p]['no_candle_alarm'])
    for r in summary[p]['rows']: print('  ', r)
ov = RingOverlay(cam.with_resolution(W // 2, H // 2), W // 2, H // 2, radii=[d + OFF for _, _, d in SEG], labels=False)
vw = cv2.VideoWriter(str(OUT / 'W3_fire6_zoom_vs_full_alarm.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H // 2 + 60))
for i, im in enumerate(imgs):
    t = frames['full'][i]['t']; mark = next((d for a, b, d in SEG if a <= t < b), None); panels = []
    for p in ('full', 'zoom'):
        v = np.clip(((im / 255.0) ** 0.6) * 255, 0, 255).astype(np.uint8); ov.apply(v)
        to640 = 640 / max(W, H) if p == 'full' else 640 / C
        g = TF.gate(frames[p][i]['dets'], to640, None); al = state[p][1][i]
        if p == 'zoom':
            x0, y0 = crops[i]; cv2.rectangle(v, (x0 // 2, y0 // 2), ((x0 + C) // 2, (y0 + C) // 2), (255, 255, 0), 1)
        txt = ''
        for d in g:
            cv2.rectangle(v, (int(d[0] / 2), int(d[1] / 2)), (int(d[2] / 2), int(d[3] / 2)), (0, 255, 0), 2)
            txt = f"{cam.box_to_floor(d[:4])['distance_m'] - OFF:.2f} m  ({d[4]:.2f})"
        st, col = ('ALARM', (0, 0, 255)) if al else (('WARNING', (0, 200, 255)) if g else ('clear', (160, 160, 160)))
        cv2.rectangle(v, (0, 0), (W // 2, 44), (0, 0, 0), -1)
        cv2.putText(v, ('FULL FRAME' if p == 'full' else 'ZOOM 320') + f"  {st}", (10, 32), 0, 0.9, col, 2)
        cv2.putText(v, txt, (560, 32), 0, 0.8, (0, 255, 0), 2); panels.append(v)
    bar = np.zeros((60, W, 3), np.uint8)
    cv2.putText(bar, f"t={t:5.1f}s   " + (f"candle on {mark:g} m mark" if mark else "no candle / moving") + "   WARNING = detector this frame   ALARM = 2 s window, 30% hits, off after 3 s", (10, 40), 0, 0.8, (255, 255, 255), 2)
    vw.write(np.vstack([np.hstack(panels), bar]))
vw.release(); print('video done')
