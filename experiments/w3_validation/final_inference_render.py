
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
    imgs.append(cv2.resize(f, (1280, 720))); crops.append((x0, y0))
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

S = 1280 / W
ov = RingOverlay(cam.with_resolution(1280, 720), 1280, 720, radii=[d + OFF for _, _, d in SEG], labels=False)
lab = {d: cam.floor_to_pixel([[0, d + OFF]])[0] * S for _, _, d in SEG}
vw = cv2.VideoWriter(str(OUT / 'W3_final_inference_fire6.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 720))
to640 = 640 / C; per, al = state['zoom']
for i, im in enumerate(imgs):
    t = frames['zoom'][i]['t']; mark = next((d for a, b, d in SEG if a <= t < b), None)
    v = np.clip(((im / 255.0) ** 0.6) * 255, 0, 255).astype(np.uint8); ov.apply(v)
    for d, p in lab.items(): cv2.putText(v, f"{d:g} m", (int(p[0]) + 170, int(p[1]) - 4), 0, 0.6, (0, 255, 255), 1)
    g = TF.gate(frames['zoom'][i]['dets'], to640, None); txt = ''
    for d in g:
        cv2.rectangle(v, (int(d[0] * S), int(d[1] * S)), (int(d[2] * S), int(d[3] * S)), (0, 255, 0), 2)
        geo = cam.box_to_floor(d[:4]); txt = f"Flame at {geo['distance_m'] - OFF:.2f} m, bearing {geo['bearing_deg']:+.0f} deg  (score {d[4]:.2f})"
    st, col = ('ALARM', (0, 0, 255)) if al[i] else (('WARNING', (0, 200, 255)) if g else ('CLEAR', (0, 200, 0)))
    cv2.rectangle(v, (0, 0), (1280, 56), (0, 0, 0), -1)
    cv2.putText(v, st, (14, 42), 0, 1.3, col, 3)
    cv2.putText(v, txt, (230, 38), 0, 0.75, (255, 255, 255), 2)
    cv2.rectangle(v, (0, 680), (1280, 720), (0, 0, 0), -1)
    cv2.putText(v, f"t={t:5.1f}s" + (f"   ground truth: {mark:g} m" if mark else "") + "   V12 + zoom | WARNING = this frame | ALARM = >=30% of 2 s, off after 3 s",
                (14, 708), 0, 0.6, (220, 220, 220), 1)
    vw.write(v)
vw.release(); print('done')
