"""full-frame V12 + locked temporal rule at several score thresholds on fire_6."""
import os, sys, json
import cv2, numpy as np, onnxruntime as ort
import os as _os
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[2]
H0 = _os.environ.get('W3_DATA', str(_REPO.parent)); R = str(_REPO)
sys.path[:0] = [R + '/src', R + '/scripts', R + '/experiments/temporal_final', R + '/experiments/yolov9_v10/workflow']
from live_flame import prepare, decode
import temporal_final as TF
from geometry import Camera
SP = os.environ.get('W3_OUT', os.path.join(H0, 'w3_demo')); CACHE = os.path.join(SP, 'fire6_full_dets_10fps.json')
RULE = json.load(open(H0 + '/W3_starter_V12/rule_v12.json'))
cam = Camera.load(R + '/experiments/w3_validation/calibrations/camera_fire6.json'); OFF = 0.415
SEG = [(11, 25, 0.5), (26, 41, 1.0), (42, 57, 1.5), (58, 71, 2.0), (72, 86, 2.5), (88, 104, 3.0)]
BASE = {0.5: (1102, 1065), 1.0: (1021, 741), 1.5: (987, 572), 2.0: (973, 460), 2.5: (963, 386), 3.0: (954, 331)}
if not os.path.exists(CACHE):
    s = ort.InferenceSession(H0 + '/W3_starter_V12/models/flame_v12_full.onnx', providers=['CPUExecutionProvider'])
    cap = cv2.VideoCapture(H0 + '/w3_demo/drive_videos/fire6.mp4'); F = []; nxt = 0.0
    while True:
        ok, f = cap.read()
        if not ok: break
        t = cap.get(0) / 1000
        if t + 1e-6 < nxt: continue
        nxt += 0.1
        x, r = prepare(f, 640)
        F.append(dict(t=t, dets=[list(map(float, b[:5])) for b in decode(s.run(None, {s.get_inputs()[0].name: x})[0], f, r, 0.05, 'yolov9')]))
    json.dump(F, open(CACHE, 'w'))
F = json.load(open(CACHE)); to640 = 640 / 1920
def on_candle(d, m):
    if m is None: return None
    bx, by = BASE[m]; return d[0] - 40 <= bx <= d[2] + 40 and d[1] - 40 <= by <= d[3] + 40
times = [q['t'] for q in F]
mark = [next((m for a, b, m in SEG if a <= q['t'] < b), None) for q in F]
print(f"{'thr':>5} | no-flame 0-9s | " + " | ".join(f"{m:g} m det/alarm" for _, _, m in SEG) + " | off-candle boxes on marks | dist err max")
for thr in (0.40, 0.30, 0.25, 0.20, 0.15, 0.10):
    TF.BASE = thr
    per = TF.track(F, to640, None)
    al = TF.alarm(times, TF.raw_flags(times, per, RULE['window_s'], RULE['min_hit_ratio'], RULE['min_mean_score']), RULE['alarm_on_after_s'])
    nf = [i for i, t in enumerate(times) if t < 9]
    nf_box = sum(len(TF.gate(F[i]['dets'], to640, None)) for i in nf); nf_al = sum(al[i] for i in nf)
    cells, off, errs = [], 0, []
    for a, b, m in SEG:
        idx = [i for i, t in enumerate(times) if a <= t < b]
        hit = sum(any(on_candle(d, m) for d in TF.gate(F[i]['dets'], to640, None)) for i in idx)
        # alarm excluding the 3 s hold carried over from the previous mark
        idx2 = [i for i in idx if times[i] >= a + 3.0]
        cells.append(f"{hit:>3}/{len(idx)} {sum(al[i] for i in idx2):>3}/{len(idx2)}")
        for i in idx:
            for d in TF.gate(F[i]['dets'], to640, None):
                if on_candle(d, m): errs.append(abs(cam.box_to_floor(d[:4])['distance_m'] - OFF - m) / m)
                else: off += 1
    print(f"{thr:>5.2f} | box {nf_box:>3} alarm {nf_al:>3} | " + " | ".join(cells) + f" | {off:>4} | {100*np.median(errs):.1f}% med, {100*np.percentile(errs,95):.1f}% p95")
