"""fire_6: is zoom needed now that the temporal layer exists? Full frame (no zoom) vs zoom-320, both with the locked
temporal rule. Also counts boxes that are NOT on the candle (reflections / glow) using the hand-read candle-base pixels.
Writes fire6_fullframe_temporal.json and a single-view full-frame video (W3_fire6_fullframe_temporal.mp4) to W3_OUT."""
import os, sys, json
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('W3_DATA', REPO.parent))
OUT = Path(os.environ.get('W3_OUT', DATA / 'w3_demo'))
VIDEO = os.environ.get('W3_VIDEO', str(DATA / 'w3_demo/drive_videos/fire6.mp4'))
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts'), str(REPO / 'experiments/temporal_final'),
                str(REPO / 'experiments/yolov9_v10/workflow')]
from live_flame import prepare, decode  # noqa: E402
import temporal_final as TF  # noqa: E402
from geometry import Camera  # noqa: E402
from geometry.overlay import RingOverlay  # noqa: E402

RULE = json.load(open(DATA / 'W3_starter_V12/rule_v12.json'))
cam = Camera.load(str(REPO / 'experiments/w3_validation/calibrations/camera_fire6.json'))
OFF, W, H, C = 0.415, 1920, 1080, 320
SEG = [(11, 25, 0.5), (26, 41, 1.0), (42, 57, 1.5), (58, 71, 2.0), (72, 86, 2.5), (88, 104, 3.0)]
BASE = {0.5: (1102, 1065), 1.0: (1021, 741), 1.5: (987, 572), 2.0: (973, 460), 2.5: (963, 386), 3.0: (954, 331)}
s = ort.InferenceSession(str(DATA / 'W3_starter_V12/models/flame_v12_full.onnx'), providers=['CPUExecutionProvider'])


def det(img):
    x, r = prepare(img, 640)
    return [list(map(float, b[:5])) for b in decode(s.run(None, {s.get_inputs()[0].name: x})[0], img, r, 0.05, 'yolov9')]


def on_candle(d, mark):
    """Box counts as the candle if the candle-base pixel lies inside the box widened by 40 px (else: reflection/other)."""
    if mark is None:
        return False
    bx, by = BASE[mark]
    return d[0] - 40 <= bx <= d[2] + 40 and d[1] - 40 <= by <= d[3] + 40


cap = cv2.VideoCapture(VIDEO)
fr = {'full': [], 'zoom': []}; imgs = []; nxt = 0.0
while True:
    ok, f = cap.read()
    if not ok:
        break
    t = cap.get(0) / 1000
    if t + 1e-6 < nxt:
        continue
    nxt += 0.1
    fr['full'].append(dict(t=t, dets=det(f)))
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (9, 9), 0); _, _, _, (x, y) = cv2.minMaxLoc(g)
    x0 = min(max(0, x - C // 2), W - C); y0 = min(max(0, y - C // 2), H - C)
    fr['zoom'].append(dict(t=t, dets=[[b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0, b[4]]
                                      for b in det(np.ascontiguousarray(f[y0:y0 + C, x0:x0 + C]))]))
    imgs.append(cv2.resize(f, (1280, 720)))

res, alarms = {}, {}
for p, F in fr.items():
    to640 = 640 / max(W, H) if p == 'full' else 640 / C
    per = TF.track(F, to640, None); times = [q['t'] for q in F]
    al = TF.alarm(times, TF.raw_flags(times, per, RULE['window_s'], RULE['min_hit_ratio'], RULE['min_mean_score']),
                  RULE['alarm_on_after_s'])
    alarms[p] = al
    mark_of = [next((d for a, b, d in SEG if a <= q['t'] < b), None) for q in F]
    rows = []
    for a, b, mark in SEG:
        idx = [i for i, q in enumerate(F) if a <= q['t'] < b]
        g = {i: TF.gate(F[i]['dets'], to640, None) for i in idx}
        cand = [d for i in idx for d in g[i] if on_candle(d, mark)]
        other = sum(1 for i in idx for d in g[i] if not on_candle(d, mark))
        # first alarm frame that is ours (alarm on while a candle box was seen in the last 2 s of this segment)
        ds = [cam.box_to_floor(d[:4])['distance_m'] - OFF for d in cand]
        rows.append(dict(mark=mark, frames=len(idx), candle_hit_frames=sum(any(on_candle(d, mark) for d in g[i]) for i in idx),
                         alarm_frames=int(sum(al[i] for i in idx)), other_boxes=other,
                         dist=round(float(np.median(ds)), 3) if ds else None))
    nc = [i for i, q in enumerate(F) if q['t'] < 9]
    other_total = sum(1 for i, q in enumerate(F) for d in TF.gate(q['dets'], to640, None) if not on_candle(d, mark_of[i]))
    res[p] = dict(no_candle_alarm_frames=int(sum(al[i] for i in nc)), boxes_not_on_candle=other_total, rows=rows)
json.dump(res, open(OUT / 'fire6_fullframe_temporal.json', 'w'), indent=1)
for p in res:
    print(f"== {p}: alarm frames before candle {res[p]['no_candle_alarm_frames']}, boxes NOT on candle (whole clip) {res[p]['boxes_not_on_candle']}")
    for r in res[p]['rows']:
        print('  ', r)

# single-view full-frame video
S = 1280 / W
ov = RingOverlay(cam.with_resolution(1280, 720), 1280, 720, radii=[d + OFF for _, _, d in SEG], labels=False)
vw = cv2.VideoWriter(str(OUT / 'W3_fire6_fullframe_temporal.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 720))
to640 = 640 / max(W, H)
for i, im in enumerate(imgs):
    t = fr['full'][i]['t']; mark = next((d for a, b, d in SEG if a <= t < b), None)
    v = np.clip(((im / 255.0) ** 0.6) * 255, 0, 255).astype(np.uint8); ov.apply(v)
    g = TF.gate(fr['full'][i]['dets'], to640, None); txt = ''
    for d in g:
        cv2.rectangle(v, (int(d[0] * S), int(d[1] * S)), (int(d[2] * S), int(d[3] * S)), (0, 255, 0), 2)
        geo = cam.box_to_floor(d[:4]); txt = f"Flame at {geo['distance_m'] - OFF:.2f} m, bearing {geo['bearing_deg']:+.0f} deg  (score {d[4]:.2f})"
    st, col = ('ALARM', (0, 0, 255)) if alarms['full'][i] else (('WARNING', (0, 200, 255)) if g else ('CLEAR', (0, 200, 0)))
    cv2.rectangle(v, (0, 0), (1280, 56), (0, 0, 0), -1)
    cv2.putText(v, st, (14, 42), 0, 1.3, col, 3); cv2.putText(v, txt, (230, 38), 0, 0.75, (255, 255, 255), 2)
    cv2.rectangle(v, (0, 680), (1280, 720), (0, 0, 0), -1)
    cv2.putText(v, f"t={t:5.1f}s" + (f"   ground truth: {mark:g} m" if mark else "") +
                "   V12 full frame (no zoom) | ALARM = >=30% of 2 s, off after 3 s", (14, 708), 0, 0.6, (220, 220, 220), 1)
    vw.write(v)
vw.release(); print('video done')
