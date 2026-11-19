"""fire_6: detector distances with the offset-corrected calibration + annotated video (real 0.40 threshold)."""
import sys, json
import cv2, numpy as np, onnxruntime as ort
sys.path[:0] = [sys.argv[1] + '/src', sys.argv[1] + '/scripts']
from geometry import Camera, intrinsics_from_hfov
from geometry.overlay import RingOverlay
from live_flame import prepare, decode
W, H = 1920, 1080; OFF = 0.415
cam = Camera((W, H), intrinsics_from_hfov(W, H, 66.0), pose={"height_m": 0.77, "tilt_deg": 20.75, "roll_deg": 0.0})
D = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]; SEG = [(11, 25), (26, 41), (42, 57), (58, 71), (72, 86), (88, 104)]
dets = json.load(open(sys.argv[2]))
def d_of(b, z):
    g = cam.pixel_to_floor(np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]], z); return np.hypot(g[:, 0], g[:, 1]) - OFF
# contact height from the 1.5 m detections only (calibration), then test the rest
b15 = np.array([x for x in dets[2]['boxes'] if x[4] >= 0.4]); zs = np.linspace(0, 0.3, 601)
z = float(zs[np.argmin([abs(np.median(d_of(b15, zz)) - 1.5) for zz in zs])])
cam.contact_height_m = z
print(f"contact height (from 1.5 m detections) = {z*100:.1f} cm")
rows = []
for k, d in enumerate(D):
    b = np.array([x for x in dets[k]['boxes'] if x[4] >= 0.4])
    if not len(b): print(f"  {d} m: not detected"); rows.append(None); continue
    e = d_of(b, z); m = float(np.median(e))
    print(f"  {'cal ' if k == 2 else 'TEST'} {d} m: {len(b)} frames  median {m:.3f} m  ({(m/d-1)*100:+.1f}%)  range {e.min():.2f}-{e.max():.2f}")
    rows.append(dict(true=d, n=len(b), median=m, lo=float(e.min()), hi=float(e.max())))
cam.save(sys.argv[3]); json.dump(dict(offset_m=OFF, contact_height_m=z, rows=rows), open(sys.argv[4], 'w'), indent=1)
# render
sess = ort.InferenceSession(sys.argv[5], providers=['CPUExecutionProvider'])
cap = cv2.VideoCapture(sys.argv[6]); fps = cap.get(5)
ov = RingOverlay(cam, W, H, radii=[r + OFF for r in D], labels=False)
vw = cv2.VideoWriter(sys.argv[7], cv2.VideoWriter_fourcc(*'mp4v'), fps / 2, (W // 2, H // 2))
lab = {}
for r in D:  # label rings with tape distances at image centre column
    p = cam.floor_to_pixel([[0, r + OFF]])[0]; lab[r] = p
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    i += 1
    if i % 2: continue
    f = np.clip(((f / 255.0) ** 0.6) * 255, 0, 255).astype(np.uint8)  # brighten for viewing only (detector sees raw frame)
    raw = cap.get(0)
    ov.apply(f)
    for r, p in lab.items():
        cv2.putText(f, f"{r:g} m", (int(p[0]) + 260, int(p[1]) - 6), 0, 0.9, (0, 255, 255), 2)
    t = raw / 1000
    k = next((j for j, (a, b) in enumerate(SEG) if a <= t < b), None)
    if k is not None:
        bx = [x for x in dets[k]['boxes'] if x[4] >= 0.4]
    cv2.putText(f, f"t={t:5.1f}s" + (f"   candle on {D[k]:g} m mark" if k is not None else ""), (20, 50), 0, 1.2, (255, 255, 255), 3)
    vw.write(cv2.resize(f, (W // 2, H // 2)))
vw.release(); print("video ->", sys.argv[7])
