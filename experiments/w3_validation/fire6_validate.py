"""fire_6 validation: pose fit from hand-read candle-base pixels, leave-one-out, detector distances."""
import sys, json, math, itertools
import numpy as np
sys.path.insert(0, sys.argv[1])
from geometry import Camera, intrinsics_from_hfov
W, H = 1920, 1080
D = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
BASE = np.array([[1102, 1065], [1021, 741], [987, 572], [973, 460], [963, 386], [954, 331]], float)  # candle base px
FLAME = np.array([1012, 709, 541, 440, 371, 313], float)                                              # flame base v

def cam(h, t, hfov):
    return Camera((W, H), intrinsics_from_hfov(W, H, hfov), pose={"height_m": h, "tilt_deg": t, "roll_deg": 0.0})

def dist(c, px, z=0.0):
    g = c.pixel_to_floor(np.atleast_2d(px), z)
    return np.hypot(g[:, 0], g[:, 1])

def fit(idx, hfov=None):
    """least squares in metres-relative error over chosen points; grid then refine."""
    best = None
    hf_list = [hfov] if hfov else np.arange(50, 85, 1.0)
    for hf in hf_list:
        for h in np.arange(0.3, 2.0, 0.02):
            for t in np.arange(5, 70, 0.5):
                c = cam(h, t, hf)
                e = dist(c, BASE[idx]) / D[idx] - 1
                if np.any(~np.isfinite(e)): continue
                s = float(np.sum(e ** 2))
                if best is None or s < best[0]: best = (s, h, t, hf)
    s, h, t, hf = best
    for step in (0.005, 0.001):  # local refine
        for _ in range(3):
            for dh, dt, dfv in itertools.product((-step, 0, step), (-step * 10, 0, step * 10), (0,) if hfov else (-step * 50, 0, step * 50)):
                c = cam(h + dh, t + dt, hf + dfv)
                e = dist(c, BASE[idx]) / D[idx] - 1
                if np.any(~np.isfinite(e)): continue
                ss = float(np.sum(e ** 2))
                if ss < s: s, h, t, hf = ss, h + dh, t + dt, hf + dfv
    return h, t, hf

out = {}
# 1) all six points, free hfov: does a single flat-floor camera explain all marks?
h, t, hf = fit(list(range(6)))
c = cam(h, t, hf)
est = dist(c, BASE)
print(f"ALL-6 fit: height {h:.3f} m, tilt {t:.2f} deg, hfov {hf:.1f} deg")
for d, e in zip(D, est): print(f"   {d:.1f} m -> {e:.3f} m  ({(e/d-1)*100:+.1f}%)")
out["fit_all"] = dict(height_m=h, tilt_deg=t, hfov_deg=hf, est=est.tolist())
HF = hf
# 2) two-point calibration (1.0 & 2.5 m) at that hfov, test the other four
cal = [1, 4]
h2, t2, _ = fit(cal, HF)
c2 = cam(h2, t2, HF)
est2 = dist(c2, BASE)
print(f"\nCALIBRATE on 1.0 & 2.5 m only (height {h2:.3f}, tilt {t2:.2f}); held-out marks:")
for i, (d, e) in enumerate(zip(D, est2)):
    tag = "cal " if i in cal else "TEST"
    print(f"   {tag} {d:.1f} m -> {e:.3f} m  error {e-d:+.3f} m ({(e/d-1)*100:+.1f}%)")
out["two_point"] = dict(cal=[float(D[i]) for i in cal], height_m=h2, tilt_deg=t2, est=est2.tolist())
# 3) leave-one-out (5 marks fit, 1 predicted), fixed hfov
print("\nLEAVE-ONE-OUT (fit on other 5 marks):")
loo = []
for k in range(6):
    idx = [i for i in range(6) if i != k]
    hk, tk, _ = fit(idx, HF)
    e = dist(cam(hk, tk, HF), BASE[[k]])[0]; loo.append(e)
    print(f"   {D[k]:.1f} m -> {e:.3f} m  error {e-D[k]:+.3f} m ({(e/D[k]-1)*100:+.1f}%)")
out["loo"] = loo
# 4) detector end-to-end with the 2-point camera; contact height from 1.5 m detections (calibration), test 0.5 & 3 m
dets = json.load(open(sys.argv[2]))
def med_bottom(seg):
    b = np.array([x for x in dets[seg]["boxes"] if x[4] >= 0.40])
    return b
zs = np.linspace(0, h2 - 0.02, 400)
b3 = med_bottom(2); bc = np.c_[(b3[:, 0] + b3[:, 2]) / 2, b3[:, 3]]
zc = zs[np.argmin([abs(np.median(dist(c2, bc, z)) - 1.5) for z in zs])]
c2.contact_height_m = float(zc)
print(f"\nDETECTOR (V12 @0.40), contact height set from the 1.5 m detections = {zc*100:.1f} cm")
det_rows = []
for seg in range(6):
    b = med_bottom(seg)
    if len(b) == 0:
        print(f"   {D[seg]:.1f} m: no detections >= 0.40"); det_rows.append(None); continue
    e = dist(c2, np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]], zc)
    tag = "cal " if seg == 2 else "TEST"
    print(f"   {tag} {D[seg]:.1f} m: {len(b)} frames, median {np.median(e):.3f} m (range {e.min():.2f}-{e.max():.2f}), error {(np.median(e)/D[seg]-1)*100:+.1f}%")
    det_rows.append(dict(n=len(b), median=float(np.median(e)), lo=float(e.min()), hi=float(e.max())))
out["detector"] = dict(contact_height_m=float(zc), rows=det_rows)
c2.save(sys.argv[3]); json.dump(out, open(sys.argv[4], "w"), indent=1)
print("\nsaved", sys.argv[3])
