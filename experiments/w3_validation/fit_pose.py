"""Fit camera height/tilt (pose model, assumed HFOV) to cup-base pixels at known distances."""
import sys, json, math
import numpy as np

sys.path.insert(0, sys.argv[1])
from geometry import Camera, intrinsics_from_hfov
HFOV = 69.0
W, H = 1920, 1080
K = intrinsics_from_hfov(W, H, HFOV)
fit = [(1.0, 1000, 750), (3.0, 967, 540)]          # (true m, u, v) cup base on floor
check = [(4.0, 932, 503)]
flames = [(1.0, 537), (3.0, 475), (4.0, 472)]       # flame-base v
def cam(p): return Camera((W, H), K, pose={"height_m": p[0], "tilt_deg": p[1], "roll_deg": 0.0})
def res(p):
    c = cam(p); return [c.floor_to_pixel([[0, d]])[0][1] - v for d, u, v in fit]
G=[(h,t) for h in np.arange(0.05,0.6,0.002) for t in np.arange(-10,10,0.05)]
class S: pass
s=S(); s.x=min(G,key=lambda p: sum(r*r for r in res(p)))
c = cam(s.x)
print(f"height={s.x[0]:.3f} m tilt={s.x[1]:.2f} deg (hfov {HFOV} assumed)")
for d, u, v in fit + check:
    e = c.box_to_floor([u - 5, v - 50, u + 5, v], 0.0)
    print(f"cup base at {d} m -> estimate {e['distance_m']:.2f} m bearing {e['bearing_deg']:.1f}")
for d, v in flames:   # height of flame base implied by the true distance
    # find z such that pixel_to_floor gives d
    zs = np.linspace(0, s.x[0] - 0.01, 2000)
    est = [c.pixel_to_floor([[960, v]], z)[0][1] for z in zs]
    z = zs[np.nanargmin(np.abs(np.array(est) - d))]
    print(f"flame base at {d} m implies contact height {z:.3f} m")
c.contact_height_m = 0.14
c.save(sys.argv[2]); print("saved", sys.argv[2])
