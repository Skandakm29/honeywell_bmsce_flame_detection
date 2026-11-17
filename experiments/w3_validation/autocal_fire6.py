"""Auto-calibration vs manual calibration on a real clip (paths as args; no data in git).

  python experiments/w3_validation/autocal_fire6.py --video fire6.mp4 --height 0.77 --hfov-deg 66 \
      --manual camera_fire6.json --marks 0.5:1102,1065 1:1021,741 1.5:987,572 2:973,460 2.5:963,386 3:954,331 \
      --tape-offset 0.415 [--t0 0 --t1 9] [--estimate-f] [--debug dbg.png]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from geometry import Camera, autocal  # noqa: E402
from geometry import intrinsics_from_hfov  # noqa: E402
import cv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--hfov-deg", type=float, required=True)
    ap.add_argument("--manual", help="manual camera.json to compare against")
    ap.add_argument("--marks", nargs="*", default=[], help="tape_m:u,v candle-base pixels")
    ap.add_argument("--tape-offset", type=float, default=0.0, help="camera distance = tape + offset")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--estimate-f", action="store_true")
    ap.add_argument("--debug")
    a = ap.parse_args()

    img = autocal.median_frame(a.video, a.frames, a.t0, a.t1)
    h, w = img.shape[:2]
    K = intrinsics_from_hfov(w, h, a.hfov_deg)
    print(f"{a.video}  window {a.t0}-{a.t1 or 'end'} s  median of {a.frames} frames  {w}x{h}  f0={K[0,0]:.0f}")
    segs = autocal.detect_segments(autocal.preprocess(img))
    print(f"segments: {len(segs)}")
    try:
        res = autocal.estimate_from_segments(segs, K, estimate_f=a.estimate_f)
    except autocal.AutoCalError as e:
        print("AUTO-CALIBRATION REFUSED:", e)
        return
    r = res.report
    print(f"auto:   tilt {res.tilt_deg:.2f} (std {r['tilt_std_deg']:.2f})  roll {res.roll_deg:.2f} "
          f"(std {r['roll_std_deg']:.2f})  f_used {res.f_used:.0f}  vert inliers {r['n_vertical_inliers']} "
          f"resid {r['vertical_resid_deg_rms']:.2f} deg  horiz inliers {r['n_horizontal_inliers']}  "
          f"f_est {r['f_est_px']}  f_est_std {r.get('f_est_std_px')}")
    manual = Camera.load(a.manual) if a.manual else None
    if manual:
        mp = manual.pose
        print(f"manual: tilt {mp['tilt_deg']:.2f}  roll {mp.get('roll_deg', 0):.2f}  f {manual.K[0,0]:.0f}  "
              f"-> tilt diff {res.tilt_deg - mp['tilt_deg']:+.2f} deg, roll diff {res.roll_deg - mp.get('roll_deg', 0):+.2f} deg")
    if a.debug:
        cv2.imwrite(a.debug, autocal.draw_debug(img, res))
    if a.marks:
        cam = Camera.from_dict({"image_size": [w, h], "K": res.K.tolist(),
                                "pose": {"height_m": a.height, "tilt_deg": res.tilt_deg, "roll_deg": res.roll_deg},
                                "contact_height_m": manual.contact_height_m if manual else 0.0})
        print(f"\ncontact_height_m {cam.contact_height_m}; true distance = tape + {a.tape_offset}")
        print("tape  true_d   auto_d  err_m  err_%" + ("   manual_d  err_%" if manual else ""))
        for m in a.marks:
            tape, uv = m.split(":")
            u, v = map(float, uv.split(","))
            true = float(tape) + a.tape_offset
            box = [u - 5, v - 20, u + 5, v]
            d = cam.box_to_floor(box)["distance_m"]
            line = f"{float(tape):4.1f}  {true:6.3f}  " + (f"{d:6.3f}  {d-true:+.3f}  {100*(d-true)/true:+5.1f}" if d else "  n/a")
            if manual:
                dm = manual.box_to_floor(box)["distance_m"]
                line += f"   {dm:6.3f}  {100*(dm-true)/true:+5.1f}"
            print(line)


if __name__ == "__main__":
    main()
