#!/usr/bin/env python
"""Camera calibration tool -> camera.json (schema in src/geometry/camera.py).

  intrinsics  checkerboard images/video -> K, dist, RMS reprojection error (px)
  floor       >=4 pixel<->floor-metre pairs (JSON/CSV or clicks) -> homography + per-point error (m)
  pose        camera height/tilt/roll entered by hand -> pose in camera.json

Each subcommand merges into --out (default camera.json) so the three can be run in sequence.
Floor frame for `floor`: X right, Y forward, Z up (right-handed); the origin can be anywhere,
the camera foot/heading/height are recovered from the homography (needs K from `intrinsics`
or --hfov-deg).
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geometry import Camera, fit_homography, intrinsics_from_hfov  # noqa: E402


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"schema": 1}


def _save(path, d):
    Path(path).write_text(json.dumps(d, indent=2))


def _parse_size(s):
    w, h = s.lower().split("x")
    return [int(w), int(h)]


# ---------------------------------------------------------------- intrinsics
def _frames(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        i = n = 0
        while n < args.max_frames:
            ok, f = cap.read()
            if not ok:
                break
            if i % args.step == 0:
                n += 1
                yield f"{args.video}#{i}", f
            i += 1
    else:
        for path in sorted(sum((glob.glob(g) for g in args.images), [])):
            f = cv2.imread(path)
            if f is not None:
                yield path, f


def cmd_intrinsics(args):
    pattern = (args.cols, args.rows)  # inner corners
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_m
    obj, img, names, size = [], [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    for name, frame in _frames(args):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif size != (gray.shape[1], gray.shape[0]):
            print(f"skip {name}: size differs", file=sys.stderr)
            continue
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj.append(objp)
        img.append(corners)
        names.append(name)
    if len(obj) < 5:
        sys.exit(f"only {len(obj)} usable views; need >=5 (ideally 15+ at varied tilts)")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None)
    per = []
    for o, c, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((proj - c) ** 2, axis=2)))))
    d = _load(args.out)
    d.update({"image_size": list(size), "K": K.tolist(), "dist": dist.ravel().tolist(),
              "intrinsics_rms_px": float(rms), "intrinsics_views": len(obj)})
    d.pop("hfov_deg", None)
    _save(args.out, d)
    print(f"{len(obj)} views, RMS reprojection error {rms:.3f} px (worst view "
          f"{max(per):.3f} px: {names[int(np.argmax(per))]})")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} -> {args.out}")
    return rms


# --------------------------------------------------------------------- floor
def _read_points(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        rows = json.loads(p.read_text())
        rows = rows.get("points", rows) if isinstance(rows, dict) else rows
        return np.array([[r["u"], r["v"], r["x"], r["y"]] if isinstance(r, dict) else r
                         for r in rows], float)
    with open(p, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].strip().startswith(("#", "u"))]
    return np.array(rows, float)


def _click_points(image):
    img = cv2.imread(image)
    if img is None:
        sys.exit(f"cannot read {image}")
    pts = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, str(len(pts)), (x + 6, y - 6), 0, 0.7, (0, 255, 255), 2)

    cv2.namedWindow("click floor points (Enter=done)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("click floor points (Enter=done)", on_mouse)
    while True:
        cv2.imshow("click floor points (Enter=done)", img)
        if cv2.waitKey(30) in (13, 10, 27):
            break
    cv2.destroyAllWindows()
    rows = []
    for i, (u, v) in enumerate(pts, 1):
        x, y = input(f"point {i} at pixel ({u},{v}): floor x,y in metres (X right, Y fwd): ").split(",")
        rows.append([u, v, float(x), float(y)])
    return np.array(rows, float)


def _camera_for_floor(d, size_arg, hfov):
    if "K" not in d:
        if not (hfov and size_arg):
            sys.exit("no K in camera file: run `intrinsics` first or pass --hfov-deg and --size WxH")
        d["image_size"] = _parse_size(size_arg)
        d["K"] = intrinsics_from_hfov(*d["image_size"], hfov).tolist()
    return d


def cmd_floor(args):
    d = _camera_for_floor(_load(args.out), args.size, args.hfov_deg)
    pts = _click_points(args.image) if args.interactive else _read_points(args.points)
    if pts.ndim != 2 or pts.shape[1] != 4 or len(pts) < 4:
        sys.exit("need >=4 rows of u,v,x_m,y_m")
    d.pop("pose", None) if args.drop_pose else None
    tmp = Camera(d["image_size"], d["K"], d.get("dist"), pose={"height_m": 1, "tilt_deg": 0})
    und = tmp.undistort_points(pts[:, :2])
    H = fit_homography(und, pts[:, 2:])
    d["homography"] = H.tolist()
    cam = Camera.from_dict(d)  # also validates decomposition
    est = cam.pixel_to_floor(pts[:, :2], 0.0)
    err = np.hypot(*(est - pts[:, 2:]).T)
    for (u, v, x, y), e in zip(pts, err):
        print(f"  px({u:.0f},{v:.0f}) -> floor({x:.3f},{y:.3f}) m   error {e * 100:.2f} cm")
    print(f"reprojection error: RMS {np.sqrt(np.mean(err ** 2)) * 100:.2f} cm, max {err.max() * 100:.2f} cm")
    print(f"recovered camera: foot=({cam.camera_xy_m[0]:.3f},{cam.camera_xy_m[1]:.3f}) m "
          f"height={cam.height_m:.3f} m tilt={cam.tilt_deg:.1f} deg heading={cam.heading_deg:.1f} deg")
    d["floor_rms_m"] = float(np.sqrt(np.mean(err ** 2)))
    _save(args.out, d)
    print(f"-> {args.out}")
    return err


# ---------------------------------------------------------------------- pose
def cmd_pose(args):
    d = _load(args.out)
    if "K" not in d:
        if not (args.hfov_deg and args.size):
            sys.exit("no K in camera file: run `intrinsics` first or pass --hfov-deg and --size WxH")
        d["image_size"] = _parse_size(args.size)
        d["K"] = intrinsics_from_hfov(*d["image_size"], args.hfov_deg).tolist()
    d["pose"] = {"height_m": args.height, "tilt_deg": args.tilt, "roll_deg": args.roll}
    d.pop("homography", None)  # a homography would override the pose
    Camera.from_dict(d)
    _save(args.out, d)
    print(f"pose height={args.height} m tilt={args.tilt} deg roll={args.roll} deg -> {args.out}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="camera.json", help="camera.json to create/merge into")

    p = sub.add_parser("intrinsics")
    common(p)
    p.add_argument("--images", nargs="+", help="image paths/globs")
    p.add_argument("--video")
    p.add_argument("--cols", type=int, required=True, help="inner corners per row")
    p.add_argument("--rows", type=int, required=True, help="inner corners per column")
    p.add_argument("--square-m", type=float, default=0.025)
    p.add_argument("--step", type=int, default=15, help="video: use every Nth frame")
    p.add_argument("--max-frames", type=int, default=60)
    p.set_defaults(fn=cmd_intrinsics)

    p = sub.add_parser("floor")
    common(p)
    p.add_argument("--points", help="JSON or CSV with u,v,x,y (pixel, floor metres)")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--image", help="frame to click on (with --interactive)")
    p.add_argument("--hfov-deg", type=float)
    p.add_argument("--size", help="WxH if camera file has no K")
    p.add_argument("--drop-pose", action="store_true")
    p.set_defaults(fn=cmd_floor)

    p = sub.add_parser("pose")
    common(p)
    p.add_argument("--height", type=float, required=True, help="camera height above floor, m")
    p.add_argument("--tilt", type=float, required=True, help="degrees below horizontal")
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--hfov-deg", type=float)
    p.add_argument("--size", help="WxH if camera file has no K")
    p.set_defaults(fn=cmd_pose)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "intrinsics" and not (args.images or args.video):
        sys.exit("give --images or --video")
    if args.cmd == "floor" and not (args.points or args.interactive):
        sys.exit("give --points or --interactive --image")
    return args.fn(args)


if __name__ == "__main__":
    main()
