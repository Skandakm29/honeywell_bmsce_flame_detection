"""Range-ring overlay, detectability band and top-down map for a calibrated camera.

Rings are floor circles centred on the camera foot, projected through the camera model,
split where they go behind the camera / above the horizon / out of range of the image, and
guarded against lens-distortion fold-back (points whose undistorted normalised radius is
beyond the image corners' radius +5% are dropped before distorting).

CLI:  python src/geometry/overlay.py --camera camera.json (--image f.png | --video v.mp4 [--frame N])
          [--rings 1 2 3 5 10] [--flame-height-m 0.03 [--min-px 25 --contact-height-m 0.05]]
          [--out preview.png] [--map map.png] [--video-out out.mp4]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geometry.camera import Camera
else:
    from .camera import Camera

DEFAULT_RINGS = (1, 2, 3, 5, 10)
_EDGE = 3.0  # keep projected points within +-3x the frame size, drop the rest


def fit_to_frame(cam: Camera, w: int, h: int) -> Camera:
    return cam if tuple(cam.image_size) == (w, h) else cam.with_resolution(w, h)


def _corner_radius(cam: Camera) -> float:
    w, h = cam.image_size
    c = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    n = (c - cam.K[:2, 2]) / np.array([cam.K[0, 0], cam.K[1, 1]])
    return float(np.hypot(n[:, 0], n[:, 1]).max())


def project_floor(cam: Camera, xy, height_m=0.0):
    """Floor points -> raw pixels, NaN where behind camera, above-horizon-ish, outside the
    fold-back radius, or absurdly far outside the frame."""
    f = np.asarray(xy, float).reshape(-1, 2)
    P = np.c_[f, np.full(len(f), float(height_m))]
    p = (P - cam.C) @ cam.R.T @ cam.K.T
    ok = p[:, 2] > 1e-9
    out = np.full((len(f), 2), np.nan)
    if not ok.any():
        return out
    u = p[ok, :2] / p[ok, 2:3]
    n = (u - cam.K[:2, 2]) / np.array([cam.K[0, 0], cam.K[1, 1]])
    keep = np.hypot(n[:, 0], n[:, 1]) <= 1.05 * _corner_radius(cam)
    res = np.full((len(u), 2), np.nan)
    if keep.any():
        res[keep] = cam.distort_points(u[keep])
    w, h = cam.image_size
    bad = ~((res[:, 0] > -_EDGE * w) & (res[:, 0] < (1 + _EDGE) * w)
            & (res[:, 1] > -_EDGE * h) & (res[:, 1] < (1 + _EDGE) * h))
    res[bad] = np.nan
    out[ok] = res
    return out


def _split(px):
    """Split an Nx2 polyline at NaNs into runs of >=2 points."""
    good = np.isfinite(px).all(axis=1)
    runs, cur = [], []
    for p, g in zip(px, good):
        if g:
            cur.append(p)
        elif cur:
            runs.append(np.array(cur))
            cur = []
    if cur:
        runs.append(np.array(cur))
    return [r for r in runs if len(r) >= 2]


def ring_polylines(cam: Camera, radius_m: float, n: int = 720):
    """Image polylines (list of Nx2 arrays) of the floor circle of radius_m around the foot."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c = np.array(cam.camera_xy_m)
    xy = c + radius_m * np.c_[np.cos(a), np.sin(a)]
    px = project_floor(cam, xy)
    # a closed circle: if first and last points are both valid, close the loop
    runs = _split(np.vstack([px, px[:1]]))
    return runs


def _label_pos(runs, w, h):
    best = None
    for r in runs:
        inside = r[(r[:, 0] > 20) & (r[:, 0] < w - 60) & (r[:, 1] > 20) & (r[:, 1] < h - 10)]
        for p in inside:
            score = abs(p[0] - w / 2)
            if best is None or score < best[0]:
                best = (score, p)
    return None if best is None else best[1]


def detectable_distance(cam: Camera, flame_height_m: float, min_px: float = 25.0,
                        model_size: int = 640, contact_height_m=None, step: float = 0.02):
    """Largest distance (m, straight ahead) at which a flame of flame_height_m, whose base
    is at contact_height_m, is still >= min_px tall at the model input (letterbox scale
    r = model_size / max(W,H) applied). None if never detectable with its base in frame."""
    w, h = cam.image_size
    hc = cam.contact_height_m if contact_height_m is None else float(contact_height_m)
    r = model_size / max(w, h)
    hd = math.radians(cam.heading_deg)
    fwd = np.array([math.sin(hd), math.cos(hd)])
    d = np.arange(step, cam.max_range_m, step)
    xy = np.array(cam.camera_xy_m) + d[:, None] * fwd
    pb = cam.floor_to_pixel(xy, hc)
    pt = cam.floor_to_pixel(xy, hc + flame_height_m)
    px = r * np.abs(pt[:, 1] - pb[:, 1])
    ok = np.isfinite(px) & (px >= min_px) & (pb[:, 1] >= 0) & (pb[:, 1] <= h)
    if not ok.any():
        return None
    i = int(np.nonzero(ok)[0][0])
    while i + 1 < len(d) and ok[i + 1]:  # contiguous run from the nearest detectable point
        i += 1
    return float(d[i])


def floor_mask(cam: Camera, max_dist_m: float, stride: int = 8):
    """Boolean image mask of pixels whose floor point lies within max_dist_m of the foot."""
    w, h = cam.image_size
    us, vs = np.meshgrid(np.arange(0, w, stride) + stride / 2, np.arange(0, h, stride) + stride / 2)
    g = np.stack([us, vs], -1).reshape(-1, 2)
    xy = cam.pixel_to_floor(g, 0.0)
    rel = xy - np.array(cam.camera_xy_m)
    m = (np.hypot(rel[:, 0], rel[:, 1]) <= max_dist_m).reshape(us.shape)
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def detectability_mask(cam: Camera, flame_height_m: float, min_px: float = 25.0,
                       model_size: int = 640, contact_height_m=None, stride: int = 8):
    """Per-pixel detectability: treating each pixel as the flame base (on the plane
    z=contact_height_m), is the flame top >= min_px away at the model input (letterbox
    scale r = model_size / max(W,H))? Returns a full-resolution boolean mask."""
    w, h = cam.image_size
    hc = cam.contact_height_m if contact_height_m is None else float(contact_height_m)
    r = model_size / max(w, h)
    us, vs = np.meshgrid(np.arange(0, w, stride) + stride / 2, np.arange(0, h, stride) + stride / 2)
    g = np.stack([us, vs], -1).reshape(-1, 2)
    xy = cam.pixel_to_floor(g, hc)
    top = cam.floor_to_pixel(xy, hc + flame_height_m)  # NaN in -> NaN out
    px = r * np.hypot(top[:, 0] - g[:, 0], top[:, 1] - g[:, 1])
    m = (np.isfinite(px) & (px >= min_px)).reshape(us.shape)
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


class RingOverlay:
    """Rings + optional detectable band, rendered once for (camera, frame size, radii, band);
    apply() is then a cheap per-frame blend + copy of the line pixels."""

    def __init__(self, cam: Camera, w: int, h: int, radii=DEFAULT_RINGS, band_m=None,
                 band_mask=None, color=(0, 255, 255), thickness=2, labels=True,
                 band_color=(0, 200, 0), band_alpha=0.25, min_label_gap_px=18):
        cam = fit_to_frame(cam, w, h)
        layer = np.zeros((h, w, 3), np.uint8)
        self.band = band_mask if band_mask is not None else (floor_mask(cam, band_m) if band_m else None)
        if self.band is not None:
            self._keep = 1.0 - band_alpha
            self._add = (np.array(band_color, np.float32) * band_alpha)
            cnts, _ = cv2.findContours(self.band.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.polylines(layer, cnts, True, band_color, thickness + 1, cv2.LINE_AA)
        self.label_pos = []
        for r in sorted(radii):  # near to far
            runs = ring_polylines(cam, r)
            for run in runs:
                cv2.polylines(layer, [np.round(run).astype(np.int32)], False, color, thickness,
                              cv2.LINE_AA)
            if not labels:
                continue
            p = _label_pos(runs, w, h)
            if p is None or any(abs(p[1] - y) < min_label_gap_px for _, y in self.label_pos):
                continue
            self.label_pos.append((r, float(p[1])))
            txt, org = f"{r:g} m", (int(p[0]) + 4, int(p[1]) - 4)
            cv2.putText(layer, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(layer, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        self.layer = layer
        self.lines = layer.any(axis=2)
        self.size = (w, h)

    def apply(self, img):
        """Draw onto img (BGR uint8, same size as built for) in place and return it."""
        if img.shape[1::-1] != self.size:
            raise ValueError(f"frame {img.shape[1::-1]} != overlay {self.size}")
        if self.band is not None:
            img[self.band] = (img[self.band] * self._keep + self._add).astype(np.uint8)
        img[self.lines] = self.layer[self.lines]
        return img


def draw_rings(img, cam: Camera, radii=DEFAULT_RINGS, band_m=None, band_mask=None, **kw):
    """One-shot convenience (rebuilds everything: use RingOverlay for video)."""
    h, w = img.shape[:2]
    return RingOverlay(cam, w, h, radii, band_m=band_m, band_mask=band_mask, **kw).apply(img)


def to_camera_frame(cam: Camera, xy):
    """Floor (x,y) -> (right, forward) metres relative to camera foot and heading."""
    d = np.asarray(xy, float).reshape(-1, 2) - np.array(cam.camera_xy_m)
    hd = math.radians(cam.heading_deg)
    return np.c_[d[:, 0] * math.cos(hd) - d[:, 1] * math.sin(hd),
                 d[:, 0] * math.sin(hd) + d[:, 1] * math.cos(hd)]


def topdown_map(cam: Camera, radii=DEFAULT_RINGS, extent_m=None, size_px=800, band_m=None,
                points_xy=None):
    """Top-down image: camera at bottom-centre facing up, rings, field-of-view wedge, optional
    straight-ahead detectable radius and detection points given as floor (x,y) in the camera.json frame."""
    ext = float(extent_m or (max(radii) * 1.1))
    s = size_px / (2 * ext)  # px per metre; width spans 2*ext, height covers ext forward + margin
    W, H = size_px, int(size_px * 0.6)
    img = np.full((H, W, 3), 32, np.uint8)
    org = np.array([W / 2, H - 20.0])

    def pt(rf):
        return (int(org[0] + rf[0] * s), int(org[1] - rf[1] * s))

    # field-of-view wedge: farthest visible floor distance per bearing bin
    w, h = cam.image_size
    st = 8
    us, vs = np.meshgrid(np.arange(0, w, st) + st / 2, np.arange(0, h, st) + st / 2)
    g = np.stack([us, vs], -1).reshape(-1, 2)
    rf = to_camera_frame(cam, cam.pixel_to_floor(g, 0.0))
    ok = np.isfinite(rf).all(axis=1) & (np.hypot(rf[:, 0], rf[:, 1]) <= ext * 1.5)
    if ok.any():
        rf = rf[ok]
        brg = np.degrees(np.arctan2(rf[:, 0], rf[:, 1]))
        dist = np.hypot(rf[:, 0], rf[:, 1])
        bins = np.arange(-90, 91, 2.0)
        poly = [pt((0, 0))]
        for lo in bins[:-1]:
            sel = (brg >= lo) & (brg < lo + 2)
            if sel.any():
                dd = min(dist[sel].max(), ext * 1.4)
                b = math.radians(lo + 1)
                poly.append(pt((dd * math.sin(b), dd * math.cos(b))))
        ov = img.copy()
        cv2.fillPoly(ov, [np.array(poly, np.int32)], (70, 70, 40))
        img = cv2.addWeighted(ov, 0.7, img, 0.3, 0)
    if band_m:
        cv2.circle(img, pt((0, 0)), int(band_m * s), (0, 200, 0), 2, cv2.LINE_AA)
    for r in radii:
        cv2.circle(img, pt((0, 0)), int(r * s), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r:g} m", pt((0.05, r)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.circle(img, pt((0, 0)), 5, (255, 255, 255), -1)
    cv2.putText(img, "camera", (pt((0, 0))[0] + 8, pt((0, 0))[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if points_xy is not None and len(points_xy):
        for q in to_camera_frame(cam, points_xy):
            cv2.circle(img, pt(q), 6, (0, 0, 255), -1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--rings", type=float, nargs="+", default=list(DEFAULT_RINGS))
    ap.add_argument("--flame-height-m", type=float)
    ap.add_argument("--min-px", type=float, default=25.0)
    ap.add_argument("--model-size", type=int, default=640)
    ap.add_argument("--contact-height-m", type=float)
    ap.add_argument("--out", default="rings_preview.png")
    ap.add_argument("--map", help="write a top-down map PNG")
    ap.add_argument("--video-out", help="annotate the whole --video into this mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(argv)
    cam = Camera.load(a.camera)

    def band_for(c):
        """-> (straight-ahead range m, per-pixel mask) or (None, None)."""
        if not a.flame_height_m:
            return None, None
        d = detectable_distance(c, a.flame_height_m, a.min_px, a.model_size, a.contact_height_m)
        print(f"flame {a.flame_height_m} m tall: >= {a.min_px:g} px at {a.model_size}px input "
              f"straight-ahead range {'n/a' if d is None else f'{d:.2f} m'} (band shaded per pixel)")
        return d, detectability_mask(c, a.flame_height_m, a.min_px, a.model_size, a.contact_height_m)

    if a.video and a.video_out:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        c = fit_to_frame(cam, w, h)
        _, bmask = band_for(c)
        ov = RingOverlay(c, w, h, a.rings, band_mask=bmask)
        vw = cv2.VideoWriter(a.video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = 0
        while True:
            ok, f = cap.read()
            if not ok or (a.max_frames and n >= a.max_frames):
                break
            vw.write(ov.apply(f))
            n += 1
        vw.release()
        print(f"{n} frames -> {a.video_out}")
        return
    if a.video:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
        ok, img = cap.read()
        if not ok:
            sys.exit("cannot read frame")
    elif a.image:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit("cannot read image")
    else:
        sys.exit("give --image or --video")
    h, w = img.shape[:2]
    c = fit_to_frame(cam, w, h)
    dstraight, bmask = band_for(c)
    cv2.imwrite(a.out, draw_rings(img, c, a.rings, band_mask=bmask))
    print(f"-> {a.out}")
    if a.map:
        cv2.imwrite(a.map, topdown_map(c, a.rings, band_m=dstraight))
        print(f"-> {a.map}")


if __name__ == "__main__":
    main()
