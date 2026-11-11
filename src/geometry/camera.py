"""Camera model + ground-plane geometry (pixel <-> floor metres).

Conventions
-----------
* Floor frame: origin = point on the floor directly below the camera, +X right,
  +Y forward (camera heading projected on the floor), +Z up (metres).
* Pixels are original-frame pixels (u right, v down) at the *current* resolution.
* tilt_deg: camera pitch below the horizontal (positive = looking down).
  roll_deg: rotation about the optical axis; positive = camera rolled clockwise as
  seen from behind (the horizon appears rotated counter-clockwise in the image).
* bearing_deg: 0 = straight ahead (+Y), positive = to the right (+X).
* The homography maps floor metres (X, Y, 1) -> *undistorted* pixels at the calibrated
  resolution. Points above the horizon (or beyond max_range_m) give NaN, never garbage.
* Rescaling assumes the new frame is the same field of view resized (no crop/letterbox).

camera.json schema (schema=1):
  image_size [w,h]            resolution K / homography were calibrated at
  K 3x3  (or hfov_deg)        intrinsics; hfov_deg builds a pinhole K for rough work
  dist [k1,k2,p1,p2,k3,...]   OpenCV distortion (optional, default zeros)
  pose {height_m,tilt_deg,roll_deg}   extrinsics (optional if homography given)
  homography 3x3              floor metres -> undistorted pixels (optional; wins over pose)
  max_range_m                 optional, default 200
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

SCHEMA = 1
_EPS = 1e-9


def intrinsics_from_hfov(w, h, hfov_deg):
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def rotation_from_pose(tilt_deg, roll_deg=0.0):
    """Rows are the camera x/y/z axes expressed in the floor frame (floor -> camera)."""
    t, r = math.radians(tilt_deg), math.radians(roll_deg)
    x = np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, math.cos(t), -math.sin(t)])
    y = np.cross(z, x)
    xr = math.cos(r) * x + math.sin(r) * y
    yr = -math.sin(r) * x + math.cos(r) * y
    return np.stack([xr, yr, z])


def homography_from_pose(K, height_m, tilt_deg, roll_deg=0.0):
    R = rotation_from_pose(tilt_deg, roll_deg)
    t = -R @ np.array([0.0, 0.0, float(height_m)])
    return np.asarray(K, float) @ np.stack([R[:, 0], R[:, 1], t], axis=1)


def fit_homography(pixels, floor_xy):
    """>=4 (undistorted pixel, floor metre) pairs -> H (floor->pixel), sign-normalised so
    that floor points in front of the camera have positive projective scale."""
    px = np.asarray(pixels, np.float64).reshape(-1, 2)
    fl = np.asarray(floor_xy, np.float64).reshape(-1, 2)
    if len(px) < 4 or len(px) != len(fl):
        raise ValueError("need >=4 matching pixel/floor point pairs")
    H, _ = cv2.findHomography(fl, px, 0)
    if H is None:
        raise ValueError("homography fit failed (degenerate points?)")
    H = H / H[2, 2]
    w = np.c_[fl, np.ones(len(fl))] @ H.T
    return H if np.median(w[:, 2]) > 0 else -H


class Camera:
    def __init__(self, image_size, K, dist=None, pose=None, homography=None, max_range_m=200.0):
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64).ravel()
        self.pose = dict(pose) if pose else None
        self.max_range_m = float(max_range_m)
        if homography is not None:
            self.H = np.asarray(homography, np.float64).reshape(3, 3)
            self.source = "homography"
        elif self.pose:
            self.H = homography_from_pose(self.K, self.pose["height_m"], self.pose["tilt_deg"],
                                          self.pose.get("roll_deg", 0.0))
            self.source = "pose"
        else:
            raise ValueError("camera needs a homography or a pose")
        self._Hinv = np.linalg.inv(self.H)

    # ---- io -------------------------------------------------------------
    @classmethod
    def from_dict(cls, d):
        w, h = d["image_size"]
        K = d.get("K")
        if K is None:
            if "hfov_deg" not in d:
                raise ValueError("camera.json needs K or hfov_deg")
            K = intrinsics_from_hfov(w, h, d["hfov_deg"])
        return cls((w, h), K, d.get("dist"), d.get("pose"), d.get("homography"),
                   d.get("max_range_m", 200.0))

    def to_dict(self):
        d = {"schema": SCHEMA, "image_size": list(self.image_size), "K": self.K.tolist(),
             "dist": self.dist.tolist(), "max_range_m": self.max_range_m}
        if self.pose:
            d["pose"] = self.pose
        if self.source == "homography":
            d["homography"] = self.H.tolist()
        return d

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # ---- resolution -----------------------------------------------------
    def with_resolution(self, w, h):
        """Same camera at another frame size (pure resize of the full field of view)."""
        S = np.diag([w / self.image_size[0], h / self.image_size[1], 1.0])
        hom = S @ self.H if self.source == "homography" else None
        return Camera((w, h), S @ self.K, self.dist, self.pose, hom, self.max_range_m)

    # ---- pixel helpers --------------------------------------------------
    def undistort_points(self, px):
        p = np.asarray(px, np.float64).reshape(-1, 1, 2)
        if not np.any(self.dist):
            return p.reshape(-1, 2).copy()
        return cv2.undistortPoints(p, self.K, self.dist, P=self.K).reshape(-1, 2)

    def distort_points(self, upx):
        """Undistorted pixels -> raw (distorted) pixels."""
        u = np.asarray(upx, np.float64).reshape(-1, 2)
        if not np.any(self.dist):
            return u.copy()
        n = (u - self.K[:2, 2]) / np.array([self.K[0, 0], self.K[1, 1]])
        pts = np.c_[n, np.ones(len(n))]
        out, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K, self.dist)
        return out.reshape(-1, 2)

    # ---- floor <-> pixel ------------------------------------------------
    def pixel_to_floor(self, px):
        """Nx2 raw pixels -> Nx2 floor metres (x,y); NaN above horizon / beyond max range."""
        u = self.undistort_points(px)
        q = np.c_[u, np.ones(len(u))] @ self._Hinv.T
        W = q[:, 2]
        ok = np.isfinite(W) & (W > _EPS)
        out = np.full((len(u), 2), np.nan)
        out[ok] = q[ok, :2] / W[ok, None]
        out[np.hypot(out[:, 0], out[:, 1]) > self.max_range_m] = np.nan
        return out

    def floor_to_pixel(self, xy):
        """Nx2 floor metres -> Nx2 raw pixels (may lie outside the image); NaN behind camera."""
        f = np.asarray(xy, np.float64).reshape(-1, 2)
        p = np.c_[f, np.ones(len(f))] @ self.H.T
        ok = p[:, 2] > _EPS
        out = np.full((len(f), 2), np.nan)
        if ok.any():
            out[ok] = self.distort_points(p[ok, :2] / p[ok, 2:3])
        return out

    def horizon_line(self):
        """(a,b,c): a*u+b*v+c = 0 in undistorted pixels is the horizon; the floor side has
        a*u+b*v+c > 0."""
        return self._Hinv[2].copy()

    # ---- detections -----------------------------------------------------
    def box_to_floor(self, box):
        """xyxy box (original-frame px) -> dict from its bottom-centre; None values when the
        contact point is above the horizon."""
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        xy = self.pixel_to_floor([[(x1 + x2) / 2.0, y2]])[0]
        if not np.all(np.isfinite(xy)):
            return {"distance_m": None, "bearing_deg": None, "x_m": None, "y_m": None}
        x, y = float(xy[0]), float(xy[1])
        return {"distance_m": math.hypot(x, y), "bearing_deg": math.degrees(math.atan2(x, y)),
                "x_m": x, "y_m": y}
