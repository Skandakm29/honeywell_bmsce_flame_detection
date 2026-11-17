"""Automatic tilt/roll (and optionally focal length) from scene lines.

Pipeline: median frame -> gamma/CLAHE -> line segments (LSD, else Canny+Hough) -> RANSAC for the
*vertical* vanishing point (door frames, wall corners, furniture edges) -> tilt/roll from the
direction K^-1 * vp. If a second, orthogonal (horizontal) VP is found, f is estimated from
f^2 = -(v1-c).(v2-c). Everything works in normalised image coordinates ((u-cx)/f0, (v-cy)/f0),
where a vanishing point in homogeneous form IS the 3D direction in camera coordinates.

Valid for |tilt| < 45 deg.
Conventions match geometry.camera: tilt positive = looking down; roll positive = camera rolled
clockwise (see camera.py). The camera height is NOT estimated: the user supplies it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


class AutoCalError(RuntimeError):
    """Auto-calibration refused (too few/unstable lines)."""


@dataclass
class AutoResult:
    tilt_deg: float
    roll_deg: float
    f_used: float                # focal length (px) the tilt/roll were computed with
    f_est: float | None          # focal length from two orthogonal VPs (None if not found)
    report: dict = field(default_factory=dict)
    segments: np.ndarray | None = None   # Nx4 pixel segments (all)
    vert_idx: np.ndarray | None = None
    horiz_idx: np.ndarray | None = None
    K: np.ndarray | None = None


# ----------------------------------------------------------------------- frames / segments
def median_frame(video, n_frames=30, t0=0.0, t1=None):
    """Median of n_frames evenly spaced in [t0, t1] seconds (denoises dark video)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = int(t0 * fps)
    b = min(total, int(t1 * fps)) if t1 else total
    idx = np.linspace(a, max(a, b - 1), n_frames).astype(int)
    frames = []
    for i in np.unique(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise AutoCalError(f"could not read frames from {video}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def preprocess(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    m = float(g.mean())
    if m < 80:  # dark: gamma up to a mean of about 0.35
        gamma = float(np.clip(math.log(0.35) / math.log(max(m, 1.0) / 255.0), 0.25, 1.0))
        g = (255.0 * (g / 255.0) ** gamma).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)


def detect_segments(gray, min_len=None):
    h, w = gray.shape
    min_len = min_len or 0.02 * math.hypot(w, h)
    segs = None
    try:
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
        if lines is not None:
            segs = lines.reshape(-1, 4)
    except (AttributeError, cv2.error):
        pass
    if segs is None:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 60, minLineLength=int(min_len), maxLineGap=8)
        segs = np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(float)
    segs = segs.astype(np.float64)
    ln = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[ln >= min_len]


# ------------------------------------------------------------------------------ geometry
def _prep(segs, K):
    c, f = K[:2, 2], K[0, 0]
    P1 = (segs[:, :2] - c) / f
    P2 = (segs[:, 2:] - c) / f
    h1, h2 = np.c_[P1, np.ones(len(P1))], np.c_[P2, np.ones(len(P2))]
    lines = np.cross(h1, h2)
    lines /= np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-12)
    d = P2 - P1
    ln = np.linalg.norm(d, axis=1)
    S = d / np.maximum(ln[:, None], 1e-12)
    return dict(lines=lines, M=(P1 + P2) / 2, S=S, L=ln)


def _resid(pre, d, idx=None):
    """Angular residual (rad) between each segment and the direction to VP d (x,y,z)."""
    M, S = pre["M"], pre["S"]
    if idx is not None:
        M, S = M[idx], S[idx]
    v = np.stack([d[0] - d[2] * M[:, 0], d[1] - d[2] * M[:, 1]], 1)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return np.arcsin(np.clip(np.abs(S[:, 0] * v[:, 1] - S[:, 1] * v[:, 0]), 0, 1))


def _refine(pre, idx, d0=None, iters=4):
    """Least-squares VP over segments idx. Algebraic SVD, then IRLS so each line's weight is
    length / distance-to-VP (algebraic residual l.d ~ distance * angular error)."""
    lines, L, M = pre["lines"][idx], pre["L"][idx], pre["M"][idx]
    w = L.copy()
    d = d0
    for _ in range(iters if d0 is not None or True else 1):
        _, _, vt = np.linalg.svd(lines * w[:, None], full_matrices=False)
        d = vt[-1] / np.linalg.norm(vt[-1])
        dist = np.hypot(d[0] - d[2] * M[:, 0], d[1] - d[2] * M[:, 1])
        w = L / np.maximum(dist, 1e-3)
    return d


def _ransac(pre, cand, accept, thr, iters, rng):
    """Best VP over random pairs among cand (indices). accept(d)->bool filters hypotheses."""
    best, best_score = None, 0.0
    if len(cand) < 2:
        return None, np.array([], int)
    lines, L = pre["lines"], pre["L"]
    for _ in range(iters):
        i, j = rng.choice(len(cand), 2, replace=False)
        d = np.cross(lines[cand[i]], lines[cand[j]])
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d /= n
        if not accept(d):
            continue
        r = _resid(pre, d, cand)
        score = L[cand][r < thr].sum()
        if score > best_score:
            best, best_score = d, score
    if best is None:
        return None, np.array([], int)
    for _ in range(2):  # refine + re-gather inliers
        inl = cand[_resid(pre, best, cand) < thr]
        if len(inl) < 2:
            break
        d = _refine(pre, inl)
        if accept(d if d @ best > 0 else -d):
            best = d if d @ best > 0 else -d
    inl = cand[_resid(pre, best, cand) < thr]
    return best, inl


def _tilt_roll(d, ratio=1.0):
    """Vertical VP direction (normalised coords, f = ratio*f0) -> (tilt_deg, roll_deg)."""
    u = np.array([d[0] / ratio, d[1] / ratio, d[2]], float)
    u /= np.linalg.norm(u)
    if u[1] > 0:
        u = -u  # world 'up' must point up in the image (negative y)
    a, b, c = u
    return math.degrees(math.asin(np.clip(-c, -1, 1))), math.degrees(math.atan2(-a, -b))


def estimate_from_segments(segs, K, thr_deg=1.5, iters=800, estimate_f=False, min_vert=8,
                           max_std_deg=1.0, boot=200, seed=0):
    """Segments (Nx4 pixels x1,y1,x2,y2) + intrinsics K -> AutoResult (raises AutoCalError)."""
    segs = np.asarray(segs, np.float64).reshape(-1, 4)
    K = np.asarray(K, np.float64)
    f0 = float(K[0, 0])
    rng = np.random.default_rng(seed)
    pre = _prep(segs, K)
    thr = math.radians(thr_deg)
    cand = np.nonzero(np.abs(pre["S"][:, 1]) > math.cos(math.radians(50)))[0]  # near-vertical in the image
    cos60 = math.cos(math.radians(60))

    def vert_ok(d):
        # near the camera y axis, and |dz| < sin45: the vertical VP is farther from the principal
        # point than the horizon VP only for |tilt| < 45 deg (beyond that the two are ambiguous)
        return abs(d[1]) > cos60 and abs(d[2]) < 0.7071

    dv, vin = _ransac(pre, cand, vert_ok, thr, iters, rng)
    rep = dict(n_segments=len(segs), n_vertical_candidates=int(len(cand)),
               n_vertical_inliers=int(len(vin)))
    if dv is None or len(vin) < min_vert:
        raise AutoCalError(f"only {len(vin)} vertical-line inliers (need >= {min_vert}; "
                           f"{len(segs)} segments found). Scene lacks usable vertical edges "
                           "or is too dark/blurred; use `pose` or `floor` instead.")
    rep["vertical_resid_deg_rms"] = float(np.degrees(np.sqrt(np.mean(_resid(pre, dv, vin) ** 2))))

    # second, orthogonal (horizontal) VP -> focal length
    rest = np.setdiff1d(np.arange(len(segs)), vin)
    xv, yv, zv = dv

    def horiz_ok(d):
        if abs(d[2]) < 0.02 or abs(zv) < 0.02:   # VP at infinity carries no f information
            return False
        r2 = -(xv * d[0] + yv * d[1]) / (zv * d[2])
        return 0.3 < r2 < 3.0

    dh, hin = _ransac(pre, rest, horiz_ok, thr, iters, rng)
    f_est = ratio = None
    rep["n_horizontal_inliers"] = int(len(hin))
    if dh is not None and len(hin) >= min_vert:
        ratio = math.sqrt(-(xv * dh[0] + yv * dh[1]) / (zv * dh[2]))
        f_est = f0 * ratio
        fs = []   # bootstrap std of f over resampled horizontal inliers (vertical VP held fixed)
        for _ in range(100):
            d = _refine(pre, rng.choice(hin, len(hin), replace=True))
            q = -(xv * d[0] + yv * d[1]) / (zv * d[2])
            if q > 0:
                fs.append(f0 * math.sqrt(q))
        rep["f_est_std_px"] = float(np.std(fs)) if len(fs) > 20 else None
        if rep["f_est_std_px"] is None or rep["f_est_std_px"] > 0.1 * f_est:
            rep["f_est_note"] = "horizontal VP too uncertain; f_est discarded"
            f_est = ratio = None
    else:
        hin = np.array([], int)
    rep["f_est_px"] = f_est
    use = estimate_f and f_est is not None
    r_used = ratio if use else 1.0
    tilt, roll = _tilt_roll(dv, r_used)

    # bootstrap std over resampled vertical inliers
    ts, rs = [], []
    for _ in range(boot):
        s = rng.choice(vin, len(vin), replace=True)
        d = _refine(pre, s)
        t, r = _tilt_roll(d, r_used)
        ts.append(t)
        rs.append(r)
    rep["tilt_std_deg"], rep["roll_std_deg"] = float(np.std(ts)), float(np.std(rs))
    rep["f_used_px"] = f0 * r_used
    if max(rep["tilt_std_deg"], rep["roll_std_deg"]) > max_std_deg:
        raise AutoCalError(f"unstable: bootstrap std tilt {rep['tilt_std_deg']:.2f} deg, roll "
                           f"{rep['roll_std_deg']:.2f} deg (> {max_std_deg}); {len(vin)} inliers")
    K2 = K.copy()
    if use:
        K2[0, 0] = K2[1, 1] = f0 * ratio
    return AutoResult(tilt, roll, f0 * r_used, f_est, rep, segs, vin, hin, K2)


def auto_calibrate(image_bgr, K, **kw):
    """Image + K -> AutoResult. See estimate_from_segments for options."""
    return estimate_from_segments(detect_segments(preprocess(image_bgr)), K, **kw)


def draw_debug(image_bgr, res: AutoResult):
    """Segments: grey = outlier, red = vertical-VP inliers, green = horizontal-VP inliers."""
    out = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    for k, s in enumerate(res.segments):
        cv2.line(out, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (110, 110, 110), 1, cv2.LINE_AA)
    for idx, col in ((res.horiz_idx, (0, 220, 0)), (res.vert_idx, (0, 0, 255))):
        for k in idx:
            s = res.segments[k]
            cv2.line(out, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), col, 2, cv2.LINE_AA)
    t = f"tilt {res.tilt_deg:.2f} roll {res.roll_deg:.2f} f {res.f_used:.0f} (vert {len(res.vert_idx)}, horiz {len(res.horiz_idx)})"
    cv2.putText(out, t, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, t, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out
