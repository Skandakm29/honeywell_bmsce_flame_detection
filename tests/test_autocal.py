import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geometry import Camera, autocal  # noqa: E402

SIZE = (1920, 1080)


def cam(tilt, roll, f_hfov=66.0, h=1.5):
    return Camera.from_dict({"image_size": list(SIZE), "hfov_deg": f_hfov,
                             "pose": {"height_m": h, "tilt_deg": tilt, "roll_deg": roll}})


def scene_segments(c, n_vert=14, noise=0.3, seed=0, horiz=True, n_y=14):
    """Pixel segments of a synthetic room: vertical posts + horizontal lines along X and Y."""
    rng = np.random.default_rng(seed)
    w, h = SIZE
    segs = []

    def add(p, q):
        pts = c.floor_to_pixel([p[:2], q[:2]], 0.0) * np.nan  # placeholder for shape
        a = c.floor_to_pixel([p[:2]], p[2])[0]
        b = c.floor_to_pixel([q[:2]], q[2])[0]
        if np.isfinite(a).all() and np.isfinite(b).all() and all(
                0 <= v[0] < w and 0 <= v[1] < h for v in (a, b)):
            segs.append(np.r_[a, b] + rng.normal(0, noise, 4))

    def fill(target, make):  # draw random lines until `target` of them landed inside the frame
        for _ in range(2000):
            if len(segs) >= target:
                return
            make()

    def vert():
        x, y = rng.uniform(-2.5, 2.5), rng.uniform(1.5, 6)
        z0 = rng.uniform(0.05, 0.6)
        add((x, y, z0), (x, y, z0 + rng.uniform(0.4, 1.0)))

    def alongy():
        x, z, y0 = rng.uniform(-3, 3), rng.uniform(0.1, 1.4), rng.uniform(1.5, 4)
        add((x, y0, z), (x, y0 + rng.uniform(2.5, 5), z))

    def alongx():
        y, z, x0 = rng.uniform(2.0, 7), rng.uniform(0.1, 1.4), rng.uniform(-3, 0)
        add((x0, y, z), (x0 + rng.uniform(1, 3), y, z))

    fill(n_vert, vert)
    if horiz:
        fill(n_vert + n_y, alongy)
        fill(n_vert + n_y + 8, alongx)
    return np.array(segs)


@pytest.mark.parametrize("tilt,roll", [(20.0, 0.0), (8.0, 2.0), (35.0, -3.0), (40.0, 1.0)])
def test_recovers_tilt_roll_given_f(tilt, roll):
    c = cam(tilt, roll)
    res = autocal.estimate_from_segments(scene_segments(c), c.K)
    assert res.tilt_deg == pytest.approx(tilt, abs=0.5)
    assert res.roll_deg == pytest.approx(roll, abs=0.5)
    assert res.report["tilt_std_deg"] < 1.0 and res.report["n_vertical_inliers"] >= 8


def test_estimates_focal_length_from_two_orthogonal_vps():
    c = cam(20.0, 1.0)
    K_wrong = c.K.copy()
    K_wrong[0, 0] = K_wrong[1, 1] = c.K[0, 0] * 1.12
    res = autocal.estimate_from_segments(scene_segments(c, n_y=40, noise=0.15), K_wrong, estimate_f=True)
    assert res.f_est == pytest.approx(c.K[0, 0], rel=0.02)
    assert res.tilt_deg == pytest.approx(20.0, abs=0.5)
    assert res.roll_deg == pytest.approx(1.0, abs=0.5)
    # without estimate_f the given (wrong) f is used and reported
    res0 = autocal.estimate_from_segments(scene_segments(c), K_wrong)
    assert res0.f_used == pytest.approx(K_wrong[0, 0])


def test_refuses_with_too_few_vertical_lines():
    c = cam(20.0, 0.0)
    segs = scene_segments(c, n_vert=4, horiz=False)
    with pytest.raises(autocal.AutoCalError, match="vertical"):
        autocal.estimate_from_segments(segs, c.K)


def test_refuses_unstable_estimate():
    c = cam(20.0, 0.0)
    segs = scene_segments(c, n_vert=12, noise=6.0, horiz=False)  # very noisy short lines
    with pytest.raises(autocal.AutoCalError):
        autocal.estimate_from_segments(segs, c.K, max_std_deg=0.05)


def test_rendered_image_detector_path():
    c = cam(18.0, 1.0)
    segs = scene_segments(c, n_vert=16, noise=0.0)
    img = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    for s in segs:
        cv2.line(img, (int(round(s[0])), int(round(s[1]))), (int(round(s[2])), int(round(s[3]))),
                 (230, 230, 230), 3, cv2.LINE_AA)
    res = autocal.auto_calibrate(img, c.K)
    assert res.tilt_deg == pytest.approx(18.0, abs=1.0)
    assert res.roll_deg == pytest.approx(1.0, abs=1.0)
    assert autocal.draw_debug(img, res).shape == img.shape
