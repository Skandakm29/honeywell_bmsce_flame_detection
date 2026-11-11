import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geometry import Camera  # noqa: E402
from geometry import overlay  # noqa: E402

DIST = [-0.30, 0.10, 0.0, 0.0, -0.02]


def cam(tilt=20.0, h=1.5, dist=None, size=(1920, 1080), hfov=75.0):
    return Camera.from_dict({"image_size": list(size), "hfov_deg": hfov, "dist": dist,
                             "pose": {"height_m": h, "tilt_deg": tilt}})


@pytest.mark.parametrize("dist", [None, DIST])
def test_ring_points_lie_on_the_ring_and_never_above_horizon(dist):
    c = cam(dist=dist)
    w, h = c.image_size
    for r in (1, 2, 5, 10):
        runs = overlay.ring_polylines(c, r)
        assert runs, r
        for run in runs:
            assert np.isfinite(run).all()
            inside = run[(run[:, 0] > 0) & (run[:, 0] < w) & (run[:, 1] > 0) & (run[:, 1] < h)]
            back = c.pixel_to_floor(inside)
            assert np.isfinite(back).all()  # nothing above the horizon was drawn
            assert np.allclose(np.hypot(back[:, 0], back[:, 1]), r, atol=5e-3 * r)


def test_ring_is_split_behind_camera_and_far_ring_clipped_by_horizon():
    c = cam(tilt=5.0)  # shallow: far rings hug the horizon
    runs = overlay.ring_polylines(c, 2.0)
    assert len(runs) >= 1
    assert all(np.isfinite(r).all() for r in runs)
    # the full circle is not drawn: the half behind the camera is dropped
    assert sum(len(r) for r in runs) < 720


def test_fold_back_guard_drops_far_outside_points():
    c = cam(dist=[0.2, 0.5, 0, 0, 0.8])  # nasty polynomial that folds back
    w, h = c.image_size
    px = overlay.project_floor(c, np.array([[0, 3.0], [40, 1.0], [-60, 2.0]]))
    keep = np.isfinite(px).all(axis=1)
    assert keep[0]
    # anything kept is within the image-corner radius guard when undistorted back
    und = c.undistort_points(px[keep])
    n = (und - c.K[:2, 2]) / c.K[0, 0]
    assert np.hypot(n[:, 0], n[:, 1]).max() <= 1.06 * overlay._corner_radius(c)


def test_detectable_distance_matches_direct_projection_and_scales_with_flame():
    c = cam(tilt=15.0, h=1.0)
    d3 = overlay.detectable_distance(c, 0.3, 25, 640, 0.0)
    assert d3 is not None
    r = 640 / 1920

    def px_at(d, hf):
        p0 = c.floor_to_pixel([[0, d]], 0.0)[0]
        p1 = c.floor_to_pixel([[0, d]], hf)[0]
        return r * abs(p1[1] - p0[1])

    assert px_at(d3, 0.3) >= 25 - 1e-6
    assert px_at(d3 + 0.05, 0.3) < 25
    d6 = overlay.detectable_distance(c, 0.6, 25, 640, 0.0)
    assert 1.6 < d6 / d3 < 2.4  # roughly linear in flame height
    # contact height raises the base and changes the answer
    assert overlay.detectable_distance(c, 0.3, 25, 640, 0.1) != pytest.approx(d3, abs=1e-3)


def test_detectable_distance_is_resolution_invariant():
    c = cam()
    d1 = overlay.detectable_distance(c, 0.3)
    d2 = overlay.detectable_distance(c.with_resolution(640, 360), 0.3)
    assert d1 == pytest.approx(d2, rel=0.03)
    assert overlay.detectable_distance(c, 0.0001) is None


def test_draw_and_map_smoke():
    c = cam(dist=DIST)
    img = np.zeros((1080, 1920, 3), np.uint8)
    out = overlay.draw_rings(img.copy(), c, band_m=2.0)
    assert out.any()
    ov = overlay.RingOverlay(c, 1920, 1080, band_m=2.0)
    cached = ov.apply(img.copy())
    assert np.array_equal(cached, out)
    assert out.any()
    small = overlay.draw_rings(np.zeros((360, 640, 3), np.uint8), c)  # auto-rescales camera
    assert small.any()
    m = overlay.topdown_map(c, band_m=2.0, points_xy=[[0.0, 3.0]])
    assert m.ndim == 3 and m.any()
    mask = overlay.floor_mask(c, 3.0)
    assert mask.shape == (1080, 1920) and mask[-1, 960] and not mask[0, 960]


def test_labels_do_not_collide():
    c = cam(tilt=5.0, h=0.5)
    ov = overlay.RingOverlay(c, 1920, 1080, radii=(1, 2, 3, 5, 10, 20))
    ys = sorted(y for _, y in ov.label_pos)
    assert len(ys) >= 1
    assert all(b - a >= 18 for a, b in zip(ys, ys[1:]))


def test_per_pixel_band_matches_straight_ahead_and_is_close_to_circle():
    c = cam(tilt=15.0, h=1.0)
    m = overlay.detectability_mask(c, 0.3, 25, 640, 0.0)
    d = overlay.detectable_distance(c, 0.3, 25, 640, 0.0)
    v_in = c.floor_to_pixel([[0, d - 0.1]])[0]
    v_out = c.floor_to_pixel([[0, d + 0.1]])[0]
    assert m[int(v_in[1]), int(v_in[0])] and not m[int(v_out[1]), int(v_out[0])]
    # pinhole size scales with depth along the optical axis, so off-axis points at the same radial
    # distance look slightly *larger*: the per-pixel band is a bit bigger than the circle, not smaller
    assert 0.9 < m.sum() / overlay.floor_mask(c, d).sum() < 1.15
    with pytest.raises(ValueError):
        overlay.RingOverlay(c, 1920, 1080).apply(np.zeros((360, 640, 3), np.uint8))
