import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geometry import Camera, fit_homography  # noqa: E402

DIST = [-0.08, 0.02, 0.001, -0.0005, 0.0]


def cam(tilt=30.0, roll=0.0, h=1.5, dist=None, size=(1920, 1080), hfov=70.0):
    return Camera.from_dict({"image_size": list(size), "hfov_deg": hfov, "dist": dist,
                             "pose": {"height_m": h, "tilt_deg": tilt, "roll_deg": roll}})


def test_optical_axis_hits_floor_at_h_over_tan_tilt():
    c = cam(tilt=30, h=1.5)
    w, h = c.image_size
    xy = c.pixel_to_floor([[w / 2, h / 2]])[0]
    assert xy[0] == pytest.approx(0, abs=1e-9)
    assert xy[1] == pytest.approx(1.5 / math.tan(math.radians(30)), rel=1e-9)


@pytest.mark.parametrize("dist", [None, DIST])
@pytest.mark.parametrize("roll", [0.0, 5.0, -7.0])
def test_round_trip_floor_pixel_floor(dist, roll):
    c = cam(roll=roll, dist=dist)
    pts = np.array([[0, 2.5], [1, 3], [-1.5, 4], [0.3, 8], [2, 2.2]])
    back = c.pixel_to_floor(c.floor_to_pixel(pts))
    assert np.allclose(back, pts, atol=1e-5)


def test_known_distance_and_bearing_zero_tilt_geometry():
    c = cam(tilt=45, h=2.0)
    # point straight ahead at 2 m (depression angle 45 deg) is at image centre
    px = c.floor_to_pixel([[0, 2.0]])[0]
    assert px == pytest.approx([960, 540], abs=1e-6)
    d = c.box_to_floor([px[0] - 10, px[1] - 40, px[0] + 10, px[1]])
    assert d["distance_m"] == pytest.approx(2.0, rel=1e-6)
    assert d["bearing_deg"] == pytest.approx(0, abs=1e-6)
    px = c.floor_to_pixel([[2.0, 2.0]])[0]
    d = c.box_to_floor([px[0] - 5, px[1] - 5, px[0] + 5, px[1]])
    assert d["distance_m"] == pytest.approx(math.hypot(2, 2), rel=1e-6)
    assert d["bearing_deg"] == pytest.approx(45, abs=1e-6)
    assert d["x_m"] > 0  # right of centre is +x


def test_rescale_invariance():
    c = cam(dist=DIST)
    c2 = c.with_resolution(640, 360)
    pts = np.array([[0, 3], [1, 5], [-2, 4]])
    p1 = c.floor_to_pixel(pts)
    p2 = c2.floor_to_pixel(pts)
    assert np.allclose(p2, p1 * [640 / 1920, 360 / 1080], atol=1e-6)
    assert np.allclose(c2.pixel_to_floor(p2), pts, atol=1e-5)
    # round trip through json at new resolution too
    c3 = Camera.from_dict(c2.to_dict()).with_resolution(1280, 720)
    assert np.allclose(c3.pixel_to_floor(c3.floor_to_pixel(pts)), pts, atol=1e-5)


def test_horizon_returns_nan_not_garbage():
    c = cam(tilt=10, h=1.5)
    above = c.pixel_to_floor([[960, 0], [960, 100], [10, 5]])
    assert np.all(np.isnan(above))
    # exactly on horizon: either NaN or beyond max_range -> NaN
    a, b, cc = c.horizon_line()
    v = -(a * 960 + cc) / b
    assert np.isnan(c.pixel_to_floor([[960, v]])).all()
    below = c.pixel_to_floor([[960, 1000]])
    assert np.all(np.isfinite(below)) and below[0, 1] > 0
    d = c.box_to_floor([900, 0, 1000, 50])
    assert d["distance_m"] is None and d["bearing_deg"] is None
    # behind the camera has no pixel
    assert np.isnan(c.floor_to_pixel([[0, -1.0]])).all()


def test_max_range_clips_far_points():
    c = cam(tilt=5, h=1.5)
    c.max_range_m = 20
    a, b, cc = c.horizon_line()
    v = -(a * 960 + cc) / b + 1  # just under horizon -> very far
    assert np.isnan(c.pixel_to_floor([[960, v]])).all()


def test_homography_mode_matches_pose_and_roundtrips_json(tmp_path):
    c = cam(tilt=25, roll=3, dist=DIST)
    fl = np.array([[-1, 2], [1, 2], [-1, 5], [1, 5], [0, 3.5]])
    px = c.undistort_points(c.floor_to_pixel(fl))
    H = fit_homography(px, fl)
    hc = Camera((1920, 1080), c.K, c.dist, homography=H)
    p = tmp_path / "camera.json"
    hc.save(p)
    hc2 = Camera.load(p)
    test = np.array([[0.5, 3.0], [-0.7, 4.4]])
    assert np.allclose(hc2.pixel_to_floor(c.floor_to_pixel(test)), test, atol=1e-5)
    assert np.isnan(hc2.pixel_to_floor([[960, -600]])).all()  # above horizon
    hc3 = hc2.with_resolution(960, 540)
    assert np.allclose(hc3.pixel_to_floor(c.floor_to_pixel(test) / 2), test, atol=1e-5)


def test_sign_normalised_homography_flips_negative_scale():
    c = cam()
    fl = np.array([[-1, 2], [1, 2], [-1, 5], [1, 5.0]])
    px = c.floor_to_pixel(fl)
    H = fit_homography(px, fl)
    assert (np.c_[fl, np.ones(4)] @ H.T)[:, 2].min() > 0


def _shifted_frame_camera(c, theta_deg=33.0, shift=(2.0, -1.0)):
    th = math.radians(theta_deg)
    Rz = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    fl = np.array([[-1, 2], [1, 2], [-1, 5], [1, 5], [0, 3.5], [0.5, 7]])
    fl2 = fl @ Rz.T + np.array(shift)
    px = c.undistort_points(c.floor_to_pixel(fl))
    return Camera(c.image_size, c.K, c.dist, homography=fit_homography(px, fl2)), shift


def test_homography_origin_recovered_distance_bearing_unchanged():
    c = cam(tilt=25, roll=3, h=1.5, dist=DIST)
    hc, shift = _shifted_frame_camera(c)
    assert hc.height_m == pytest.approx(1.5, abs=1e-6)
    assert hc.tilt_deg == pytest.approx(25, abs=1e-4)  # iterative undistort limits precision
    assert hc.camera_xy_m == pytest.approx(shift, abs=1e-6)
    assert abs(hc.heading_deg) > 1  # frame was rotated, heading reflects that
    for xy in ([0.5, 3.0], [-0.7, 4.4], [1.2, 6.0]):
        px = c.floor_to_pixel([xy])[0]
        box = [px[0] - 4, px[1] - 20, px[0] + 4, px[1]]
        a, b = c.box_to_floor(box), hc.box_to_floor(box)
        assert b["distance_m"] == pytest.approx(a["distance_m"], abs=1e-5)
        assert b["bearing_deg"] == pytest.approx(a["bearing_deg"], abs=1e-4)


def test_contact_height_plane_intersection():
    c = cam(tilt=25, roll=2, h=1.5, dist=DIST)
    hc, _ = _shifted_frame_camera(c)
    pt = np.array([0.4, 3.0])
    px = c.floor_to_pixel([pt], height_m=0.1)[0]
    assert c.pixel_to_floor([px], 0.1)[0] == pytest.approx(pt, abs=1e-6)
    low = cam(tilt=25, roll=2, h=1.4, dist=DIST)  # pose lowered by the contact height
    assert low.pixel_to_floor([px], 0.0)[0] == pytest.approx(pt, abs=1e-6)
    box = [px[0] - 3, px[1] - 12, px[0] + 3, px[1]]
    a, b = c.box_to_floor(box, 0.1), hc.box_to_floor(box, 0.1)
    assert a["distance_m"] == pytest.approx(math.hypot(*pt), abs=1e-6)
    assert b["distance_m"] == pytest.approx(a["distance_m"], abs=1e-5)
    assert a["contact_height_m"] == 0.1
    assert c.box_to_floor(box, 0.0)["distance_m"] != pytest.approx(a["distance_m"], abs=1e-3)
    assert np.isnan(c.pixel_to_floor([px], 1.5)).all()  # plane at camera height
    c.contact_height_m = 0.1
    assert Camera.from_dict(c.to_dict()).contact_height_m == 0.1
    assert c.box_to_floor(box)["distance_m"] == pytest.approx(a["distance_m"])


def test_with_resolution_rejects_crop():
    with pytest.raises(ValueError):
        cam().with_resolution(1280, 960)
    cam().with_resolution(1280, 720)
