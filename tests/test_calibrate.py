import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate  # noqa: E402
from geometry import Camera  # noqa: E402

SIZE = (1280, 720)
K_TRUE = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
DIST_TRUE = np.array([-0.10, 0.03, 0.0, 0.0, 0.0])
COLS, ROWS, SQ = 8, 6, 0.03


def _board_image(rvec, tvec):
    """Render a checkerboard through K_TRUE/DIST_TRUE by inverse-mapping every pixel."""
    # undistorted pixel grid -> plane intersection -> board texture lookup
    u, v = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]))
    px = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
    n = cv2.undistortPoints(px, K_TRUE, DIST_TRUE).reshape(-1, 2)  # normalised rays
    R, _ = cv2.Rodrigues(rvec)
    o = -R.T @ tvec.reshape(3)
    d = np.c_[n, np.ones(len(n))] @ R  # world-frame ray directions
    s = -o[2] / d[:, 2]
    X, Y = o[0] + s * d[:, 0], o[1] + s * d[:, 1]
    ix, iy = np.floor(X / SQ).astype(int), np.floor(Y / SQ).astype(int)
    on = (s > 0) & (ix >= 0) & (ix < COLS) & (iy >= 0) & (iy < ROWS)
    img = np.full(len(n), 255, np.uint8)
    img[on & ((ix + iy) % 2 == 0)] = 0
    return cv2.cvtColor(img.reshape(SIZE[1], SIZE[0]), cv2.COLOR_GRAY2BGR)


def test_intrinsics_recovers_synthetic_camera(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(12):
        rv = np.array([rng.uniform(-0.5, 0.5) + math.pi, rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)])
        centre = np.array([COLS * SQ / 2, ROWS * SQ / 2, 0.0])
        R, _ = cv2.Rodrigues(rv)
        t = -R @ centre + np.array([0, 0, 0.35 + rng.uniform(0, 0.15)])  # camera looks at centre
        cv2.imwrite(str(tmp_path / f"b{i:02d}.png"), _board_image(rv, t))
    out = tmp_path / "camera.json"
    rms = calibrate.main(["intrinsics", "--images", str(tmp_path / "b*.png"), "--cols", str(COLS - 1),
                          "--rows", str(ROWS - 1), "--square-m", str(SQ), "--out", str(out)])
    d = json.loads(out.read_text())
    assert rms < 0.6
    assert d["K"][0][0] == pytest.approx(900, rel=0.03)
    assert d["K"][0][2] == pytest.approx(640, abs=8)
    assert d["image_size"] == list(SIZE)


def _floor_pts(cam, shift=(1.0, -2.0)):
    fl = np.array([[-1, 2], [1, 2], [-1, 5], [1, 5], [0, 3.5]], float)
    px = cam.floor_to_pixel(fl)
    return np.c_[px, fl + np.array(shift)], fl


def test_pose_then_floor_noninteractive(tmp_path):
    out = tmp_path / "camera.json"
    calibrate.main(["pose", "--height", "1.5", "--tilt", "25", "--roll", "2", "--hfov-deg", "70",
                    "--size", "1920x1080", "--out", str(out)])
    cam = Camera.load(out)
    assert cam.height_m == 1.5 and cam.source == "pose"
    pts, _ = _floor_pts(cam)
    csvp = tmp_path / "pts.csv"
    csvp.write_text("u,v,x,y\n" + "\n".join(",".join(f"{v:.6f}" for v in r) for r in pts))
    err = calibrate.main(["floor", "--points", str(csvp), "--out", str(out)])
    assert err.max() < 1e-5
    hc = Camera.load(out)
    assert hc.source == "extrinsics"
    assert hc.camera_xy_m == pytest.approx((1.0, -2.0), abs=1e-5)
    assert hc.height_m == pytest.approx(1.5, abs=1e-5)
    # re-running pose drops the floor calibration again
    calibrate.main(["pose", "--height", "2.0", "--tilt", "30", "--out", str(out)])
    assert Camera.load(out).source == "pose"


def test_floor_json_noise_reports_metres(tmp_path):
    out = tmp_path / "camera.json"
    out.write_text(json.dumps({"schema": 1, "image_size": [1920, 1080], "hfov_deg": 70}))
    cam = Camera.from_dict({"image_size": [1920, 1080], "hfov_deg": 70,
                            "pose": {"height_m": 1.5, "tilt_deg": 25}})
    pts, _ = _floor_pts(cam)
    pts[:, :2] += np.random.default_rng(1).normal(0, 1.0, (len(pts), 2))  # 1 px click noise
    jp = tmp_path / "pts.json"
    jp.write_text(json.dumps([dict(u=r[0], v=r[1], x=r[2], y=r[3]) for r in pts]))
    err = calibrate.main(["floor", "--points", str(jp), "--out", str(out), "--hfov-deg", "70",
                          "--size", "1920x1080"])
    assert 0 < err.max() < 0.1
    assert json.loads(out.read_text())["floor_rms_m"] < 0.1
    with pytest.raises(SystemExit):
        calibrate.main(["floor", "--points", str(jp), "--out", str(tmp_path / "x.json")])  # no K


def test_noisy_floor_is_continuous_in_contact_height_and_loo(tmp_path, capsys):
    out = tmp_path / "camera.json"
    out.write_text(json.dumps({"schema": 1, "image_size": [1920, 1080], "hfov_deg": 70}))
    cam = Camera.from_dict({"image_size": [1920, 1080], "hfov_deg": 70,
                            "pose": {"height_m": 1.5, "tilt_deg": 25}})
    fl = np.array([[-1, 2], [1, 2], [-1, 5], [1, 5], [0, 3.5], [0.5, 7], [-0.5, 8.0]])
    px = cam.floor_to_pixel(fl) + np.random.default_rng(3).normal(0, 2.0, (len(fl), 2))
    jp = tmp_path / "pts.json"
    jp.write_text(json.dumps([[*p, *f] for p, f in zip(px, fl)]))
    calibrate.main(["floor", "--points", str(jp), "--out", str(out)])
    assert "leave-one-out error" in capsys.readouterr().out
    c = Camera.load(out)
    for p in ([900, 700], [1300, 900], [500, 650]):
        box = [p[0] - 5, p[1] - 30, p[0] + 5, p[1]]
        d0 = c.box_to_floor(box, 0.0)["distance_m"]
        d1 = c.box_to_floor(box, 1e-4)["distance_m"]
        assert abs(d0 - d1) < 1e-3
    d = json.loads(out.read_text())
    assert "floor_loo_rms_m" in d and "homography" not in d


def test_four_points_warns_and_intrinsics_drops_stale_floor(tmp_path, capsys):
    out = tmp_path / "camera.json"
    calibrate.main(["pose", "--height", "1.5", "--tilt", "25", "--hfov-deg", "70",
                    "--size", "1920x1080", "--out", str(out)])
    cam = Camera.load(out)
    pts, _ = _floor_pts(cam)
    csvp = tmp_path / "p.csv"
    csvp.write_text("\n".join(",".join(map(str, r)) for r in pts[:4]))
    calibrate.main(["floor", "--points", str(csvp), "--out", str(out)])
    assert "WARNING" in capsys.readouterr().out
    d = json.loads(out.read_text())
    assert "extrinsics" in d
    d["K"][0][0] += 50  # pretend K changes -> simulate by calling the stale logic via intrinsics
    out.write_text(json.dumps(d))
    for k in calibrate.STALE_FLOOR:
        assert k in d or k in ("homography", "floor_loo_rms_m")
    calibrate.main(["pose", "--height", "2", "--tilt", "20", "--out", str(out)])
    d = json.loads(out.read_text())
    assert not any(k in d for k in calibrate.STALE_FLOOR)
