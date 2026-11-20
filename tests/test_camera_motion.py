"""Camera movement smaller than the 60% rule: shift, rotation, vibration, settling (src/obstruction/motion.py)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
import cv2
import numpy as np
from obstruction import ReferenceDetector, SceneMonitor
from test_obstruction_zones import config


def structured(seed=0):
    """320x240 scene of random rectangles: corners for the feature matcher (a real scene has them too)."""
    rng = np.random.default_rng(seed)
    image = np.full((240, 320), 110, np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, 300), rng.integers(0, 220)
        cv2.rectangle(image, (int(x), int(y)), (int(x + rng.integers(8, 40)), int(y + rng.integers(8, 40))),
                      int(rng.integers(30, 230)), -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def moved(image, dx=0., dy=0., angle=0.):
    matrix = cv2.getRotationMatrix2D((160, 120), angle, 1)
    matrix[:, 2] += [dx, dy]
    return cv2.warpAffine(image, matrix, (320, 240), borderMode=cv2.BORDER_REFLECT)


def run(frames, fps=10):
    monitor = SceneMonitor(config(), ReferenceDetector(learn_s=1.))
    return [monitor.update(f, i / fps) for i, f in enumerate(frames)]


class CameraMotionTests(unittest.TestCase):
    def test_small_shift_is_view_changed_not_obstructed(self):
        bg = structured()
        outs = run([bg] * 20 + [moved(bg, dx=3)] * 60)
        self.assertTrue(all(o['view_changed'] for o in outs[23:]))
        self.assertFalse(any(o['zones']['left']['active'] for o in outs))
        self.assertAlmostEqual(outs[-1]['camera_motion']['shift_px'], 3, delta=.3)
        self.assertLess(outs[-1]['foreground_pct'], 60)            # the old 60% rule alone would have missed it

    def test_small_rotation_is_view_changed(self):
        bg = structured()
        outs = run([bg] * 20 + [moved(bg, angle=1.)] * 30)
        self.assertTrue(outs[-1]['view_changed'])
        self.assertAlmostEqual(abs(outs[-1]['camera_motion']['angle_deg']), 1., delta=.2)

    def test_vibration_then_settling_back(self):
        bg = structured()
        shake = [moved(bg, dx=3 if i % 2 else -3, dy=2 if i % 3 else -2) for i in range(30)]
        outs = run([bg] * 20 + shake + [bg] * 40)
        self.assertTrue(any(o['view_changed'] for o in outs[20:50]))
        self.assertFalse(any(o['view_changed'] for o in outs[55:]))  # back in place: checks resume
        self.assertFalse(any(o['zones']['left']['active'] for o in outs))

    def test_object_arriving_is_not_camera_movement(self):
        bg = structured()
        box = bg.copy()
        box[20:220, 10:120] = 20
        outs = run([bg] * 20 + [box] * 60)
        self.assertFalse(any(o['view_changed'] for o in outs))
        self.assertTrue(outs[-1]['zones']['left']['active'])
        self.assertLess(outs[-1]['camera_motion']['shift_px'], .5)

    def test_movement_below_the_limit_is_measured_not_flagged(self):
        bg = structured()
        outs = run([bg] * 20 + [moved(bg, dx=.5)] * 20)
        self.assertTrue(outs[-1]['camera_motion']['known'])
        self.assertFalse(outs[-1]['view_changed'])

    def test_featureless_scene_reports_movement_unknown(self):
        bg = np.full((240, 320, 3), 120, np.uint8)
        outs = run([bg] * 30)
        self.assertFalse(outs[-1]['camera_motion']['known'])          # never "not moved" by assumption


if __name__ == '__main__':
    unittest.main()
