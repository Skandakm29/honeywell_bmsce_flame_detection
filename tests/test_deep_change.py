"""DeepChangeDetector plumbing (src/obstruction/deep.py) with a small stand-in feature extractor, so the tests need
neither the DINOv2 ONNX file nor its speed. The stand-in, like DINOv2, describes local structure and ignores a
global brightness gain."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
import cv2
import numpy as np
from obstruction import ReferenceDetector, SceneMonitor
from obstruction.deep import DeepChangeDetector, neighbour_similarity
from test_obstruction_zones import config, scene, with_box

FIRE_BOX = np.array([[180, 100, 220, 150, .8]])


class PatchStructure:
    """Per 20 px patch: its 5x5 grey thumbnail, mean removed, unit length -> [12, 16, 25] for a 240x320 frame."""
    calls = 0

    def __call__(self, frame):
        PatchStructure.calls += 1
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        small = cv2.resize(grey, (80, 60), interpolation=cv2.INTER_AREA)
        f = small.reshape(12, 5, 16, 5).transpose(0, 2, 1, 3).reshape(12, 16, 25)
        f = f - f.mean(-1, keepdims=True) + 1e-3
        return f / np.linalg.norm(f, axis=-1, keepdims=True)


def detector(**kw):
    kw.setdefault('min_threshold', .2)
    return DeepChangeDetector(extractor=PatchStructure(), pixel=ReferenceDetector(learn_s=1.), **kw)


def run(frames, det, boxes=None, fps=10):
    monitor = SceneMonitor(config(), det)
    boxes = boxes or [None] * len(frames)
    return monitor, [monitor.update(f, i / fps, exclude_boxes=b) for i, (f, b) in enumerate(zip(frames, boxes))]


class DeepChangeTests(unittest.TestCase):
    def test_similarity_is_one_for_identical_features_and_tolerates_one_patch_shift(self):
        f = PatchStructure()(scene())
        self.assertTrue(np.allclose(neighbour_similarity(f, f[None]), 1, atol=1e-5))
        shifted = np.roll(f, 1, axis=1)
        inner = neighbour_similarity(shifted, f[None])[:, 1:-1]
        self.assertTrue(np.allclose(inner, 1, atol=1e-5))

    def test_object_in_zone_alarms_like_the_pixel_method(self):
        bg = scene()
        frames = [bg] * 20 + [with_box(bg, 10, 20, 120, 220)] * 60
        for method in ('deep', 'fused'):
            _, outs = run(frames, detector(method=method))
            first = next(i for i, o in enumerate(outs) if o['zones']['left']['active'])
            self.assertAlmostEqual(first / 10, 2 + 3, delta=.25, msg=method)      # persist_s = 3

    def test_threshold_comes_from_learning_data_with_a_floor(self):
        det = detector()
        run([scene()] * 15, det)
        self.assertIsNotNone(det.threshold)
        self.assertGreaterEqual(det.threshold, .2)
        self.assertGreaterEqual(det.threshold, det.factor * det.normal_q995 - 1e-9)

    def test_global_brightness_change_is_not_an_obstruction(self):
        bg = scene()
        darker = (bg.astype(np.float32) * .6).astype(np.uint8)
        _, outs = run([bg] * 20 + [darker] * 40, detector())
        self.assertFalse(any(o['zones']['left']['active'] for o in outs))

    def test_flame_box_is_excluded(self):
        bg = scene()
        fire = with_box(bg, 180, 100, 220, 150, value=250)
        _, outs = run([bg] * 20 + [fire] * 30, detector(), [None] * 20 + [FIRE_BOX] * 30)
        self.assertEqual(max(o['frame_pct'] for o in outs), 0)

    def test_every_s_limits_feature_maps(self):
        PatchStructure.calls = 0
        run([scene()] * 60, detector(every_s=.5))                              # 6 s at 10 fps
        self.assertLess(PatchStructure.calls, 6 + 5 * 2 + 3)                     # learning samples + 2 per second

    def test_background_worker_never_blocks_and_eventually_detects(self):
        bg, box = scene(), with_box(scene(), 10, 20, 120, 220)
        det = detector(background=True)
        monitor = SceneMonitor(config(), det)
        t = 0.
        for _ in range(20):
            monitor.update(bg, t)
            t += .1
        import time
        out = None
        for _ in range(80):
            out = monitor.update(box, t)
            t += .1
            time.sleep(.005)                                                     # give the worker thread time
        self.assertGreater(out['zones']['left']['blocked_pct'], 50)


if __name__ == '__main__':
    unittest.main()
