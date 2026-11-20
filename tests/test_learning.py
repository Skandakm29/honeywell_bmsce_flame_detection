"""Learning the empty scene: ghost correction, calibration failure, operator confirmation. Synthetic frames only."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
import numpy as np
from obstruction import ReferenceDetector, SceneMonitor
from test_obstruction_zones import config, scene, with_box


def run(frames, fps=10, monitor=None):
    monitor = monitor or SceneMonitor(config(), ReferenceDetector(learn_s=1.))
    return monitor, [monitor.update(f, i / fps) for i, f in enumerate(frames)]


class GhostTests(unittest.TestCase):
    def test_object_present_while_learning_that_leaves_is_corrected(self):
        bg = scene()
        worker = with_box(bg, 10, 20, 120, 220)                      # in the zone while the scene is learned
        frames = [worker] * 20 + [bg] * 60                           # leaves at 2 s
        monitor, outs = run(frames)
        self.assertGreater(outs[21]['zones']['left']['blocked_pct'], 50)   # the ghost is seen at first
        self.assertFalse(any(o['zones']['left']['active'] for o in outs))  # corrected before the 3 s persistence
        self.assertEqual(outs[-1]['zones']['left']['blocked_pct'], 0)
        self.assertEqual(len(monitor.detector.ghost_events), 1)

    def test_without_correction_the_ghost_stays_blocked(self):
        bg = scene()
        frames = [with_box(bg, 10, 20, 120, 220)] * 20 + [bg] * 60
        _, outs = run(frames, monitor=SceneMonitor(config(), ReferenceDetector(learn_s=1., ghost_s=1e9)))
        self.assertTrue(outs[-1]['zones']['left']['active'])          # the old behaviour (webcam test A: 21%)

    def test_real_dark_object_arriving_is_never_corrected(self):
        bg = scene()
        frames = [bg] * 20 + [with_box(bg, 10, 20, 120, 220)] * 300  # 30 s
        monitor, outs = run(frames)
        self.assertTrue(outs[-1]['zones']['left']['active'])
        self.assertEqual(monitor.detector.ghost_events, [])

    def test_plain_bright_sheet_over_texture_is_never_corrected(self):
        bg = scene()
        frames = [bg] * 20 + [with_box(bg, 10, 20, 120, 220, value=200)] * 300   # smooth, unlike the background
        monitor, outs = run(frames)
        self.assertTrue(outs[-1]['zones']['left']['active'])
        self.assertEqual(monitor.detector.ghost_events, [])


class CalibrationTests(unittest.TestCase):
    def test_scene_that_never_stills_is_reported_as_failed(self):
        a, b = scene(1), with_box(scene(1), 100, 50, 300, 200)
        frames = [a if i % 2 else b for i in range(400)]             # changes every frame for 40 s
        _, outs = run(frames)
        self.assertTrue(all(o['state'] == 'learning' for o in outs))
        self.assertFalse(outs[200]['calibration_failed'])            # 20 s: still trying
        self.assertTrue(outs[-1]['calibration_failed'])              # past 30 s: flagged, not silent

    def test_nothing_is_armed_until_the_operator_confirms(self):
        bg = scene()
        box = with_box(bg, 10, 20, 120, 220)
        monitor = SceneMonitor(config(), ReferenceDetector(learn_s=1.), require_confirmation=True)
        _, outs = run([bg] * 20 + [box] * 50, monitor=monitor)
        self.assertEqual(outs[-1]['state'], 'awaiting_confirmation')
        self.assertFalse(any(o['zones']['left']['active'] for o in outs))
        self.assertEqual(monitor.reference_image().shape, (240, 320))
        monitor.confirm()
        outs = [monitor.update(box, 7 + i / 10) for i in range(40)]
        self.assertTrue(outs[-1]['zones']['left']['active'])

    def test_relearn_asks_for_confirmation_again(self):
        monitor = SceneMonitor(config(), ReferenceDetector(learn_s=1.), require_confirmation=True)
        run([scene()] * 20, monitor=monitor)
        monitor.confirm()
        monitor.relearn(2.)
        outs = [monitor.update(scene(), 2.1 + i / 10) for i in range(20)]
        self.assertEqual(outs[-1]['state'], 'awaiting_confirmation')


if __name__ == '__main__':
    unittest.main()
