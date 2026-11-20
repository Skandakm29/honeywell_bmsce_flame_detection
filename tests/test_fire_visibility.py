"""Previously detected fire - visibility compromised (built on the Week 2 alarm), and flame present while learning.
Synthetic frames only."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
import numpy as np
from flame_alarm import FlameAlarm
from obstruction import FireVisibility, ReferenceDetector, SceneMonitor
from test_obstruction_zones import config, scene, with_box

FIRE = (180, 100, 220, 150)                       # flame box, frame pixels (320x240 scene)
FIRE_BOX = np.array([[*FIRE, .8]])
NONE = np.zeros((0, 5))
SHAPE = (240, 320, 3)


def fire(image):
    return with_box(image, *FIRE, value=250)


def run(frames, boxes, fps=10):
    """Zone monitor fed with the Week 2 alarm output, exactly as live_flame.py does."""
    monitor, week2 = SceneMonitor(config(), ReferenceDetector(learn_s=1.)), FlameAlarm(frame_dt=1 / fps)
    outs = []
    for i, (f, b) in enumerate(zip(frames, boxes)):
        flame = week2.update(b, i / fps, f.shape)
        outs.append(monitor.update(f, i / fps, exclude_boxes=b, flame=flame))
    return monitor, outs


def compromised(outs):
    return [o['visibility']['visibility_compromised'] for o in outs]


class FireVisibilityTests(unittest.TestCase):
    def test_fire_covered_by_object_is_flagged_until_it_is_seen_again(self):
        bg = scene()
        blocker = with_box(bg, 140, 60, 280, 200, value=30)          # covers the fire and around it
        frames = [bg] * 20 + [fire(bg)] * 20 + [blocker] * 40 + [fire(bg)] * 10
        boxes = [NONE] * 20 + [FIRE_BOX] * 20 + [NONE] * 40 + [FIRE_BOX] * 10
        _, outs = run(frames, boxes)
        flags = compromised(outs)
        self.assertFalse(any(flags[:40]))
        first = flags.index(True)
        self.assertLessEqual(first / 10, 4 + .6)                     # covered from 4 s; flagged after lost_s
        self.assertTrue(all(flags[first:80]))
        self.assertFalse(flags[-1])                                  # fire visible again
        # Covered for 4 s: Week 2 dropped the track after 1 s, so the fire comes back as a new Week 2 track nearby.
        self.assertEqual([e['reason'] for o in outs for e in o['visibility']['events']], ['fire visible nearby'])

    def test_long_cover_stays_flagged_for_two_minutes(self):
        bg = scene()
        blocker = with_box(bg, 140, 60, 280, 200, value=30)
        frames = [bg] * 20 + [fire(bg)] * 20 + [blocker] * 1200 + [bg] * 40   # 120 s covered, then gone (fire out)
        boxes = [NONE] * 20 + [FIRE_BOX] * 20 + [NONE] * 1240
        _, outs = run(frames, boxes)
        flags = compromised(outs)
        first = flags.index(True)
        self.assertTrue(all(flags[first:1240]))                      # Week 2 dropped the track after 1 s; still flagged
        self.assertFalse(flags[-1])                                  # area clear for 2 s: forgotten
        events = [e for o in outs for e in o['visibility']['events']]
        self.assertEqual([e['reason'] for e in events], ['area clear'])
        self.assertGreater(events[0]['end_s'] - events[0]['start_s'], 119)
        self.assertAlmostEqual(events[0]['last_seen_s'], 3.9, delta=.05)

    def test_only_fires_confirmed_by_week2_are_followed(self):
        bg = scene()
        blocker = with_box(bg, 140, 60, 280, 200, value=30)
        frames = [bg] * 20 + [bg] * 3 + [blocker] * 40
        boxes = [NONE] * 20 + [FIRE_BOX] * 3 + [NONE] * 40           # 3 of 20 frames < 30%: Week 2 never confirms
        _, outs = run(frames, boxes)
        self.assertFalse(any(compromised(outs)))
        self.assertEqual(outs[-1]['visibility']['fires'], [])

    def test_fire_going_out_in_clear_view_is_not_flagged(self):
        bg = scene()
        frames = [bg] * 20 + [fire(bg)] * 20 + [bg] * 40
        boxes = [NONE] * 20 + [FIRE_BOX] * 20 + [NONE] * 40
        _, outs = run(frames, boxes)
        self.assertFalse(any(compromised(outs)))
        self.assertEqual(outs[-1]['visibility']['fires'], [])

    def test_fire_area_cover_is_reported_while_the_fire_is_still_visible(self):
        bg = scene()
        side = with_box(fire(bg), 140, 60, 178, 200, value=30)       # object right beside the fire, fire visible
        frames = [bg] * 20 + [fire(bg)] * 20 + [side] * 20
        boxes = [NONE] * 20 + [FIRE_BOX] * 40
        _, outs = run(frames, boxes)
        self.assertFalse(any(compromised(outs)))                      # fire still seen
        self.assertEqual(outs[39]['visibility']['fire_area_covered_pct'], 0)
        self.assertGreater(outs[-1]['visibility']['fire_area_covered_pct'], 10)

    def test_view_change_is_not_called_compromised(self):
        bg = scene()
        ramp = np.tile(np.linspace(0, 255, 320, dtype=np.float32), (240, 1)).astype(np.uint8)
        other = np.dstack([ramp] * 3)                                 # a different view (camera moved, scene cut)
        frames = [bg] * 20 + [fire(bg)] * 20 + [other] * 30
        boxes = [NONE] * 20 + [FIRE_BOX] * 20 + [NONE] * 30
        _, outs = run(frames, boxes)
        self.assertTrue(any(o['view_changed'] for o in outs[40:]))
        self.assertFalse(any(compromised(outs)))

    def test_small_flame_vanishing_next_to_a_visible_fire_is_not_flagged(self):
        week2, vis = FlameAlarm(frame_dt=.1), FireVisibility()
        mask = np.ones((240, 320), bool)                             # area around it "changed" (smoke)
        lick = np.array([[185, 62, 200, 85, .5]])                    # small flame above the fire: own Week 2 track
        sequence = [np.vstack([FIRE_BOX, lick])] * 10 + [FIRE_BOX] * 30
        outs = [vis.update(week2.update(b, i / 10, SHAPE), i / 10, mask, None, SHAPE) for i, b in enumerate(sequence)]
        self.assertEqual(len({f['id'] for o in outs[:10] for f in o['fires']}), 2)   # both were confirmed fires
        self.assertFalse(any(o['visibility_compromised'] for o in outs))

    def test_never_touches_flame_boxes(self):
        week2, vis, boxes = FlameAlarm(frame_dt=.1), FireVisibility(), FIRE_BOX.copy()
        for i in range(10):
            vis.update(week2.update(boxes, i / 10, SHAPE), i / 10, None, None, SHAPE)
        np.testing.assert_array_equal(boxes, FIRE_BOX)


class StartUpFlameTests(unittest.TestCase):
    def test_flame_while_learning_is_unknown_not_normal(self):
        bg = scene()
        frames = [fire(bg)] * 20 + [bg] * 30                       # fire from frame 1, goes out at 2 s
        boxes = [FIRE_BOX] * 20 + [NONE] * 30
        _, outs = run(frames, boxes)
        self.assertGreater(outs[12]['unknown_pct'], 0)               # learned at 1 s with the fire area unknown
        self.assertTrue(all(o['frame_pct'] == 0 for o in outs[20:]))  # without the fix: ~14% of the frame
        self.assertFalse(any(compromised(outs)))

    def test_unknown_area_is_filled_after_it_stays_clear(self):
        bg = scene()
        frames = [fire(bg)] * 20 + [bg] * 100                        # 10 s clear after the fire
        boxes = [FIRE_BOX] * 20 + [NONE] * 100
        monitor, outs = run(frames, boxes)
        self.assertIsNone(monitor.detector.unknown)
        self.assertEqual(outs[-1]['unknown_pct'], 0.)
        out = monitor.update(with_box(bg, 140, 60, 280, 200, value=30), 12.1, exclude_boxes=NONE)
        self.assertGreater(out['frame_pct'], 5)

    def test_object_in_front_of_start_up_fire_is_counted(self):
        bg = scene()
        blocker = with_box(bg, 140, 60, 280, 200, value=30)
        frames = [fire(bg)] * 20 + [blocker] * 30
        boxes = [FIRE_BOX] * 20 + [NONE] * 30
        _, outs = run(frames, boxes)
        fire_cells = (220 - 180) * (150 - 100) / (320 * 240) * 100
        self.assertGreater(outs[-1]['frame_pct'], 20 + fire_cells)
        self.assertTrue(outs[-1]['visibility']['visibility_compromised'])

    def test_object_in_front_of_out_fire_is_not_learned(self):
        bg = scene()
        blocker = with_box(bg, 140, 60, 280, 200, value=30)
        frames = [fire(bg)] * 20 + [blocker] * 150
        boxes = [FIRE_BOX] * 20 + [NONE] * 150
        monitor, outs = run(frames, boxes)
        self.assertIsNotNone(monitor.detector.unknown)
        self.assertGreater(outs[-1]['frame_pct'], 5)


if __name__ == '__main__':
    unittest.main()
