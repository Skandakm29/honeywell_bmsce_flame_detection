import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2
import numpy as np
from obstruction import (CellTimers, ReferenceDetector, SceneMonitor, ZoneAlarm, ZoneGrid, cell_coverage,
                         parse_zones, rasterise)

LEFT_HALF = [[0, 0], [.5, 0], [.5, 1], [0, 1]]


def config(**zone):
    return parse_zones({'grid': {'cols': 32, 'rows': 24},
                        'zones': [dict({'name': 'left', 'polygon': LEFT_HALF, 'min_blocked_pct': 20,
                                        'persist_s': 3, 'clear_s': 2}, **zone)]})


def scene(seed=0):
    """Textured 240x320 background, so edges and grey levels both carry information."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.uniform(40, 200, (240, 320)).astype(np.float32), (0, 0), 3)
    return cv2.cvtColor(noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def with_box(image, x1, y1, x2, y2, value=20):
    out = image.copy()
    out[y1:y2, x1:x2] = value
    return out


class ZoneConfigTests(unittest.TestCase):
    def test_valid_and_invalid_configs(self):
        self.assertEqual(config().zones[0].name, 'left')
        bad = [{'zones': []},
               {'zones': [{'name': 'a', 'polygon': [[0, 0], [2, 0], [0, 1]]}]},
               {'zones': [{'name': 'a', 'polygon': [[0, 0], [1, 0]]}]},
               {'zones': [{'name': 'a', 'polygon': [[0, 0], [.5, .5], [1, 1]]}]},  # zero area
               {'zones': [{'name': 'a', 'polygon': LEFT_HALF}, {'name': 'a', 'polygon': LEFT_HALF}]},
               {'zones': [{'name': 'a', 'polygon': LEFT_HALF, 'min_blocked_pct': 0}]}]
        for data in bad:
            with self.assertRaises(ValueError, msg=data):
                parse_zones(data)

    def test_rasterise_scales_normalised_polygon(self):
        self.assertAlmostEqual(rasterise(LEFT_HALF, 320, 240).mean(), .5, delta=.01)


    def test_drawn_pixel_polygons_become_a_valid_normalised_config(self):
        from draw_zones import build_config
        data = build_config([[(0, 0), (320, 0), (320, 480), (0, 480)]], [[(600, 0), (640, 0), (640, 60)]],
                            640, 480, 25, 8, 4)
        cfg = parse_zones(data)
        self.assertEqual(cfg.zones[0].polygon[1], (.5, 0.))
        self.assertEqual((cfg.zones[0].persist_s, len(cfg.ignore)), (8., 1))
        with self.assertRaises(ValueError):
            build_config([[(0, 0), (10, 10), (20, 20)]], [], 640, 480, 25, 8, 4)   # zero-area zone


class GridTests(unittest.TestCase):
    def test_cell_coverage_exact_for_uneven_sizes(self):
        mask = np.zeros((7, 10), bool)
        mask[:, :5] = True
        coverage = cell_coverage(mask, 2, 1)
        np.testing.assert_allclose(coverage, [[1, 0]])
        self.assertAlmostEqual(cell_coverage(np.ones((7, 10), bool), 3, 3).min(), 1)

    def test_overlapping_objects_are_not_double_counted(self):
        grid = ZoneGrid(config(), 320, 240)
        one = np.zeros((240, 320), bool)
        one[:, :80] = True                       # a quarter of the frame = half of the left zone
        both = one.copy()
        both[:, 40:120] = True                   # a second object overlapping the first
        pct_one = grid.percent(grid.blocked_cells(one))['left']
        pct_both = grid.percent(grid.blocked_cells(both))['left']
        self.assertAlmostEqual(pct_one, 50, delta=1)
        self.assertAlmostEqual(pct_both, 75, delta=1)  # union of [0, 120) of a 160-px-wide zone, not 100

    def test_ignore_polygon_never_counts(self):
        cfg = parse_zones({'zones': [{'name': 'all', 'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}],
                           'ignore': [LEFT_HALF]})
        grid = ZoneGrid(cfg, 320, 240)
        blocked = np.zeros((240, 320), bool)
        blocked[:, :160] = True
        self.assertEqual(grid.percent(grid.blocked_cells(blocked))['all'], 0)

    def test_dwell_survives_short_gaps_and_resets_after_long_ones(self):
        timers, on, off = CellTimers((1, 1), gap_s=1.), np.ones((1, 1), bool), np.zeros((1, 1), bool)
        timers.update(on, 0.)
        timers.update(off, .5)                    # missed for 0.5 s: still the same spell
        self.assertEqual(timers.update(on, 2.)[0, 0], 2.)
        timers.update(off, 3.5)                   # free for 1.5 s > gap_s
        self.assertEqual(timers.update(on, 4.)[0, 0], 0.)


class AlarmTests(unittest.TestCase):
    def test_hysteresis_and_event_summary(self):
        alarm = ZoneAlarm('z', 30, clear_s=2)
        self.assertEqual(alarm.update(10, 0), (False, None))
        self.assertTrue(alarm.update(40, 1)[0])
        self.assertTrue(alarm.update(10, 2)[0])   # dip shorter than clear_s keeps the alarm
        self.assertTrue(alarm.update(60, 3)[0])
        alarm.update(0, 4)
        active, event = alarm.update(0, 6)
        self.assertFalse(active)
        self.assertEqual((event['start_s'], event['end_s'], event['peak_pct']), (1, 4, 60))
        self.assertAlmostEqual(event['mean_pct'], (40 + 10 + 60) / 3)  # trailing clear spell left out
        self.assertIsNone(alarm.close(7))


class DetectorTests(unittest.TestCase):
    def test_learning_then_static_scene_is_clear(self):
        detector, background = ReferenceDetector(learn_s=1.), scene()
        self.assertEqual(detector.update(background, 0.)['state'], 'learning')
        for i in range(1, 15):
            result = detector.update(background, i * .1)
        self.assertEqual(result['state'], 'ready')
        self.assertEqual(result['foreground_pct'], 0)

    def test_motion_during_learning_restarts_it(self):
        detector, background = ReferenceDetector(learn_s=1.), scene()
        person = with_box(background, 100, 40, 160, 220)
        for i in range(8):                         # someone walks out of view during the first 0.7 s
            detector.update(with_box(background, 100 + 30 * i, 40, 160 + 30 * i, 220) if i < 7 else background, i / 10)
        self.assertFalse(detector.ready)
        for i in range(8, 20):
            out = detector.update(background, i / 10)
        self.assertTrue(detector.ready)
        self.assertGreater(detector.update(person, 2.1)['mask'][40:220, 100:160].mean(), .9)  # not learned

    def test_global_lighting_change_is_compensated(self):
        detector, background = ReferenceDetector(learn_s=0.), scene()
        detector.update(background, 0.)
        darker = (background.astype(np.float32) * .7).astype(np.uint8)
        self.assertLess(detector.update(darker, .1)['foreground_pct'], 1)

    def test_object_is_foreground_where_it_is(self):
        detector, background = ReferenceDetector(learn_s=0.), scene()
        detector.update(background, 0.)
        result = detector.update(with_box(background, 40, 60, 140, 200), .1)
        self.assertGreater(result['mask'][60:200, 40:140].mean(), .95)
        self.assertLess(result['mask'][:, 200:].mean(), .01)
        self.assertEqual(len(result['blobs']), 1)


class MonitorTests(unittest.TestCase):
    def run_clip(self, frames, fps=10, detector=None):
        monitor = SceneMonitor(config(), detector or ReferenceDetector(learn_s=1.))
        states, events = [], []
        for i, frame in enumerate(frames):
            out = monitor.update(frame, i / fps)
            states.append(out['zones']['left'])
            events += out['events']
        return states, events + monitor.close(len(frames) / fps)

    def test_object_left_in_zone_alarms_after_persist_and_clears(self):
        background = scene()
        box = with_box(background, 10, 20, 120, 220)  # about two thirds of the left zone
        frames = [background] * 20 + [box] * 60 + [background] * 40   # 2 s clear, 6 s object, 4 s clear
        states, events = self.run_clip(frames)
        first = next(i for i, s in enumerate(states) if s['active'])
        self.assertAlmostEqual(first / 10, 2 + 3, delta=.15)          # appears at 2 s, persist_s = 3
        self.assertGreater(max(s['persistent_pct'] for s in states), 50)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]['end_s'], 8, delta=.15)      # removed at 8 s; clear_s counted from then
        self.assertFalse(states[-1]['active'])

    def test_person_walking_across_never_alarms(self):
        background = scene()
        frames = [background] * 20 + [with_box(background, x, 40, x + 40, 220) for x in range(0, 280, 8)]
        states, events = self.run_clip(frames)
        self.assertFalse(any(s['active'] for s in states))
        self.assertEqual(events, [])
        self.assertTrue(any(s['blocked_pct'] > 10 for s in states))   # it was seen, just never persistent

    def test_detection_gap_does_not_reset_persistence(self):
        background = scene()
        box = with_box(background, 10, 20, 120, 220)
        frames = [background] * 20 + ([box] * 9 + [background]) * 6    # drops out one frame in ten
        states, _ = self.run_clip(frames)
        self.assertTrue(states[-1]['active'])

    def test_same_alarm_time_at_5_and_10_fps(self):
        background, box = scene(), with_box(scene(), 10, 20, 120, 220)
        start = {}
        for fps in (5, 10):
            frames = [background] * (2 * fps) + [box] * (6 * fps)
            states, _ = self.run_clip(frames, fps)
            start[fps] = next(i for i, s in enumerate(states) if s['active']) / fps
        self.assertAlmostEqual(start[5], start[10], delta=.21)

    def test_stationary_object_is_not_learned_into_the_reference(self):
        background = scene()
        box = with_box(background, 10, 20, 120, 220)
        detector = ReferenceDetector(learn_s=1., adapt_s=2.)          # adapts fast, to stress the freeze
        states, _ = self.run_clip([background] * 20 + [box] * 300, detector=detector)  # 30 s object
        self.assertTrue(states[-1]['active'])
        self.assertGreater(states[-1]['persistent_pct'], 50)

    def test_relearn_closes_alarm_and_learns_the_new_scene(self):
        background, box = scene(), with_box(scene(), 10, 20, 120, 220)
        monitor = SceneMonitor(config(), ReferenceDetector(learn_s=1.))
        for i in range(80):
            out = monitor.update(background if i < 20 else box, i / 10)
        self.assertTrue(out['zones']['left']['active'])
        events = monitor.relearn(8.)
        self.assertEqual((len(events), events[0]['end_s']), (1, 8.))
        for i in range(81, 130):                   # the box is now part of the learned scene
            out = monitor.update(box, i / 10)
        self.assertEqual(out['state'], 'ready')
        self.assertFalse(out['zones']['left']['active'])
        self.assertEqual(out['zones']['left']['blocked_pct'], 0)

    def test_flame_inside_zone_is_never_an_obstruction(self):
        background = scene()
        flame_box = np.array([[40., 60., 100., 180., .9]])          # frame pixels, like live_flame's boxes
        burning = [with_box(background, 40, 60, 100, 180, value=250 - 40 * (i % 3)) for i in range(80)]
        for exclude, expect_alarm in ((flame_box, False), (None, True)):
            monitor = SceneMonitor(config(min_blocked_pct=10), ReferenceDetector(learn_s=1.))
            for i in range(20):
                monitor.update(background, i / 10)
            for i, frame in enumerate(burning):
                out = monitor.update(frame, 2 + i / 10, exclude_boxes=exclude)
            self.assertEqual(out['zones']['left']['active'], expect_alarm)
            if exclude is not None:
                self.assertEqual(out['zones']['left']['blocked_pct'], 0)
                self.assertGreater(out['excluded_pct'], 0)

    def test_flickering_flame_does_not_stall_learning(self):
        monitor = SceneMonitor(config(), ReferenceDetector(learn_s=1.))
        flame_box = np.array([[40., 60., 100., 180., .9]])
        for i in range(15):
            out = monitor.update(with_box(scene(), 40, 60, 100, 180, value=250 - 60 * (i % 2)), i / 10,
                                 exclude_boxes=flame_box)
        self.assertEqual(out['state'], 'ready')

    def test_blind_map_accumulates_obstruction_time_per_zone(self):
        background = scene()
        box = with_box(background, 10, 20, 120, 220)   # about two thirds of the left zone
        monitor = SceneMonitor(config(), ReferenceDetector(learn_s=1.))
        frames = [background] * 20 + [box] * 80 + [background] * 20   # 1 s learn+clear, 8 s box, 2 s clear
        for i, frame in enumerate(frames):
            monitor.update(frame, i / 10)
        s = monitor.blind_map.summary()
        self.assertAlmostEqual(s['monitored_s'], 11.9 - 1., delta=.15)      # learning time is not counted
        left = s['zones']['left']
        self.assertAlmostEqual(left['longest_obstruction_s'], 7.9, delta=.15)  # 8 s on the floor, minus 1 frame
        self.assertGreater(left['ever_obstructed_pct'], 50)
        expected = left['ever_obstructed_pct'] * (7.9 - 3) / s['monitored_s']   # obstructed after persist_s = 3
        self.assertAlmostEqual(left['mean_obstructed_pct'], expected, delta=2)
        self.assertGreaterEqual(left['mean_blocked_pct'], left['mean_obstructed_pct'])
        self.assertLess(s['frame']['ever_obstructed_pct'], left['ever_obstructed_pct'])

    def test_other_zone_and_floor_area_are_reported_separately(self):
        cfg = parse_zones({'zones': [{'name': 'left', 'polygon': LEFT_HALF},
                                     {'name': 'right', 'polygon': [[.5, 0], [1, 0], [1, 1], [.5, 1]]}]})
        monitor = SceneMonitor(cfg, ReferenceDetector(learn_s=0.))
        background = scene()
        monitor.update(background, 0.)
        out = monitor.update(with_box(background, 10, 20, 120, 220), .1)
        self.assertGreater(out['zones']['left']['blocked_pct'], 50)
        self.assertEqual(out['zones']['right']['blocked_pct'], 0)
        self.assertIsNone(out['zones']['left']['area_m2'])            # no calibration yet: no metres claimed


if __name__ == '__main__':
    unittest.main()
