import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from flame_verification import Verifier, circular_hue, iou_matrix, match, roi_features


def box(x, y=10, size=20, score=.9):
    return [x, y, x+size, y+size, score]


def frame(value=60, region=None, region_value=None):
    image = np.full((120, 160, 3), value, np.uint8)
    if region is not None:
        x1, y1, x2, y2 = region
        image[y1:y2, x1:x2] = region_value
    return image


def track(reports, track_id):
    return next(r for r in reports if r['track_id'] == track_id)


class TrackingTests(unittest.TestCase):
    def test_iou_matrix(self):
        overlap = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [50, 50, 60, 60]])
        np.testing.assert_allclose(overlap, [[1, 1/3, 0]])

    def test_one_detection_across_consecutive_frames(self):
        v = Verifier(3, 5)
        ids = [v.update([box(0)])[0]['track_id'] for _ in range(5)]
        self.assertEqual(ids, [1]*5)

    def test_empty_frames(self):
        v = Verifier(1, 3)
        self.assertEqual(v.update([]), [])
        self.assertEqual(v.update(np.zeros((0, 5))), [])

    def test_temporary_disappearance_keeps_identity_but_never_alarms(self):
        v = Verifier(2, 4, max_missed=2)
        v.update([box(0)])
        self.assertTrue(v.update([box(0)])[0]['persistent'])
        held = v.update([])[0]
        self.assertEqual(held['track_id'], 1)
        self.assertFalse(held['matched'])
        self.assertFalse(held['persistent'])  # a missed track never alarms on its own
        self.assertFalse(held['fused'])
        returned = v.update([box(1)])[0]
        self.assertEqual(returned['track_id'], 1)  # same object back after one missed frame
        self.assertEqual(returned['stats']['hits'], 3)  # the miss did not count as a hit
        self.assertTrue(returned['persistent'])

    def test_track_expiration_and_ids_not_reused(self):
        v = Verifier(2, 4, max_missed=1)
        v.update([box(0)])
        self.assertEqual(len(v.update([])), 1)
        self.assertEqual(v.update([]), [])  # second consecutive miss exceeds max_missed
        self.assertEqual(v.update([box(0)])[0]['track_id'], 2)

    def test_default_max_missed_is_window_minus_hits(self):
        self.assertEqual(Verifier(4, 6).max_missed, 2)
        self.assertEqual(Verifier(6, 6).max_missed, 1)

    def test_two_simultaneous_objects(self):
        v = Verifier(2, 3)
        for _ in range(3):
            reports = v.update([box(0), box(100)])
        self.assertEqual(sorted(r['track_id'] for r in reports), [1, 2])
        self.assertTrue(all(r['persistent'] for r in reports))

    def test_multiple_detections_cannot_share_a_track(self):
        v = Verifier(1, 3)
        v.update([box(0)])
        reports = v.update([box(0), box(2)])  # both overlap track 1; only the better one continues it
        self.assertEqual(sorted(r['track_id'] for r in reports), [1, 2])
        self.assertEqual(track(reports, 1)['box'][0], 0)
        self.assertEqual(len(match([[0, 0, 20, 30]], [[0, 0, 20, 30], [1, 0, 21, 30]], .1)), 1)

    def test_fast_small_object_stays_on_one_track_with_distance_fallback(self):
        # A 20 px box moving 30 px per frame never overlaps its previous box (IoU 0).
        boxes = [[box(10 + 30*i)] for i in range(6)]
        iou_only, fallback = Verifier(4, 6), Verifier(4, 6, distance=3)
        for b in boxes:
            split, joined = iou_only.update(b), fallback.update(b)
        self.assertEqual(iou_only.next_id, 7)
        self.assertFalse(any(r['persistent'] for r in split))
        self.assertEqual([r['track_id'] for r in joined], [1])
        self.assertTrue(joined[0]['persistent'])

    def test_distance_fallback_limits(self):
        self.assertEqual(match([[0, 0, 20, 20]], [[100, 0, 120, 20]], .1, distance=3), [])  # 5 lengths away
        self.assertEqual(match([[0, 0, 20, 20]], [[30, 0, 110, 80]], .1, distance=3), [])  # 16x the area
        self.assertEqual(match([[0, 0, 20, 20]], [[30, 0, 50, 20]], .1, distance=3), [(0, 0)])
        # IoU pairs are taken first; the fallback only uses what is left.
        pairs = match([[0, 0, 20, 20], [200, 0, 220, 20]], [[5, 0, 25, 20], [240, 0, 260, 20]], .1, distance=3)
        self.assertEqual(sorted(pairs), [(0, 0), (1, 1)])

    def test_object_moving_across_the_frame(self):
        v = Verifier(3, 5)
        for x in range(0, 120, 5):  # consecutive boxes overlap; first and last do not
            reports = v.update([box(x)])
        self.assertEqual([r['track_id'] for r in reports], [1])
        self.assertAlmostEqual(reports[0]['stats']['motion_px'], 5)

    def test_different_object_at_same_location_after_expiry_needs_fresh_hits(self):
        v = Verifier(3, 4)  # default max_missed = 1
        for _ in range(3):
            v.update([box(0)])
        v.update([])
        v.update([])  # object A expired
        first = v.update([box(0)])[0]
        self.assertEqual(first['track_id'], 2)
        self.assertFalse(first['persistent'])  # B inherits no support from A
        v.update([box(0)])
        self.assertTrue(v.update([box(0)])[0]['persistent'])  # three fresh hits

    def test_alarm_confirmed_after_k_hits(self):
        v = Verifier(3, 4)
        results = [v.update(d)[0]['persistent'] for d in ([box(0)], [box(0)], [box(0)], [box(0)])]
        self.assertEqual(results, [False, False, True, True])

    def test_statistics_only_from_window(self):
        v = Verifier(1, 4, max_missed=3)
        for d in ([box(0, score=.2)], [box(0, score=.2)], [box(0, score=.9)], [], [], [box(0, score=.9)]):
            report = v.update(d)[0]
        self.assertEqual(report['stats']['hits'], 2)
        self.assertEqual(report['stats']['observations'], 2)
        self.assertAlmostEqual(report['stats']['score_mean'], .9)

    def test_invalid_settings(self):
        for arguments in [dict(min_hits=4, window=3), dict(min_hits=0, window=3)]:
            with self.assertRaises(ValueError):
                Verifier(**arguments)
        with self.assertRaises(ValueError):
            Verifier(2, 3, rules={'min_colour': 1})


class FeatureTests(unittest.TestCase):
    def run_sequence(self, values, rules=None, move=0):
        v = Verifier(2, 8, max_missed=6, rules=rules)
        for i, value in enumerate(values):
            x = 20 + move*i
            reports = v.update([[x, 20, x+40, 60, .9]], frame(60, (x, 20, x+40, 60), value), i/30)
        return reports[0]

    def test_invalid_or_empty_rois(self):
        image = frame()
        self.assertIsNone(roi_features(image, np.array([10, 10, 10, 30])))  # zero width
        self.assertIsNone(roi_features(image, np.array([500, 500, 520, 520])))  # off-frame
        self.assertIsNone(roi_features(image, np.array([np.nan, 0, 10, 10])))
        self.assertIsNone(roi_features(None, np.array([0, 0, 10, 10])))
        self.assertIsNotNone(roi_features(image, np.array([-5, -5, 10, 10])))  # clipped to the frame
        report = Verifier(1, 3).update([[0, 0, 10, 10, .9]], None)[0]  # no frame: tracking still works
        self.assertTrue(report['persistent'])
        self.assertIsNone(report['stats']['flicker'])

    def test_static_region(self):
        stats = self.run_sequence([200]*6)['stats']
        for key in ('flicker', 'intensity_cv', 'area_cv', 'aspect_cv', 'motion_px', 'saturation_std', 'value_std'):
            self.assertEqual(stats[key], 0, key)

    def test_changing_intensity(self):
        stats = self.run_sequence([120, 220, 140, 230, 110, 210])['stats']
        self.assertGreater(stats['flicker'], .2)
        self.assertGreater(stats['intensity_cv'], .2)
        self.assertGreater(stats['value_std'], 20)

    def test_rules_need_evidence_and_only_add_to_persistence(self):
        self.assertFalse(self.run_sequence([200]*6, {'min_flicker': .05})['fused'])
        flickering = self.run_sequence([120, 220]*3, {'min_flicker': .05})
        self.assertTrue(flickering['persistent'] and flickering['fused'])
        single = Verifier(1, 4, rules={'min_flicker': .05}).update([[20, 20, 60, 60, .9]], frame(), 0)[0]
        self.assertTrue(single['persistent'])
        self.assertFalse(single['fused'])  # flicker not measurable from one frame: rule fails, never passes

    def test_motion_rule(self):
        report = self.run_sequence([200]*6, {'max_motion_relative': .05}, move=4)
        self.assertGreater(report['stats']['motion_relative'], .05)
        self.assertFalse(report['checks']['max_motion_relative'])

    def test_saturation_rule(self):
        # Saturation is 0-255 (OpenCV HSV). A saturated orange ROI fails, a whitish-yellow core passes.
        orange = self.run_sequence([(0, 120, 255)]*6, {'max_saturation': 100})
        core = self.run_sequence([(200, 240, 255)]*6, {'max_saturation': 100})
        self.assertGreater(orange['stats']['saturation_mean'], 200)
        self.assertFalse(orange['fused'])
        self.assertLess(core['stats']['saturation_mean'], 100)
        self.assertTrue(core['persistent'] and core['fused'])

    def test_circular_hue_wraps_around_red(self):
        mean, std = circular_hue([178, 179, 0, 1])
        self.assertLess(min(mean, 180-mean), 1)
        self.assertLess(std, 3)
        self.assertGreater(circular_hue([0, 45, 90, 135])[1], 30)

    def test_uniform_colour_roi(self):
        # Regression: a single-hue ROI made the circular-mean length round above 1 (math domain error).
        self.assertEqual(circular_hue([15]*1024)[1], 0)
        coloured = frame(60, (20, 20, 60, 60), (0, 120, 255))
        features = roi_features(coloured, np.array([20, 20, 60, 60]))
        self.assertAlmostEqual(features['hue_mean'], 14, delta=1)


if __name__ == '__main__':
    unittest.main()
