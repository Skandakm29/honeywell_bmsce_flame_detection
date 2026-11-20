import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_clips import events, judge, totals


class EvaluationTests(unittest.TestCase):
    times = [i / 10 for i in range(30)]  # 3 s at 10 frames per second

    def flags(self, *on):
        return [any(a <= t <= b for a, b in on) for t in self.times]

    def test_repeated_boxes_on_one_object_are_one_event(self):
        flickering = [i % 2 == 0 for i in range(10)]  # alarm every other frame for 1 s
        self.assertEqual(len(events(self.times[:10], flickering, gap=.25)), 1)
        self.assertEqual(len(events(self.times[:10], flickering, gap=0)), 5)
        self.assertEqual(events(self.times, [False]*30, gap=1), [])

    def test_frame_and_event_metrics(self):
        result = judge(self.times, self.flags((1.2, 1.9), (2.6, 2.9)), [[1.0, 2.0]], gap=.15)
        self.assertEqual(result['positive_frames'], 11)  # 1.0..2.0 inclusive
        self.assertEqual(result['true_alarm_frames'], 8)  # 1.2..1.9
        self.assertAlmostEqual(result['frame_precision'], 8 / 12)
        self.assertAlmostEqual(result['frame_recall'], 8 / 11)
        self.assertEqual((result['genuine_events'], result['false_alarm_events']), (1, 1))
        self.assertEqual(result['missed_flame_intervals'], 0)
        self.assertEqual(result['delay_seconds'], [0.2])

    def test_missed_interval_and_negative_clip(self):
        missed = judge(self.times, self.flags((0.0, 0.3)), [[1.0, 2.0]], gap=.15)
        self.assertEqual(missed['missed_flame_intervals'], 1)
        self.assertEqual(missed['event_recall'], 0)
        self.assertEqual(missed['delay_seconds'], [None])
        negative = judge(self.times, self.flags((0.5, 0.8)), [], gap=.15)
        self.assertEqual(negative['false_alarm_events'], 1)
        self.assertIsNone(negative['event_recall'])
        self.assertIsNone(negative['frame_recall'])

    def test_unlabelled_clip_is_not_judged(self):
        result = judge(self.times, self.flags((0.5, 0.8)), None, gap=.15)
        self.assertEqual(result['ground_truth'], 'not labelled')
        self.assertNotIn('false_alarm_events', result)
        self.assertIsNone(totals([{'raw': result}], 'raw'))


if __name__ == '__main__':
    unittest.main()
