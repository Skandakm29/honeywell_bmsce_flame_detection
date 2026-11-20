"""The live Week 2 alarm (src/flame_alarm.py) must equal the locked offline rule (experiments/temporal_final)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments/temporal_final'), str(ROOT / 'experiments/yolov9_v10/workflow')]
import numpy as np
import temporal_final as TF
from flame_alarm import FlameAlarm

RULE = json.loads((ROOT / 'results/week2/rule.json').read_text())
SHAPE = (720, 1280, 3)


def offline(frames):
    """frames: [(t, [[x1, y1, x2, y2, score], ...])] -> per-frame (assigned, alarm) from the locked offline code."""
    to640 = 640 / max(SHAPE[:2])
    data = [dict(t=t, dets=[list(map(float, d)) for d in dets]) for t, dets in frames]
    per = TF.track(data, to640, RULE['small_box']['threshold'])
    times = [t for t, _ in frames]
    flags = TF.raw_flags(times, per, RULE['window_s'], RULE['min_hit_ratio'], RULE['min_mean_score'])
    return per, TF.alarm(times, flags, RULE['alarm_on_after_s'])


def live(frames, fps):
    alarm = FlameAlarm(RULE, frame_dt=1 / fps)
    outs = [alarm.update(np.array(d, float).reshape(-1, 5), t, SHAPE) for t, d in frames]
    return [o['assigned'] for o in outs], [o['alarm'] for o in outs]


def random_clip(seed, n=600, fps=10):
    """Flickering fires that move, split, vanish; small boxes near the small-box gate; random false boxes."""
    rng = np.random.default_rng(seed)
    frames, fires = [], [[400., 300., 460., 380.], [900., 200., 908., 210.]]
    for i in range(n):
        dets = []
        for j, f in enumerate(fires):
            f[0] += rng.normal(0, 3); f[2] += rng.normal(0, 3)
            on = (i // 80) % 3 != 2 if j == 0 else (i // 50) % 2 == 0
            if on and rng.random() < (.35 if j == 0 else .5):
                dets.append([*f, float(rng.uniform(.2, .9))])
        if rng.random() < .05:
            x, y = rng.uniform(0, 1200), rng.uniform(0, 650)
            dets.append([x, y, x + rng.uniform(5, 80), y + rng.uniform(5, 80), float(rng.uniform(.3, .7))])
        frames.append((i / fps, dets))
    return frames


class FlameAlarmEquivalence(unittest.TestCase):
    def test_same_tracks_and_alarm_as_the_locked_offline_rule(self):
        for seed in range(5):
            for fps in (10, 5, 24):
                frames = random_clip(seed, fps=fps)
                per_off, alarm_off = offline(frames)
                per_live, alarm_live = live(frames, fps)
                self.assertEqual(alarm_live, alarm_off, f'seed {seed} fps {fps}')
                self.assertEqual([sorted(k for k, _ in a) for a in per_live],
                                 [sorted(k for k, _ in a) for a in per_off], f'seed {seed} fps {fps}')
                self.assertTrue(any(alarm_off) and not all(alarm_off))           # the clip exercises both states

    def test_confirmed_only_after_the_flag_rule(self):
        alarm = FlameAlarm(RULE, frame_dt=.1)
        box = [[400, 300, 460, 380, .8]]
        outs = [alarm.update(np.array(box if i % 5 == 0 else []).reshape(-1, 5), i / 10, SHAPE) for i in range(20)]
        self.assertFalse(any(k['confirmed'] for o in outs for k in o['tracks']))  # 10% of frames < 30%
        outs = [alarm.update(np.array(box).reshape(-1, 5), 2 + i / 10, SHAPE) for i in range(10)]
        self.assertTrue(outs[-1]['tracks'][0]['confirmed'])
        self.assertTrue(outs[-1]['alarm'])


if __name__ == '__main__':
    unittest.main()
