"""SYNTHETIC labelled clips for checking scripts/sweep_zone_thresholds.py end to end (not for choosing settings).

Objects are composited onto the empty boiler-room frame of week_4_test_3_seg3.mp4 (first frame), so the true
obstruction times are known exactly. 10 s each at 10 fps; clear for the first 2 s (learning).
Writes test_clips/synth_*.mp4 (git-ignored) and configs/zone_sweep_synthetic.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS, SECONDS = 10, 10
ZONE = 'configs/zones/week4_test3_fire_area.json'


def person(frame, x, y, h, colour=(40, 50, 60)):
    w = h // 3
    cv2.rectangle(frame, (x, y + h // 5), (x + w, y + h), colour, -1)            # body
    cv2.circle(frame, (x + w // 2, y + h // 10), h // 10, (60, 80, 110), -1)      # head
    return frame


def haze(frame, t, density=.6):
    h, w = frame.shape[:2]
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random((2 * h, w)).astype(np.float32), (0, 0), 30)
    base = (base - base.min()) / (base.max() - base.min())
    s = int(t * 40) % h
    a = np.clip(base[h - s:2 * h - s] * 1.6 - .3, 0, 1) * density
    return (frame * (1 - a[..., None]) + 225 * a[..., None]).astype(np.uint8)


def scenarios(zone_px):
    x1, y1, x2, y2 = zone_px
    zw, zh = x2 - x1, y2 - y1
    return {
        'empty': (lambda f, t: f, []),
        'walk_past': (lambda f, t: person(f, int(x1 - 200 + 180 * t), y1, zh) if 2 <= t <= 9 else f, []),
        'two_walkers': (lambda f, t: person(person(f, int(x2 - 120 * (t - 2)), y1 + 10, zh - 20),
                                            int(x1 - 100 + 150 * (t - 2)), y1, zh) if t >= 2 else f, []),
        'person_stands': (lambda f, t: person(f, x1 + zw // 3, y1, zh) if 3 <= t < 8 else f, [[3., 8.]]),
        'box_left': (lambda f, t: cv2.rectangle(f, (x1 + zw // 6, y1 + zh // 3), (x1 + 4 * zw // 6, y2),
                                                (30, 70, 120), -1) if t >= 3 else f, [[3., 10.]]),
        'pallet_partial': (lambda f, t: cv2.rectangle(f, (x1, y1 + zh // 2), (x1 + zw // 2, y2), (60, 60, 60), -1)
                           if t >= 4 else f, [[4., 10.]]),
        'small_object': (lambda f, t: cv2.rectangle(f, (x1 + zw // 2, y2 - zh // 6), (x1 + zw // 2 + zw // 8, y2),
                                                    (30, 30, 30), -1) if t >= 3 else f, []),
        'steam': (lambda f, t: haze(f, t) if t >= 3 else f, []),
        'lights_dim': (lambda f, t: (f * .65).astype(np.uint8) if t >= 4 else f, []),
        'person_then_box': (lambda f, t: (person(f, x1 + zw // 4, y1, zh) if 3 <= t < 5 else f) if t < 6 else
                            cv2.rectangle(f, (x1 + zw // 5, y1 + zh // 4), (x1 + 4 * zw // 5, y2), (30, 70, 120), -1),
                            [[3., 5.], [6., 10.]]),
    }


def main():
    cap = cv2.VideoCapture(str(ROOT / 'test_clips/week_4_test_3_seg3.mp4'))
    ok, background = cap.read()
    if not ok:
        raise SystemExit('cannot read the background clip')
    zone = json.loads((ROOT / ZONE).read_text())['zones'][0]['polygon']
    h, w = background.shape[:2]
    zone_px = (int(min(p[0] for p in zone) * w), int(min(p[1] for p in zone) * h),
               int(max(p[0] for p in zone) * w), int(max(p[1] for p in zone) * h))
    clips = []
    for i, (name, (draw, intervals)) in enumerate(scenarios(zone_px).items()):
        path = ROOT / f'test_clips/synth_{name}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
        for k in range(FPS * SECONDS):
            writer.write(draw(background.copy(), k / FPS))
        writer.release()
        clips.append(dict(video=f'../test_clips/synth_{name}.mp4', scenario=name, obstruction_intervals=intervals,
                          split='tune' if i % 2 == 0 else 'test'))
    manifest = dict(_comment='SYNTHETIC labelled clips (experiments/week4_sweep/make_labelled_clips.py): a pipeline '
                             'check of the sweep, not a basis for choosing settings.',
                    settings=dict(zones='zones/week4_test3_fire_area.json', learn_s=2, event_merge_gap_seconds=1.),
                    clips=clips)
    (ROOT / 'configs/zone_sweep_synthetic.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(len(clips), 'clips; manifest configs/zone_sweep_synthetic.json')


if __name__ == '__main__':
    main()
