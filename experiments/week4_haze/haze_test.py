"""SYNTHETIC haze / steam / smoke over recorded clips: pixel vs DINOv2 zone monitor, and flame detection under smoke.

Not real smoke: a drifting, smooth random field blended over the frame (light grey = steam, dark grey = smoke) at a
fixed peak density. It tests how each change detector reacts to see-through haze of rising density, and how the
unchanged flame model copes with smoke over a fire. Real smoke and steam must still be recorded.

Part A (no fire, week_4_test_3_seg3 0-8 s, empty boiler room): haze over the zone from 3 s; nothing is blocking, so
  any OBSTRUCTED alarm is a false alarm.
Part B (smoke over a fire, week_4_test.mp4, fire from ~3 s): smoke over the fire from 5 s; reports the share of
  frames with a flame box, the Week 2 ALARM, the zone result and the fire-visibility flag.
  python experiments/week4_haze/haze_test.py --model ~/Downloads/yolov9.onnx
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from flame_alarm import FlameAlarm  # noqa: E402
from live_flame import decode, prepare  # noqa: E402
from obstruction import ReferenceDetector, SceneMonitor, load_zones  # noqa: E402
from obstruction.deep import DeepChangeDetector, OnnxPatchFeatures  # noqa: E402

DENSITIES = (0., .2, .4, .6, .8, .95)


def haze_field(shape, t, seed=0):
    """Smooth 0..1 field drifting upwards (same for every density, so only the density changes): mostly 0.4-1 with
    thinner gaps, like a plume."""
    h, w = shape[:2]
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h * 2, w)).astype(np.float32), (0, 0), 25)
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    shift = int(t * 40) % h                      # rises 40 px per second
    return np.clip(base[h - shift:2 * h - shift] * 1.4 - .1, .25, 1.)


def apply_haze(frame, density, t, region, colour, start_s):
    """Blend colour over the region with opacity density x field (feathered edges, fading in over 0.5 s)."""
    if density <= 0 or t < start_s:
        return frame
    x1, y1, x2, y2 = region
    edge = np.zeros(frame.shape[:2], np.float32)
    edge[y1:y2, x1:x2] = 1.
    edge = cv2.GaussianBlur(edge, (0, 0), 12)           # feather only the region border
    # Opacity = density x (0.75..1): a drifting texture, but the stated density is what covers the region.
    alpha = (.75 + .25 * haze_field(frame.shape, t)) * edge * density * min(1., (t - start_s) / .5)
    out = frame.astype(np.float32) * (1 - alpha[..., None]) + np.array(colour, np.float32) * alpha[..., None]
    return out.astype(np.uint8)


def run(frames, fps, zones, learn_s, session, density, region, colour, start_s, extractor):
    cfg = lambda: load_zones(zones)
    monitors = dict(pixel=SceneMonitor(cfg(), ReferenceDetector(learn_s=learn_s)),
                    deep=SceneMonitor(cfg(), DeepChangeDetector(extractor, pixel=ReferenceDetector(learn_s=learn_s),
                                                                every_s=.2, min_threshold=.2)))
    week2 = FlameAlarm(frame_dt=1 / fps)
    stats = {k: dict(alarm=False, alarm_from=None, max_persistent=0., compromised=False) for k in monitors}
    flame_frames = alarm_frames = hazed = 0
    for i, frame in enumerate(frames):
        t = i / fps
        frame = apply_haze(frame, density, t, region, colour, start_s)
        tensor, scale = prepare(frame, 640, 'yolov9')
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        boxes = decode(raw, frame, scale, .40, 'yolov9')
        w2 = week2.update(decode(raw, frame, scale, .25, 'yolov9'), t, frame.shape)
        if t >= start_s:
            hazed += 1
            flame_frames += len(boxes) > 0
            alarm_frames += w2['alarm']
        for name, m in monitors.items():
            o = m.update(frame, t, exclude_boxes=boxes, flame=w2)
            z = next(iter(o['zones'].values()))
            s = stats[name]
            if t >= start_s:
                s['max_persistent'] = max(s['max_persistent'], z['persistent_pct'])
                if z['active'] and not s['alarm']:
                    s['alarm'], s['alarm_from'] = True, round(t, 2)
                s['compromised'] |= o['visibility']['visibility_compromised']
    return dict(density=density, flame_box_frames_pct=round(100 * flame_frames / max(1, hazed), 1),
                week2_alarm_frames_pct=round(100 * alarm_frames / max(1, hazed), 1),
                **{f'{k}_{f}': v for k, s in stats.items() for f, v in s.items()})


def read(path, start_s, end_s):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or i / fps >= end_s:
            break
        if i / fps >= start_s:
            frames.append(frame)
        i += 1
    return frames, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/obstruction_eval/haze_synthetic')
    args = parser.parse_args()
    session = ort.InferenceSession(str(args.model.expanduser()), providers=['CPUExecutionProvider'])
    extractor = OnnxPatchFeatures()
    args.output.mkdir(parents=True, exist_ok=True)
    parts = {
        'A_steam_no_fire': dict(clip='week_4_test_3_seg3.mp4', zones='week4_test3_fire_area.json', clip_s=(0, 8),
                                learn_s=2., start_s=3., colour=(225, 225, 225)),
        'A_smoke_no_fire': dict(clip='week_4_test_3_seg3.mp4', zones='week4_test3_fire_area.json', clip_s=(0, 8),
                                learn_s=2., start_s=3., colour=(70, 70, 70)),
        'B_smoke_over_fire': dict(clip='week_4_test.mp4', zones='week4_test_bin.json', clip_s=(0, 10),
                                  learn_s=1., start_s=5., colour=(70, 70, 70), over_fire=True),
    }
    report, sheet = {}, []
    for name, p in parts.items():
        frames, fps = read(ROOT / 'test_clips' / p['clip'], *p['clip_s'])
        zones = ROOT / 'configs/zones' / p['zones']
        zone = load_zones(zones).zones[0].polygon
        h, w = frames[0].shape[:2]
        xs, ys = [q[0] * w for q in zone], [q[1] * h for q in zone]
        if p.get('over_fire'):                   # cover the detected fire itself (union of its boxes before haze)
            boxes = []
            for frame in frames[int(p['start_s'] * fps) - 10:int(p['start_s'] * fps)]:
                tensor, scale = prepare(frame, 640, 'yolov9')
                raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                boxes += decode(raw, frame, scale, .40, 'yolov9').tolist()
            if not boxes:
                raise SystemExit(f'{name}: no fire detected before the smoke starts')
            b = np.array(boxes)
            xs, ys = [b[:, 0].min(), b[:, 2].max()], [b[:, 1].min(), b[:, 3].max()]
        pad = .15 * (max(xs) - min(xs))
        region = (int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad)), int(min(w, max(xs) + pad)),
                  int(min(h, max(ys) + pad)))
        report[name] = []
        for density in DENSITIES:
            r = run(frames, fps, zones, p['learn_s'], session, density, region, p['colour'], p['start_s'], extractor)
            report[name].append(r)
            print(name, json.dumps(r), flush=True)
            sample_t = p['start_s'] + 2
            tile = apply_haze(frames[int(sample_t * fps)], density, sample_t, region, p['colour'], p['start_s'])
            tile = cv2.resize(tile, (384, 216))
            cv2.putText(tile, f'{name} d={density}', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            sheet.append(tile)
    rows = [np.hstack(sheet[i:i + len(DENSITIES)]) for i in range(0, len(sheet), len(DENSITIES))]
    cv2.imwrite(str(args.output / 'haze_samples.jpg'), np.vstack(rows))
    (args.output / 'report.json').write_text(json.dumps(dict(model=str(args.model), parts=parts, results=report),
                                                       indent=2), encoding='utf-8')
    print(args.output / 'report.json')


if __name__ == '__main__':
    main()
