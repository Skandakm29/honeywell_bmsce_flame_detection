"""2-panel comparison on the 8 Nov clips: V10 + small-box threshold | V12, both with the same WARNING/ALARM rule.

Same base as compare3.py (5 Nov): every frame is decoded once and the SAME pixels go to both ONNX models, using
the live viewer's own prepare()/decode() (aspect-preserving resize to 640, pad 114, NMS 0.65). V10 runs once;
"V10 + box" reuses V10's detections with the week-2 rule (week2_final/rule.json):
  - a box counts if score >= 0.40, or if its longest side at the 640 input is < 16 px and score >= 0.25
  - WARNING = any counted box in the frame; ALARM = locked temporal rule (2 s window, hit ratio 0.30, off after 3 s),
    evaluated at 10 fps as tuned, held between samples.
  python compare_v10box_v12.py infer   then   python compare_v10box_v12.py render
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(r'E:\Projects\honeywell')
sys.path[:0] = [str(ROOT / r'honeywell_bmsce_flame_detection\scripts'), str(ROOT / r'investigation_week2\temporal_final'),
                str(ROOT / r'investigation_week2\v10_package\workflow')]
from live_flame import decode, prepare  # noqa: E402
import temporal_final as TF            # noqa: E402  (gate, tracker, alarm rule exactly as locked)

MODELS = {'V10': (ROOT / r'Flame_Detection_Test_Pack\YOLOv9_V10\flame.onnx', '6686d04a'),
          'V12': (ROOT / r'investigation_week2\lab_backup_2026-11-07\work\select_v12_rec\export\flame.onnx', '14bee807')}
RULE = json.loads((ROOT / r'investigation_week2\week2_final\rule.json').read_text())
FLOOR, CONF = .25, .40
CLIPS = {
    'fire_candle_outdoor_night_hand_occlusion.mov': 'Lit candle outdoors at night next to a street lamp, hand passes (FIRE)',
    'nofire_candle_unlit_outdoor_night_streetlamp.mov': 'Same scene, candle NOT lit, street lamp (NO fire)',
    'fire_candle_corridor_near_to_far.mov': 'Lit candle under corridor ceiling lights, getting smaller (FIRE)',
    'fire_lighter_far_to_close.mp4': 'Lighter moving from far to close (FIRE)',
    'nofire_diwali_string_lights_orange.mov': 'Orange LED string lights on trees (NO fire)',
}
PANELS = ('V10 + box', 'V12')
SEPARATE = False   # --separate: also write one video per model
CAMERA, CONTACT = None, None   # --camera camera.json / --contact-height-m (spatial calibration, optional)
COLOURS = {'V10': (0, 220, 0), 'V10 + box': (0, 200, 255), 'V12': (255, 160, 0)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer(folder, out):
    sessions = {}
    for name, (path, prefix) in MODELS.items():
        if not sha(path).startswith(prefix):
            raise RuntimeError(f'{name}: unexpected model hash')
        o = ort.SessionOptions()
        o.intra_op_num_threads = 6
        sessions[name] = ort.InferenceSession(str(path), o, providers=['CPUExecutionProvider'])
    for clip in CLIPS:
        video = folder / clip
        logs = out / 'logs' / video.stem
        logs.mkdir(parents=True, exist_ok=True)
        if (logs / 'done.json').exists():
            print('skip', clip, flush=True)
            continue
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        files = {k: open(logs / f'{k}.jsonl', 'w') for k in MODELS}
        index, started = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tensor, scale = prepare(frame, 640, 'yolov9')
            for name, s in sessions.items():
                rows = decode(s.run(None, {'images': tensor})[0], frame, scale, FLOOR, 'yolov9')
                files[name].write(json.dumps(dict(frame=index, t=round(index / fps, 3),
                                                  dets=[[round(float(v), 2) for v in r[:4]] + [round(float(r[4]), 4)] for r in rows])) + '\n')
            index += 1
        for f in files.values():
            f.close()
        (logs / 'done.json').write_text(json.dumps(dict(video=str(video), frames=index, fps=fps,
                                                        seconds=round(time.time() - started, 1))))
        print(f'{clip}: {index} frames in {time.time() - started:.0f} s', flush=True)


def week2_states(frames, to640, small_t):
    """Per frame: counted boxes (gate), WARNING, ALARM (rule at 10 fps, held between samples)."""
    counted = [TF.gate(f['dets'], to640, small_t) for f in frames]
    sampled, next_t = [], 0.
    for i, f in enumerate(frames):
        if f['t'] + 1e-6 >= next_t:
            sampled.append(i)
            next_t += .1
    sub = [frames[i] for i in sampled]
    per = TF.track(sub, to640, small_t)
    times = [f['t'] for f in sub]
    flags = TF.raw_flags(times, per, RULE['window_s'], RULE['min_hit_ratio'], RULE['min_mean_score'])
    on = TF.alarm(times, flags, RULE['alarm_on_after_s'])
    alarm, k = [], -1
    for i in range(len(frames)):
        while k + 1 < len(sampled) and sampled[k + 1] <= i:
            k += 1
        alarm.append(on[k] if k >= 0 else False)
    return counted, [bool(c) for c in counted], alarm


def render(folder, out):
    (out / 'grid').mkdir(exist_ok=True)
    summary = {}
    for clip, caption in CLIPS.items():
        video = folder / clip
        logs = {k: [json.loads(l) for l in open(out / 'logs' / video.stem / f'{k}.jsonl')] for k in MODELS}
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        to640 = 640 / max(W, H)
        gcam = ov = None
        if CAMERA is not None:   # distance_m/bearing_deg/x_m/y_m per box + range rings (rings built once per clip)
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
            from geometry.overlay import RingOverlay, fit_to_frame
            gcam = fit_to_frame(CAMERA, int(W), int(H))
            ov = RingOverlay(gcam, int(W), int(H))
            (out / 'geo').mkdir(exist_ok=True)
        geo_files, first_geo = {}, {p: None for p in PANELS}
        st = {'V10 + box': week2_states(logs['V10'], to640, RULE['small_box']['threshold']),
              'V12': week2_states(logs['V12'], to640, None)}
        stats = {p: 0 for p in PANELS}
        first = {p: None for p in PANELS}
        alarm_frames = {p: 0 for p in PANELS}
        first_alarm = {p: None for p in PANELS}
        writer, index = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            if ov is not None:
                ov.apply(frame)
            boxes = {p: st[p][0][index] for p in PANELS}
            panels = []
            scale_down = 640 / max(frame.shape[:2])          # keep the grid a sensible size
            for p in PANELS:
                panel = cv2.resize(frame, (round(frame.shape[1] * scale_down), round(frame.shape[0] * scale_down)))
                geo = [gcam.box_to_floor(b[:4], CONTACT) for b in boxes[p]] if gcam else []
                for i, (x1, y1, x2, y2, s) in enumerate(boxes[p]):
                    q = [int(v * scale_down) for v in (x1, y1, x2, y2)]
                    cv2.rectangle(panel, (q[0], q[1]), (q[2], q[3]), COLOURS[p], 2)
                    d = geo[i]['distance_m'] if geo else None
                    cv2.putText(panel, f'{s:.2f}' + (f' {d:.1f}m' if d is not None else ''), (q[0], max(12, q[1] - 4)), 0, .45, COLOURS[p], 1)
                if boxes[p]:
                    stats[p] += 1
                    first[p] = t if first[p] is None else first[p]
                bar = np.zeros((30, panel.shape[1], 3), np.uint8)
                label = f'{p}  boxes: {len(boxes[p])}'
                cv2.putText(bar, label, (6, 20), 0, .55, COLOURS[p], 1)
                _, warning, alarm = st[p]
                state, col = ('ALARM', (0, 0, 255)) if alarm[index] else (('WARNING', (0, 200, 255)) if warning[index] else ('clear', (160, 160, 160)))
                cv2.putText(bar, state, (panel.shape[1] - 95, 20), 0, .6, col, 2)
                if gcam:
                    if p not in geo_files:
                        geo_files[p] = open(out / 'geo' / f"{video.stem}_{p.replace(' + ', '_').replace(' ', '')}.jsonl", 'w')
                    geo_files[p].write(json.dumps(dict(frame=index, t=round(t, 3), alarm=bool(alarm[index]),
                                                       dets=[dict(box=[round(float(v), 1) for v in b[:4]], score=round(float(b[4]), 4), **g)
                                                             for b, g in zip(boxes[p], geo)])) + '\n')
                    if alarm[index] and first_alarm[p] is None:
                        first_geo[p] = geo
                if alarm[index]:
                    alarm_frames[p] += 1
                    first_alarm[p] = t if first_alarm[p] is None else first_alarm[p]
                panels.append(np.vstack([bar, panel]))
            gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
            grid = np.hstack([panels[0], gap, panels[1]])
            footer = np.zeros((26, grid.shape[1], 3), np.uint8)
            cv2.putText(footer, f'{caption}   |   t={t:4.1f}s   |   left: small boxes from 0.25; right: 0.40; both: same alarm rule',
                        (6, 18), 0, .45, (255, 255, 255), 1)
            grid = np.vstack([grid, footer])
            if writer is None:
                writer = cv2.VideoWriter(str(out / 'grid' / f'{video.stem}_V10box_vs_V12.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (grid.shape[1], grid.shape[0]))
                singles = {}
                if SEPARATE:
                    (out / 'single').mkdir(exist_ok=True)
                    for p, pan in zip(PANELS, panels):
                        tag = p.replace(' + ', '_').replace(' ', '')
                        singles[p] = cv2.VideoWriter(str(out / 'single' / f'{video.stem}_{tag}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                     fps, (pan.shape[1], pan.shape[0] + footer.shape[0]))
            writer.write(grid)
            for p, pan in zip(PANELS, panels):
                if p in singles:
                    f2 = np.zeros((footer.shape[0], pan.shape[1], 3), np.uint8)
                    cv2.putText(f2, f'{caption[:60]}  t={t:4.1f}s', (6, 18), 0, .45, (255, 255, 255), 1)
                    singles[p].write(np.vstack([pan, f2]))
            index += 1
        writer.release()
        for f in geo_files.values():
            f.close()
        for s in singles.values():
            s.release()
        summary[clip] = dict(frames=index, seconds=round(index / fps, 1),
                             frames_with_box={p: round(stats[p] / max(index, 1), 3) for p in PANELS},
                             first_box_s={p: first[p] for p in PANELS},
                             alarm_share={p: round(alarm_frames[p] / max(index, 1), 3) for p in PANELS}, first_alarm_s=first_alarm,
                             **(dict(first_alarm_geo=first_geo) if gcam else {}))
        print(clip, summary[clip], flush=True)
    (out / ('summary.json' if len(CLIPS) > 1 else f'summary_{Path(next(iter(CLIPS))).stem}.json')).write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['infer', 'render'])
    p.add_argument('--videos', default=str(ROOT / r'testing\videos_2026-11-08'))
    p.add_argument('--out', default=str(ROOT / r'testing\compare_2026-11-08'))
    p.add_argument('--clip', help='run only this file (inside --videos) with --caption')
    p.add_argument('--caption', default='')
    p.add_argument('--separate', action='store_true')
    p.add_argument('--camera', help='camera.json: add distance_m/bearing_deg/x_m/y_m to each box and alarm, draw rings')
    p.add_argument('--contact-height-m', type=float, help='override camera.json contact_height_m')
    a = p.parse_args()
    SEPARATE = a.separate
    if a.camera:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
        from geometry import Camera
        CAMERA, CONTACT = Camera.load(a.camera), a.contact_height_m
    if a.clip:
        CLIPS.clear()
        CLIPS[a.clip] = a.caption or a.clip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (infer if a.step == 'infer' else render)(Path(a.videos), out)
