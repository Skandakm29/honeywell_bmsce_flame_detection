"""Score V9 / V10 / A1 on the user's test videos. Rules were fixed BEFORE looking at any model output.

Labels (from viewing 1-per-second frame strips; Claude's visual labelling, not human):
  8.54.37 PM  skipped entirely (owner: flame not visible in the video)
  8.54.39 PM  flame (lighter in a dark room, struck on/off) -> per-frame label: flame visible if >= 3 pixels
              have max channel > 200 (measured: on-frames reach 255 with 8-35 such pixels, off-frames max <= 35)
  8.54.42 PM  no flame: laptop screen playing a fire video
  8.54.46 PM  no flame: LED lantern + orange/red objects (handheld camera)
  8.54.51 PM  flame: lit candle for the whole clip, next to an LED lantern, orange sock, red object
  8.54.53 PM  UNSCORED: lighter lit at times but not visible (owner) -> boxes reported, never called hit or false alarm
  8.54.57 PM  no flame: LED lantern + orange object, lights switching dark/bright
  8.56.03 PM  flame: lit red candle for the whole clip, camera walking away (near -> far)
Metrics at the operating point (confidence >= 0.40, same as live):
  flame frames: detection rate = frames with >= 1 box / flame-visible frames (frame level; box placement is
                checked separately by a visual audit of sampled boxes)
  non-flame frames: false-alarm frame rate = frames with >= 1 box / non-flame frames
  plus: longest run of consecutive missed flame frames (seconds), and for the lighter video the delay from each
        visible strike to the first detection.
No threshold, label or rule is changed after seeing results.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'E:\Projects\honeywell\testing')
OUT = ROOT / 'compare_out'
MODELS = ('V9', 'V10', 'A1')
CONF = .40
VIDEOS = {
    '8.54.39 PM': ('flame', 'lighter struck on/off, dark room', 'lighter'),
    '8.54.42 PM': ('none', 'laptop screen playing a fire video', None),
    '8.54.46 PM': ('none', 'LED lantern + orange/red objects', None),
    '8.54.51 PM': ('flame', 'lit candle + LED lantern + orange sock + red object', 'whole'),
    '8.54.53 PM': ('unscored', 'lighter, flame not visible', None),
    '8.54.57 PM': ('none', 'LED lantern + orange object, lights switching', None),
    '8.56.03 PM': ('flame', 'lit red candle, camera walking away (near -> far)', 'whole'),
}


def stem(key):
    return f'WhatsApp Video 2026-11-05 at {key}'


def lighter_labels(path):
    capture, labels = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        labels.append(int((frame.max(2) > 200).sum()) >= 3)
    return labels


def runs(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def main():
    table, audit = [], {}
    for key, (kind, desc, mode) in VIDEOS.items():
        folder = OUT / 'logs' / stem(key)
        done = json.loads((folder / 'done.json').read_text())
        fps, n = done['fps'], done['frames']
        logs = {m: [json.loads(l) for l in open(folder / f'{m}.jsonl')] for m in MODELS}
        assert all(len(v) == n for v in logs.values())
        if mode == 'lighter':
            labels = lighter_labels(ROOT / f'{stem(key)}.mp4')
        elif kind == 'flame':
            labels = [True] * n
        else:
            labels = [False] * n
        row = dict(video=key, kind=kind, content=desc, frames=n, seconds=round(n / fps, 1),
                   flame_frames=sum(labels), nonflame_frames=n - sum(labels))
        for m in MODELS:
            flagged = [any(d[4] >= CONF for d in r['dets']) for r in logs[m]]
            boxes = [sum(d[4] >= CONF for d in r['dets']) for r in logs[m]]
            r = {}
            if kind == 'unscored':
                r['frames_with_box'] = sum(flagged) / n
            else:
                if sum(labels):
                    hits = [f for f, l in zip(flagged, labels) if l]
                    r['detection_rate'] = sum(hits) / len(hits)
                    r['longest_miss_s'] = round(runs([not h for h in hits]) / fps, 2)
                    r['mean_boxes_on_flame_frames'] = round(float(np.mean([b for b, l in zip(boxes, labels) if l])), 2)
                if n - sum(labels):
                    r['false_alarm_frame_rate'] = sum(f for f, l in zip(flagged, labels) if not l) / (n - sum(labels))
                if kind == 'flame' and mode == 'whole':   # near vs far half for the walk-away clip
                    half = n // 2
                    r['detection_first_half'] = sum(flagged[:half]) / half
                    r['detection_second_half'] = sum(flagged[half:]) / (n - half)
                if mode == 'lighter':
                    delays, i = [], 0
                    while i < n:
                        if labels[i] and (i == 0 or not labels[i - 1]):          # strike onset
                            j = i
                            while j < n and labels[j] and not flagged[j]:
                                j += 1
                            delays.append(round((j - i) / fps, 2) if j < n and labels[j] else None)
                        i += 1
                    r['strike_delays_s'] = delays
            row[m] = r
        table.append(row)
        # audit sample: up to 18 boxes per model, evenly spread over the clip, for visual placement check
        if kind == 'flame':
            audit[key] = {}
            for m in MODELS:
                picks = [(r['frame'], d) for r in logs[m] for d in r['dets'] if d[4] >= CONF]
                step = max(1, len(picks) // 18)
                audit[key][m] = picks[::step][:18]
    (OUT / 'scores.json').write_text(json.dumps(table, indent=1))
    (OUT / 'audit_picks.json').write_text(json.dumps(audit))
    print(json.dumps(table, indent=1))


if __name__ == '__main__':
    main()
