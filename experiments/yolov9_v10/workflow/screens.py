"""Synthetic negatives: FASDD *train* flame images shown on Open Images screens (TV/monitor/phone/laptop/tablet).

Product decision (3 Nov): a fire shown on a screen must not raise an alarm, so these carry no boxes.
Group = the Open Images base image, so a screen image and its source never straddle train/hn_val.
"""
import argparse
import random

import cv2
import numpy as np

from common import make_record, read_json, work_dir, write_json


def render(base, flame, box, rng):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    w, h = x2 - x1, y2 - y1
    if w < 24 or h < 24:
        return None
    # cover-fit the flame image into the screen rectangle
    scale = max(w / flame.shape[1], h / flame.shape[0])
    resized = cv2.resize(flame, (max(w, int(flame.shape[1]*scale)+1), max(h, int(flame.shape[0]*scale)+1)))
    oy, ox = (resized.shape[0] - h) // 2, (resized.shape[1] - w) // 2
    content = resized[oy:oy+h, ox:ox+w].astype(np.float32) / 255.
    content = content ** rng.uniform(.75, 1.35) * rng.uniform(.6, 1.05)          # display gamma/brightness
    if rng.random() < .6:                                                         # scanlines / moire
        period = rng.uniform(2.5, 7.)
        rows = (np.sin(np.arange(h) * 2 * np.pi / period) * rng.uniform(.02, .07))[:, None, None]
        content = content * (1 - rows)
    if rng.random() < .5:
        content = cv2.GaussianBlur(content, (0, 0), rng.uniform(.4, 1.4))
    out = base.copy().astype(np.float32) / 255.
    alpha = rng.uniform(.82, 1.)                                                  # glass reflection of the room
    out[y1:y2, x1:x2] = alpha * content + (1 - alpha) * out[y1:y2, x1:x2]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    work = work_dir()
    rng = random.Random(a.seed)
    bases = [r for r in read_json(work / 'sources' / 'openimages.json')['records'] if r.get('screens')]
    flames = [r for r in read_json(work / 'manifests_fasdd' / 'train.json') if r['boxes']]
    if not bases or not flames:
        raise RuntimeError('Need Open Images screen boxes and FASDD train manifests first')
    out = work / 'data' / 'screens'
    out.mkdir(parents=True, exist_ok=True)
    records, attempts = [], 0
    while len(records) < a.count and attempts < a.count * 3:
        attempts += 1
        base_record = rng.choice(bases)
        base = cv2.imread(base_record['image'])
        flame = cv2.imread(rng.choice(flames)['image'])
        box = max(base_record['screens'], key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        image = render(base, flame, box, rng) if base is not None and flame is not None else None
        if image is None:
            continue
        path = out / f'{base_record["source_image_id"]}_{len(records):05d}.jpg'
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append(make_record(path, [], source='screens', category='synthetic_screen',
                                   group=base_record['group'], screen_box=box))
    print(f'{len(records)} synthetic screen images')
    write_json(dict(records=records), work / 'sources' / 'screens.json')


if __name__ == '__main__':
    main()
