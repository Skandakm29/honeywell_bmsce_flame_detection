"""Step 1: manifests_w4 + preview sheet.

  train.json, hn_val.json, validation_full.json, test.json  copied byte-for-byte from the source manifests
                                                            (training obstructions are drawn on the fly)
  val.json   = the 2,000-image monitor set (unchanged) + N fixed synthetic obstructions on validation_full images
               that are NOT in the monitor set (donor images for 'blurred_scene' come from the same pool).
               Nothing from train, hn_val or test is used, so there is no leakage.
  preview_val.jpg     48 of the synthetic validation images with their boxes (red = flame, blue = obstruction).
                      GATE: a person looks at it before training starts.
  build_report.json   counts, kinds/shapes, blind-share histogram, flames hidden by obstructions
"""
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import read_json, sha256, write_json
from common_w4 import CONFIG, MANIFESTS_W4, SOURCE_MANIFESTS
from data_w4 import FLAME, Flames2, labels_of, synthetic_record
import occluders as O


def build(source=SOURCE_MANIFESTS, out=MANIFESTS_W4, n=None, seed=None):
    source, out = Path(source), Path(out)
    v = CONFIG['validation']
    n, seed = n if n is not None else v['synthetic_images'], seed if seed is not None else v['seed']
    out.mkdir(parents=True, exist_ok=True)
    for name in ('train.json', 'hn_val.json', 'validation_full.json', 'test.json'):
        shutil.copyfile(source / name, out / name)
    monitor = read_json(source / 'val.json')
    full = read_json(source / 'validation_full.json')
    used = {r['image'] for r in monitor}
    pool = [r for r in full if r['image'] not in used]
    if len(pool) < n:
        raise RuntimeError(f'only {len(pool)} validation_full images outside the monitor set; need {n}')
    rng = random.Random(seed)
    order = rng.sample(range(len(pool)), len(pool))
    occ = CONFIG['occluder']
    synthetic, k = [], 0
    while len(synthetic) < n and k < len(order):
        record = pool[order[k]]
        k += 1
        spec = O.sample_spec(rng, occ)
        if spec['kind'] == 'blurred_scene':
            spec['donor'] = pool[rng.randrange(len(pool))]['image']
        made = synthetic_record(record, spec, hide_if_covered=occ['hide_if_covered'], min_fraction=occ['min_fraction'])
        if made is not None:
            synthetic.append(made)
    if len(synthetic) < n:
        raise RuntimeError(f'could only make {len(synthetic)} synthetic validation images')
    val = monitor + synthetic
    write_json(val, out / 'val.json')
    fractions = np.array([r['blind_fraction'] for r in synthetic])
    report = dict(source=str(source), source_sha256={name: sha256(source / name) for name in
                                                     ('train.json', 'val.json', 'validation_full.json', 'hn_val.json', 'test.json')},
                  output_sha256={name: sha256(out / name) for name in
                                 ('train.json', 'val.json', 'validation_full.json', 'hn_val.json', 'test.json')},
                  monitor_images=len(monitor), synthetic_images=len(synthetic), seed=seed,
                  kinds=dict(Counter(r['occluder']['kind'] for r in synthetic)),
                  shapes=dict(Counter(r['occluder']['shape'] for r in synthetic)),
                  blind_share_histogram={f'{10 * i}-{10 * i + 10}%': int(c) for i, c in
                                         enumerate(np.histogram(fractions, bins=10, range=(0., 1.))[0])},
                  flames_hidden=int(sum(r['hidden_flames'] for r in synthetic)),
                  synthetic_with_visible_flame=int(sum(FLAME in labels_of(r) for r in synthetic)),
                  leakage='synthetic images and donors come only from validation_full minus the monitor set')
    write_json(report, out / 'build_report.json')
    preview(synthetic[:48], out / 'preview_val.jpg')
    print({k: report[k] for k in ('monitor_images', 'synthetic_images', 'kinds', 'shapes', 'flames_hidden')}, flush=True)
    return report


def preview(records, path, tile=320, cols=6):
    dataset = Flames2(records, 640)
    tiles = []
    for i, r in enumerate(records):
        canvas, boxes, labels, _ = dataset.load2(i, random.Random(0))
        h, w = round(r['height'] * 640 / max(r['width'], r['height'])), round(r['width'] * 640 / max(r['width'], r['height']))
        image = np.ascontiguousarray(canvas[:h, :w])
        for (x1, y1, x2, y2), l in zip(boxes, labels):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), (0, 0, 255) if l == FLAME else (255, 128, 0), 2)
        cv2.putText(image, f'{r["occluder"]["kind"]}/{r["occluder"]["shape"]} {100 * r["blind_fraction"]:.0f}%',
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        scale = tile / max(image.shape[:2])
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        cell = np.full((tile, tile, 3), 255, np.uint8)
        cell[:image.shape[0], :image.shape[1]] = image
        tiles.append(cell)
    while len(tiles) % cols:
        tiles.append(np.full((tile, tile, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    build()
