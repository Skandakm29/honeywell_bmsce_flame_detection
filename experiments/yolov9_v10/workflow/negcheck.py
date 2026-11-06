"""Whole-image safety check for accepted public hard negatives (audit fix, automatic, no review round).

The mining review looked at one detection crop, but an empty label applies to the whole image. Every public image
accepted as a hard negative (D-Fire, Open Images, Roboflow) gets a whole-image CLIP P(fire); images at or above
--threshold are dropped from both training and hn_val (written to mine/negcheck.json).
"""
import argparse

import cv2

from common import Clip, FireScorer, read_json, work_dir, write_json

WILD = ('dfire.json', 'openimages.json', 'roboflow.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threshold', type=float, default=.5)
    a = p.parse_args()
    work = work_dir()
    decisions = read_json(work / 'mine' / 'decisions.json')['decisions']
    records = [r for name in WILD if (work / 'sources' / name).exists()
               for r in read_json(work / 'sources' / name)['records']
               if not r['boxes'] and decisions.get(r['image']) == 'hard_negative']
    clip = Clip()
    scorer = FireScorer(clip)
    scores = {}
    for start in range(0, len(records), 256):
        chunk = records[start:start+256]
        fire, _ = scorer.score(clip.images([cv2.imread(r['image']) for r in chunk]))
        scores.update({r['image']: float(f) for r, f in zip(chunk, fire)})
    drop = sorted(i for i, s in scores.items() if s >= a.threshold)
    by_source = {}
    for r in records:
        if r['image'] in drop:
            by_source[r['source']] = by_source.get(r['source'], 0) + 1
    write_json(dict(threshold=a.threshold, checked=len(records), drop=drop, dropped_by_source=by_source,
                    scores=scores), work / 'mine' / 'negcheck.json')
    print(f'whole-image check: {len(records)} accepted public hard negatives, {len(drop)} dropped', by_source)


if __name__ == '__main__':
    main()
