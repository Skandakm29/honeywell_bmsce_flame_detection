"""Step 1: find flames left UNBOXED inside training images that do have flame boxes (never audited before).

Frozen V10 runs over every unique positive training image of manifests_v10/train.json. A candidate is a V10 box
with score >= .50 that has IoU < .10 with every labelled box and whose centre lies outside every labelled box.
Each candidate crop is scored with the same zero-shot CLIP fire scorer used in V10 (P(fire) vs nuisance prompts).

Decision rule (fixed in PROTOCOL_V11.md before any result):
  add_box    V10 score >= .60 and CLIP P(fire) >= .70  -> the V10 box is added as a label
  drop_image V10 score >= .50 and CLIP P(fire) >= .30 otherwise (ambiguous) -> the whole image leaves training
  ignore     CLIP P(fire) < .30 -> image unchanged (V10 false positive on clutter)
  One drop_image candidate drops its image even if the image also has add_box candidates.
Spot check (step 2 gate): 72 random add_box crops (seed 0) are shown on 2 sheets. If fewer than 80% are real
physical flames, every add_box becomes drop_image (fall back to the conservative V10 policy).
Validation / hn_val / test are never touched.
"""
import random
import zipfile
from pathlib import Path

import numpy as np

from common import Clip, FireScorer, contact_sheet, crop, detect_records, load_detector, read_json, work_dir, write_json
from paths11 import V10_FROZEN, V10_FROZEN_SHA, OUTBOX, iou


def main():
    work = work_dir()
    out = work / 'labelfix_v11'
    out.mkdir(parents=True, exist_ok=True)
    train = read_json(work / 'manifests_v10' / 'train.json')
    unique = {}
    for r in train:
        if r['boxes'] and r['image'] not in unique:
            unique[r['image']] = r
    records = list(unique.values())
    print('unique positive training images', len(records), flush=True)
    model = load_detector(V10_FROZEN, V10_FROZEN_SHA)
    detections = detect_records(model, records, floor=.5)
    del model
    candidates = []
    for r, dets in zip(records, detections):
        gt = np.asarray(r['boxes'], float).reshape(-1, 4)
        for d in dets:
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            inside = ((gt[:, 0] <= cx) & (cx <= gt[:, 2]) & (gt[:, 1] <= cy) & (cy <= gt[:, 3])).any()
            if iou([d[:4]], gt).max(initial=0) < .1 and not inside:
                candidates.append(dict(image=r['image'], box=[round(v, 1) for v in d[:4]], score=round(d[4], 4),
                                       source=r.get('source'), width=r['width'], height=r['height']))
    print('candidates', len(candidates), flush=True)
    import cv2
    scorer = FireScorer(Clip())
    for start in range(0, len(candidates), 256):
        chunk = candidates[start:start + 256]
        pictures = [crop(cv2.imread(c['image']), c['box']) for c in chunk]
        p_fire, nearest = scorer.score(scorer.clip.images(pictures))
        for c, p, n in zip(chunk, p_fire, nearest):
            c['clip_p_fire'], c['clip_nearest'] = round(float(p), 4), n
    for c in candidates:
        c['decision'] = ('add_box' if c['score'] >= .6 and c['clip_p_fire'] >= .7 else
                         'drop_image' if c['clip_p_fire'] >= .3 else 'ignore')
    write_json(candidates, out / 'candidates.json')
    counts = {k: sum(c['decision'] == k for c in candidates) for k in ('add_box', 'drop_image', 'ignore')}
    images = {k: len({c['image'] for c in candidates if c['decision'] == k}) for k in counts}
    write_json(dict(positive_images=len(records), candidates=len(candidates), boxes=counts, images=images,
                    rule=__doc__.strip()), out / 'summary.json')
    print('decisions (boxes)', counts, 'images', images, flush=True)
    rng = random.Random(0)
    sheets = []
    for name, k, n in (('spotcheck_add_box', 'add_box', 72), ('sample_drop_image', 'drop_image', 36),
                       ('sample_ignore', 'ignore', 36)):
        pool = [c for c in candidates if c['decision'] == k]
        pick = rng.sample(pool, min(n, len(pool)))
        write_json(pick, out / f'{name}.json')
        for s in range(0, len(pick), 36):
            path = out / f'{name}_{s // 36 + 1}.jpg'
            contact_sheet([(c['image'], c['box'], f'{s + i + 1} s{c["score"]:.2f} p{c["clip_p_fire"]:.2f}')
                           for i, c in enumerate(pick[s:s + 36])], path)
            sheets.append(path)
    OUTBOX.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTBOX / 'v11_step1_labelfix.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sheets + [out / 'summary.json', out / 'spotcheck_add_box.json']:
            z.write(path, path.name)
    print('Sheets in', out, '-> fill inbox/labelfix_spotcheck.json (see START_HERE.md) before step 2.')


if __name__ == '__main__':
    main()
