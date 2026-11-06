"""Hard-negative mining with V9 and the review hand-off.

run       V9 (score floor .20) on every candidate negative + FASDD train no-box images; CLIP P(fire) on hits.
          Writes outbox/step2_review.zip (contact sheets + review_index.csv) for review on the laptop.
finalize  Reads inbox/review_answers.csv (id,verdict: fire|not_fire|unsure) and writes mine/decisions.json.

Rules (labels come from script + CLIP + AI visual review, never called human-verified):
  wild images (D-Fire negatives, Open Images) with a V9 hit:
      CLIP P(fire) of every hit crop < .15 and max score < .50  -> hard negative (auto)
      any crop P(fire) >= .85                                    -> dropped (probably real fire)
      otherwise                                                  -> review
  wild images without a hit but whole-image P(fire) >= .50       -> dropped
  own non-fire clips and synthetic screens are negatives by construction; their hits are hard negatives.
  FASDD train no-box images with V9 >= .40                       -> review; confirmed fire -> excluded from training
"""
import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from common import (PACKAGE, V9_SHA256, Clip, FireScorer, contact_sheet, crop, detect_records, load_detector,
                    read_json, work_dir, write_json)

MAX_REVIEW = 3000


def pools(work):
    sources = work / 'sources'
    wild, constructed = [], []
    if (sources / 'dfire.json').exists():
        wild += [r for r in read_json(sources / 'dfire.json')['records'] if not r['boxes']]
    for name in ('openimages.json', 'roboflow.json'):
        if (sources / name).exists():
            wild += read_json(sources / name)['records']
    for name in ('screens.json', 'own_frames.json'):
        if (sources / name).exists():
            constructed += read_json(sources / name)['records']
    fasdd = [r for r in read_json(work / 'manifests_fasdd' / 'train.json') if not r['boxes']]
    return wild, constructed, fasdd


def run(checkpoint, batch):
    work = work_dir()
    out = work / 'mine'
    wild, constructed, fasdd = pools(work)
    print(f'mining: wild {len(wild)}, constructed {len(constructed)}, FASDD train no-box {len(fasdd)}', flush=True)
    detections_path = out / 'v9_detections.json'
    if detections_path.exists():
        detections = read_json(detections_path)
    else:
        model = load_detector(checkpoint, V9_SHA256)
        everything = wild + constructed + fasdd
        found = detect_records(model, everything, batch=batch, floor=.2)
        detections = {r['image']: d for r, d in zip(everything, found)}
        write_json(detections, detections_path)
        del model
    clip = Clip()
    scorer = FireScorer(clip)
    items, decisions = [], {}
    # wild: crop-level CLIP on hits, image-level CLIP on everything
    for start in range(0, len(wild), 512):
        chunk = wild[start:start+512]
        pictures = [cv2.imread(r['image']) for r in chunk]
        image_fire, _ = scorer.score(clip.images(pictures))
        for r, picture, p_image in zip(chunk, pictures, image_fire):
            hits = detections[r['image']]
            if not hits:
                decisions[r['image']] = 'drop' if p_image >= .5 else 'negative'
                continue
            crop_fire, nearest = scorer.score(clip.images([crop(picture, h) for h in hits]))
            top = max(h[4] for h in hits)
            if crop_fire.max() >= .85:
                decisions[r['image']] = 'drop'
            elif crop_fire.max() < .15 and top < .5:
                decisions[r['image']] = 'hard_negative'
            else:
                decisions[r['image']] = 'review'
                best = int(np.argmax([h[4] for h in hits]))
                items.append(dict(image=r['image'], pool=r['source'], category=r['category'], box=hits[best][:4],
                                  score=hits[best][4], p_fire=float(crop_fire[best]), nearest=nearest[best],
                                  ambiguity=abs(float(crop_fire.max()) - .5)))
        print(f'CLIP check {min(start+512, len(wild))}/{len(wild)}', flush=True)
    for r in constructed:
        decisions[r['image']] = 'hard_negative' if detections[r['image']] else 'negative'
    for r in fasdd:
        hits = [h for h in detections[r['image']] if h[4] >= .4]
        if hits:
            best = max(hits, key=lambda h: h[4])
            items.append(dict(image=r['image'], pool='fasdd_train_nobox', category='fasdd', box=best[:4], score=best[4],
                              p_fire=-1., nearest='', ambiguity=-1.))
    # review budget: FASDD label checks first, then the most ambiguous wild hits; the rest are dropped (conservative)
    items.sort(key=lambda i: (i['pool'] != 'fasdd_train_nobox', i['ambiguity'], -i['score']))
    for item in items[MAX_REVIEW:]:
        if item['pool'] != 'fasdd_train_nobox':
            decisions[item['image']] = 'drop'
    items = items[:MAX_REVIEW]
    review = work / 'outbox' / 'step2_review'
    shutil.rmtree(review, ignore_errors=True)
    review.mkdir(parents=True)
    with open(review / 'review_index.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'sheet', 'pool', 'category', 'score', 'p_fire', 'clip_nearest', 'image', 'box'])
        for k, item in enumerate(items):
            item['id'] = f'R{k:05d}'
            writer.writerow([item['id'], f'sheet_{k // 36:03d}.jpg', item['pool'], item['category'], f'{item["score"]:.3f}',
                             f'{item["p_fire"]:.3f}', item['nearest'], item['image'], [round(v, 1) for v in item['box']]])
    for s in range(0, len(items), 36):
        contact_sheet([(i['image'], i['box'], f'{i["id"]} {i["score"]:.2f}') for i in items[s:s+36]],
                      review / f'sheet_{s // 36:03d}.jpg')
    (review / 'HOW_TO_REVIEW.txt').write_text(
        'Question per crop: is there a REAL physical flame inside the red box?\n'
        'Screens showing fire = not_fire. Glowing heater coils, lamps, sun, reflections, sparks = not_fire.\n'
        'Answer file: inbox/review_answers.csv with columns id,verdict (fire | not_fire | unsure).\n')
    counts = {}
    for v in decisions.values():
        counts[v] = counts.get(v, 0) + 1
    summary = dict(decisions_before_review=counts, review_items=len(items),
                   review_fasdd=sum(i['pool'] == 'fasdd_train_nobox' for i in items))
    write_json(summary, review / 'summary.json')
    write_json(dict(decisions=decisions, items=items), out / 'pending.json')
    archive = work / 'outbox' / 'step2_review.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(review.rglob('*')):
            z.write(path, path.relative_to(review.parent))
    print(summary)
    print(f'\nSTOP 1: send {archive} for review. Then pull the answers and run: run.ps1 2b', flush=True)


def finalize(answers):
    work = work_dir()
    pending = read_json(work / 'mine' / 'pending.json')
    verdicts = {row['id'].strip(): row['verdict'].strip().lower()
                for row in csv.DictReader(open(answers, newline='', encoding='utf-8-sig'))}
    missing = [i['id'] for i in pending['items'] if i['id'] not in verdicts]
    if missing:
        raise RuntimeError(f'{len(missing)} review ids have no verdict, e.g. {missing[:5]}')
    if set(verdicts.values()) - {'fire', 'not_fire', 'unsure'}:
        raise RuntimeError(f'unknown verdicts: {set(verdicts.values())}')
    decisions = pending['decisions']
    exclusions = list(read_json(PACKAGE / 'refs' / 'fasdd_exclusions.json')['train_images'])
    for item in pending['items']:
        verdict = verdicts[item['id']]
        if item['pool'] == 'fasdd_train_nobox':
            if verdict == 'fire':
                exclusions.append(Path(item['image']).name)
        else:
            decisions[item['image']] = {'not_fire': 'hard_negative'}.get(verdict, 'drop')
    counts = {}
    for v in decisions.values():
        counts[v] = counts.get(v, 0) + 1
    write_json(dict(decisions=decisions, fasdd_train_exclusions=sorted(set(exclusions)), counts=counts,
                    verdict_counts={v: list(verdicts.values()).count(v) for v in set(verdicts.values())},
                    answers_file=str(answers)), work / 'mine' / 'decisions.json')
    print('final decisions', counts, '| FASDD train exclusions', len(set(exclusions)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['run', 'finalize'])
    p.add_argument('--checkpoint')
    p.add_argument('--answers', default=str(PACKAGE / 'inbox' / 'review_answers.csv'))
    p.add_argument('--batch', type=int, default=32)
    a = p.parse_args()
    run(a.checkpoint, a.batch) if a.action == 'run' else finalize(a.answers)
