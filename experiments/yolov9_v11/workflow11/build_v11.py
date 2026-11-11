"""Step 2: manifests_v11 + flame bank + augmentation preview.

train.json   = manifests_v10/train.json (same records, same repeats) with the step-1 label fixes applied
               (+ optional extra small-flame frames from TRAINING locations, x3, from --extra/extra.json).
val.json, validation_full.json, hn_val.json, test.json = byte-identical copies of manifests_v10 (fair comparison).
flame_bank.npz = up to 6,000 flame crops (TRAIN boxes only, longest side >= 24 px, <= 25% of the image,
               from images with no label-fix candidate), 10% margin, longest side 48 px.
Preview: 48 augmented training samples with boxes + box-size statistics before/after augmentation.
"""
import argparse
import hashlib
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from common import contact_sheet, read_json, write_json
from paths11 import EVAL_MANIFESTS, INBOX, OUTBOX, WORK
from data11 import factory

BANK_SIDE = 48


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_mode():
    check = INBOX / 'labelfix_spotcheck.json'
    if not check.exists():
        raise RuntimeError('inbox/labelfix_spotcheck.json missing: do the step-1 spot check first (START_HERE.md)')
    c = read_json(check)
    total, correct = int(c['add_box_total']), int(c['add_box_real_flame'])
    if total <= 0:
        return 'drop', c
    return ('add' if correct / total >= .80 else 'drop'), c


def build_bank(records, flagged, out, limit=6000, seed=0):
    pool = []
    for r in records:
        if r['image'] in flagged or not r['boxes']:
            continue
        for b in r['boxes']:
            w, h = b[2] - b[0], b[3] - b[1]
            if max(w, h) >= 24 and w * h <= .25 * r['width'] * r['height']:
                pool.append((r['image'], b))
    random.Random(seed).shuffle(pool)
    crops = np.zeros((min(limit, len(pool)), BANK_SIDE, BANK_SIDE, 3), np.uint8)
    hw = np.zeros((len(crops), 2), np.int32)
    n = 0
    for image_path, (x1, y1, x2, y2) in pool:
        if n == len(crops):
            break
        image = cv2.imread(image_path)
        if image is None:
            continue
        mx, my = .1 * (x2 - x1), .1 * (y2 - y1)
        c = image[int(max(0, y1 - my)):int(min(image.shape[0], y2 + my)), int(max(0, x1 - mx)):int(min(image.shape[1], x2 + mx))]
        if c.size == 0:
            continue
        s = BANK_SIDE / max(c.shape[:2])
        c = cv2.resize(c, (max(2, round(c.shape[1] * s)), max(2, round(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
        crops[n, :c.shape[0], :c.shape[1]] = c
        hw[n] = c.shape[:2]
        n += 1
    np.savez(out, crops=crops[:n], hw=hw[:n])
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extra', action='append', default=[], help='folder(s) with extra.json: records (image,width,height,boxes,location,split=train)')
    a = p.parse_args()
    src, out = WORK / 'manifests_v10', WORK / 'manifests_v11'
    out.mkdir(parents=True, exist_ok=True)
    mode, check = label_mode()
    candidates = read_json(WORK / 'labelfix_v11' / 'candidates.json')
    drop = {c['image'] for c in candidates if c['decision'] == 'drop_image' or (mode == 'drop' and c['decision'] == 'add_box')}
    add = {}
    if mode == 'add':
        for c in candidates:
            if c['decision'] == 'add_box' and c['image'] not in drop:
                add.setdefault(c['image'], []).append(c['box'])
    train = read_json(src / 'train.json')
    new = []
    for r in train:
        if r['image'] in drop:
            continue
        if r['image'] in add:
            r = dict(r, boxes=r['boxes'] + add[r['image']], label_fix='added_v10_box')
        new.append(r)
    extra = []
    for folder in a.extra:
        for r in read_json(Path(folder) / 'extra.json'):
            if r.get('split') != 'train':
                raise RuntimeError(f'extra record not from a training location: {r["image"]}')
            extra += [dict(r, source='own_v11', category=r.get('category', 'own_v11_small_flame'))] * 3
    new += extra
    eval_images = set()
    for name in EVAL_MANIFESTS:
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
            assert sha(src / name) == sha(out / name)
            if name != 'val.json':
                eval_images |= {r['image'] for r in read_json(out / name)}
    leak = eval_images & {r['image'] for r in new}
    if leak:
        raise RuntimeError(f'{len(leak)} evaluation images in V11 train, e.g. {sorted(leak)[:3]}')
    write_json(new, out / 'train.json')
    flagged = {c['image'] for c in candidates}
    bank_n = build_bank(new, flagged, out / 'flame_bank.npz')
    positives = sum(bool(r['boxes']) for r in new)
    summary = dict(label_mode=mode, spotcheck=check, v10_train_records=len(train), v11_train_records=len(new),
                   images_dropped=len(drop), images_with_added_boxes=len(add), boxes_added=sum(map(len, add.values())),
                   extra_records=len(extra), positive_share=positives / len(new), flame_bank_crops=bank_n,
                   eval_manifests_sha256={n: sha(out / n) for n in EVAL_MANIFESTS if (out / n).exists()},
                   train_sha256=sha(out / 'train.json'))
    write_json(summary, out / 'build_summary.json')
    print(summary, flush=True)

    # augmentation preview and sanity statistics (epoch 0, mosaic on)
    cfg = read_json(Path(__file__).with_name('config_v11.json'))
    make = factory(cfg['augment_v11'], out / 'flame_bank.npz')
    ds = make(new, 640, augment=True, seed=cfg['seed'], mosaic=cfg['mosaic_probability'])
    rng = random.Random(1)
    idx = rng.sample(range(len(new)), min(2000, len(new)))
    before, after, items = [], [], []
    plain = make(new, 640)
    for k, i in enumerate(idx):
        tensor, target, _, _ = ds[i]
        boxes = target[target[:, 0] == 0, 1:].numpy()
        if len(boxes):
            assert np.isfinite(boxes).all() and (boxes >= -1e-3).all() and (boxes <= 640 + 1e-3).all()
            assert ((boxes[:, 2:] - boxes[:, :2]) >= .99).all()
            after += list((boxes[:, 2:] - boxes[:, :2]).max(1))
        pb = plain[i][1]
        pb = pb[pb[:, 0] == 0, 1:].numpy()
        before += list((pb[:, 2:] - pb[:, :2]).max(1)) if len(pb) else []
        if k < 48:
            image = (tensor.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
            for b in boxes:
                cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
            items.append((image, None, f'#{i} boxes {len(boxes)}'))
    bins = [0, 8, 16, 32, 64, 1e9]
    hist = lambda v: {f'{bins[j]}-{bins[j + 1]:g}px': int(((np.array(v) >= bins[j]) & (np.array(v) < bins[j + 1])).sum())
                      for j in range(len(bins) - 1)}
    preview = dict(samples=len(idx), boxes_without_aug=hist(before), boxes_with_v11_aug=hist(after))
    write_json(preview, out / 'augment_preview.json')
    print('box longest side at 640 input', preview, flush=True)
    sheet = out / 'augment_preview.jpg'
    contact_sheet(items, sheet, cols=8, tile=320)
    OUTBOX.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTBOX / 'v11_step2_build.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in (out / 'build_summary.json', out / 'augment_preview.json', sheet):
            z.write(path, path.name)


if __name__ == '__main__':
    main()
