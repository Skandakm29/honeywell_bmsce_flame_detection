"""Automatic COCO-to-flame manifests and deterministic training augmentation."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(root, output, monitor=2000, workers=8, expected_annotations=None):
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'policy': 'fire/flame only; retain smoke/background negatives; preserve supplied splits',
              'quarantined': [], 'annotations_sha256': {}, 'splits': {}}
    if expected_annotations:
        for split, expected in expected_annotations.items():
            if sha256(root/'annotations'/f'{split}.json') != expected:
                raise ValueError(f'{split} annotation checksum differs from the supplied dataset; stop rather than silently change splits')
    splits = {}
    for split in ('train', 'val', 'test'):
        source = root / 'annotations' / f'{split}.json'
        coco = json.loads(source.read_text())
        report['annotations_sha256'][split] = sha256(source)
        categories = {c['id']: c['name'].strip().lower() for c in coco['categories']}
        fire = [k for k, v in categories.items() if v in ('fire', 'flame')]
        if len(fire) != 1 or set(categories.values()) - {'fire', 'flame', 'smoke'}:
            raise ValueError(f'Unexpected categories: {categories}')
        images = coco['images']
        if len({r['id'] for r in images}) != len(images):
            raise ValueError(f'Duplicate image IDs in {split}')
        known = {r['id'] for r in images}
        boxes = collections.defaultdict(list)
        for ann in coco['annotations']:
            if ann['image_id'] not in known or ann['category_id'] not in categories:
                raise ValueError(f'Orphan/unknown annotation in {split}: {ann["id"]}')
            if ann['category_id'] == fire[0]:
                if ann.get('iscrowd', 0):
                    raise ValueError('Crowd flame boxes need an explicit evaluation policy')
                boxes[ann['image_id']].append(ann['bbox'])

        def inspect(image):
            path = (root / 'images' / split / image['file_name']).resolve()
            if not path.is_relative_to(root / 'images' / split):
                return None, 'unsafe image path'
            # Match teammate's OpenCV decoding, including EXIF orientation.
            picture = cv2.imread(str(path))
            if picture is None or list(picture.shape[:2]) != [image['height'], image['width']]:
                return None, 'decode/dimension mismatch'
            converted = []
            for x, y, w, h in boxes[image['id']]:
                if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > image['width']+1 or y+h > image['height']+1:
                    return None, 'invalid bounding box'
                converted.append([x, y, min(x+w, image['width']), min(y+h, image['height'])])
            return dict(image=str(path), width=image['width'], height=image['height'],
                        boxes=converted, source_image_id=image['id'], sha256=sha256(path)), None

        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            checked = list(pool.map(inspect, images))
        records = []
        for image, (record, reason) in zip(images, checked):
            if reason:
                report['quarantined'].append(dict(split=split, image=image['file_name'], reason=reason))
            else:
                records.append(record)
        splits[split] = records
        report['splits'][split] = dict(images=len(records), boxes=sum(len(r['boxes']) for r in records),
                                      positive_images=sum(bool(r['boxes']) for r in records))
        print(split, report['splits'][split], flush=True)
    owners = collections.defaultdict(set)
    for split, records in splits.items():
        for r in records:
            owners[r['sha256']].add(split)
    report['cross_split_exact_duplicate_hashes'] = dict(collections.Counter(
        '+'.join(sorted(s)) for s in owners.values() if len(s) > 1))
    report['duplicates_policy'] = 'reported, not removed; matches teammate split policy; independence unproven'
    outputs = {'train': splits['train'], 'validation_full': splits['val'], 'test': splits['test'],
               'val': random.Random(42).sample(splits['val'], min(monitor, len(splits['val'])))}
    for name, records in outputs.items():
        (output / f'{name}.json').write_text(json.dumps(records))
    report['manifest_sha256'] = {name: sha256(output / f'{name}.json') for name in outputs}
    (output / 'prepare_report.json').write_text(json.dumps(report, indent=2))
    return report


def letterbox(image, boxes, size):
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    width, height = max(1, round(w * ratio)), max(1, round(h * ratio))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:height, :width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    scale = np.array([width/w, height/h, width/w, height/h], dtype=np.float32)
    return canvas, np.asarray(boxes, dtype=np.float32).reshape(-1, 4) * scale, scale


class Flames(Dataset):
    def __init__(self, records, size=640, augment=False, seed=42, mosaic=.5):
        self.records = records
        self.size, self.augment, self.seed, self.mosaic = size, augment, seed, mosaic
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def load(self, index):
        r = self.records[index]
        image = cv2.imread(r['image'])
        if image is None or image.shape[:2] != (r['height'], r['width']):
            raise ValueError(f'Image changed since preparation: {r["image"]}')
        return letterbox(image, r['boxes'], self.size)

    def __getitem__(self, index):
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        image, boxes, scale = self.load(index)
        if self.augment and rng.random() < self.mosaic:
            side = self.size
            canvas = np.full((2*side, 2*side, 3), 114, dtype=np.uint8)
            parts = []
            for k, idx in enumerate([index] + [rng.randrange(len(self)) for _ in range(3)]):
                tile, b, _ = self.load(idx)
                x, y = (k % 2)*side, (k // 2)*side
                canvas[y:y+side, x:x+side] = tile
                parts.append(b + np.array([x, y, x, y]))
            x, y = rng.randrange(side+1), rng.randrange(side+1)
            image = canvas[y:y+side, x:x+side].copy()
            boxes = np.concatenate(parts) - np.array([x, y, x, y])
            original = np.maximum(0, boxes[:, 2:]-boxes[:, :2]).prod(1)
            boxes = boxes.clip(0, side)
            wh = boxes[:, 2:]-boxes[:, :2]
            keep = (wh.min(1) >= 1) & (wh.prod(1) / np.maximum(original, 1e-6) >= .2)
            boxes = boxes[keep]
        if self.augment:
            if rng.random() < .5:
                image = image[:, ::-1].copy()
                boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-.01, .01)*180) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1]*rng.uniform(.7, 1.3), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2]*rng.uniform(.7, 1.3), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # TI YOLOv9 contract differs from YOLOX: RGB, float32 0..1.
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float()/255.
        target = torch.full((max(1, len(boxes)), 5), -1., dtype=torch.float32)
        target[:, 1:] = 0
        if len(boxes):
            target[:len(boxes), 0] = 0
            target[:len(boxes), 1:] = torch.from_numpy(boxes.astype(np.float32))
        return tensor, target, index, scale


def collate(batch):
    target = torch.full((len(batch), max(len(b[1]) for b in batch), 5), -1.)
    target[:, :, 1:] = 0
    for i, b in enumerate(batch):
        target[i, :len(b[1])] = b[1]
    return torch.stack([b[0] for b in batch]), target, [b[2] for b in batch], np.stack([b[3] for b in batch])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    expected = json.loads(Path(__file__).with_name('dataset_source.json').read_text())['annotation_sha256']
    prepare(a.root, a.output, workers=a.workers, expected_annotations=expected)
