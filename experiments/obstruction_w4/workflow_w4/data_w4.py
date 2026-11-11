"""Two-class training data (0 = flame, 1 = obstruction) on top of the V10/V12 pipeline.

Records keep the V9/V10 manifest format. One optional key is added:
  labels    list of class ids, parallel to `boxes`. Missing -> every box is a flame (all existing manifests).
  occluder  obstruction spec (occluders.py) for fixed validation images; `boxes`/`labels` then already contain the
            obstruction box and only the flames that stay visible, in ORIGINAL image pixels.

Pipeline (training): load + letterbox exactly as data.Flames -> with probability occluder.p draw a random
obstruction on the image content (never on the 114 padding), add its box (class 1) and drop flames it hides ->
mosaic -> optional V11/V12 additions (zoom-out, small-flame paste, camera degradation; data11.py, unchanged) ->
flip -> HSV. Without augmentation, a record with `occluder` is rendered exactly the same way every time.
"""
import random

import cv2
import numpy as np
import torch

import occluders as O
from data import Flames, letterbox

PAD = 114
FLAME, OBSTRUCTION = 0, 1


def labels_of(record):
    labels = record.get('labels')
    if labels is None:
        return [FLAME] * len(record['boxes'])
    if len(labels) != len(record['boxes']) or set(labels) - {FLAME, OBSTRUCTION}:
        raise ValueError(f'labels must be 0/1 and parallel to boxes: {record["image"]}')
    return list(labels)


def class_records(records, cls):
    """Same records with only the boxes of one class (for the unchanged 1-class metric code)."""
    return [dict(r, boxes=[b for b, l in zip(r['boxes'], labels_of(r)) if l == cls]) for r in records]


class Flames2(Flames):
    def __init__(self, records, size=640, augment=False, seed=42, mosaic=.5, augment_cfg=None, bank=None,
                 occluder_cfg=None):
        super().__init__(records, size, augment, seed, mosaic)
        self.augment_cfg, self.bank, self.occluder_cfg = augment_cfg, bank, occluder_cfg or {}

    def _donor(self, rng):
        """A different image of the same split, used as a defocused nearby object."""
        r = self.records[rng.randrange(len(self.records))]
        image = cv2.imread(r['image'])
        return None if image is None else cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)

    def load2(self, index, rng):
        """(canvas, boxes, labels, scale): letterboxed image and its boxes in canvas pixels."""
        r = self.records[index]
        image = cv2.imread(r['image'])
        if image is None or image.shape[:2] != (r['height'], r['width']):
            raise ValueError(f'Image changed since preparation: {r["image"]}')
        canvas, boxes, scale = letterbox(image, r['boxes'], self.size)
        labels = np.asarray(labels_of(r), np.int64)
        width, height = round(r['width'] * scale[0]), round(r['height'] * scale[1])
        spec = r.get('occluder')
        if spec is not None:                                    # fixed validation obstruction, labels already final
            donor = cv2.imread(spec['donor']) if spec.get('donor') else None
            canvas[:height, :width] = O.render(canvas[:height, :width], spec, donor)
            return canvas, boxes, labels, scale
        cfg = self.occluder_cfg
        if self.augment and cfg.get('p', 0) > 0 and rng.random() < cfg['p']:
            spec = O.sample_spec(rng, cfg)
            box, _, mask = O.label(spec, width, height, cfg.get('min_fraction', .03))
            if box is not None:
                donor = self._donor(rng) if spec['kind'] == 'blurred_scene' else None
                canvas[:height, :width] = O.render(canvas[:height, :width], spec, donor)
                keep = [i for i in O.visible_flames(boxes[labels == FLAME], mask, cfg.get('hide_if_covered', .5))]
                flame_idx = np.flatnonzero(labels == FLAME)[keep]
                other = np.flatnonzero(labels != FLAME)
                boxes = np.concatenate([boxes[flame_idx], boxes[other], np.float32([box])]).astype(np.float32)
                labels = np.concatenate([labels[flame_idx], labels[other], [OBSTRUCTION]]).astype(np.int64)
        return canvas, boxes, labels, scale

    def __getitem__(self, index):
        rng = random.Random(self.seed + self.epoch * 1000003 + index)            # same stream as data.Flames
        extra = random.Random((self.seed + self.epoch * 1000003 + index) ^ 0x0B57)  # week-4 stream
        image, boxes, labels, scale = self.load2(index, extra)
        if self.augment and rng.random() < self.mosaic:
            side = self.size
            canvas = np.full((2 * side, 2 * side, 3), PAD, dtype=np.uint8)
            parts, part_labels = [], []
            for k, idx in enumerate([index] + [rng.randrange(len(self)) for _ in range(3)]):
                tile, b, l, _ = (image, boxes, labels, scale) if k == 0 else self.load2(idx, extra)
                x, y = (k % 2) * side, (k // 2) * side
                canvas[y:y + side, x:x + side] = tile
                parts.append(b.reshape(-1, 4) + np.array([x, y, x, y]))
                part_labels.append(l)
            x, y = rng.randrange(side + 1), rng.randrange(side + 1)
            image = canvas[y:y + side, x:x + side].copy()
            boxes = np.concatenate(parts) - np.array([x, y, x, y])
            labels = np.concatenate(part_labels)
            original = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
            boxes = boxes.clip(0, side)
            wh = boxes[:, 2:] - boxes[:, :2]
            keep = (wh.min(1) >= 1) & (wh.prod(1) / np.maximum(original, 1e-6) >= .2)
            boxes, labels = boxes[keep], labels[keep]
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if self.augment and self.augment_cfg:                                  # V11/V12 additions, unchanged code
            from data11 import degrade, paste, zoom_out
            v11 = random.Random((self.seed + self.epoch * 1000003 + index) ^ 0x5F1A11)
            image, boxes = zoom_out(image, boxes, v11, self.augment_cfg)
            before = len(boxes)
            image, boxes = paste(np.ascontiguousarray(image), boxes, v11, self.augment_cfg, self.bank)
            labels = np.concatenate([labels, np.full(len(boxes) - before, FLAME, np.int64)])  # pasted = flames
            image = degrade(image, v11, self.augment_cfg)
        if self.augment:
            if rng.random() < .5:
                image = image[:, ::-1].copy()
                boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-.01, .01) * 180) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(.7, 1.3), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(.7, 1.3), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float() / 255.
        target = torch.full((max(1, len(boxes)), 5), -1., dtype=torch.float32)
        target[:, 1:] = 0
        if len(boxes):
            target[:len(boxes), 0] = torch.from_numpy(labels.astype(np.float32))
            target[:len(boxes), 1:] = torch.from_numpy(boxes.astype(np.float32))
        return tensor, target, index, scale


def factory(occluder_cfg, augment_cfg=None, bank_path=None):
    """Drop-in replacement for data.Flames inside train_v10.train (validation calls pass augment=False)."""
    bank = None
    if bank_path:
        from pathlib import Path
        from data11 import Bank
        bank = Bank(bank_path) if Path(bank_path).exists() else None

    def make(records, size=640, augment=False, seed=42, mosaic=.5):
        return Flames2(records, size, augment, seed, mosaic, augment_cfg=augment_cfg if augment else None,
                       bank=bank, occluder_cfg=occluder_cfg)
    return make


def synthetic_record(record, spec, w=None, h=None, hide_if_covered=.5, min_fraction=.03):
    """Fixed obstruction on a copy of `record` (original pixels). None if the obstruction would be too small."""
    w, h = w or record['width'], h or record['height']
    box, fraction, mask = O.label(spec, w, h, min_fraction)
    if box is None:
        return None
    boxes = [b for b, l in zip(record['boxes'], labels_of(record)) if l == FLAME]
    kept = [boxes[i] for i in O.visible_flames(np.float32(boxes).reshape(-1, 4), mask, hide_if_covered)]
    return dict(record, boxes=kept + [box], labels=[FLAME] * len(kept) + [OBSTRUCTION], occluder=spec,
                blind_fraction=fraction, hidden_flames=len(boxes) - len(kept),
                source_image_id=f'{record.get("source_image_id", record["image"])}__obstruction')
