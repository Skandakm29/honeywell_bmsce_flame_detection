"""V11 training data: V10's Flames pipeline + three additions aimed at small / far flames.

Unchanged from V9/V10 (same code path, same random stream): letterbox to 640, mosaic (p .5), horizontal
flip, small HSV jitter. Without augmentation (validation, hn_val, benchmarks) Flames11 IS Flames.

Added (own random stream, so the base stream is identical to V10):
  1. zoom-out   p .5: shrink the whole 640 canvas by s ~ log-uniform[.25, 1] and place it at a random spot on
                a grey (114) canvas, i.e. the same padding the model already sees at inference. s is raised so
                no labelled flame becomes smaller than 4 px (no unlabelled-looking tiny flames are created).
  2. small-flame paste  p .25: 1-3 flame crops from a TRAIN-only bank, resized to 6-32 px, blended with a
                brightness-based soft mask, never on an existing box or on padding; the pasted box is labelled.
  3. camera degradations, each independent: down/up resample p .15, Gaussian blur p .15, sensor noise p .15,
                JPEG q 30-85 p .25 (phone / CCTV / WhatsApp-like). Applied after 1-2 so pasted flames match.
All parameters live in config_v11.json["augment_v11"]; nothing here is tuned on validation.
"""
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from data import Flames

PAD = 114


class Bank:
    """Train-only flame crops, (N, S, S, 3) uint8 padded + (N, 2) real h, w. Loaded lazily per worker (mmap)."""

    def __init__(self, path):
        self.path, self._crops, self._hw = str(path), None, None

    def _load(self):
        if self._crops is None:
            data = np.load(self.path, mmap_mode='r')
            self._crops, self._hw = data['crops'], data['hw']

    def __len__(self):
        self._load()
        return len(self._hw)

    def get(self, k):
        self._load()
        h, w = self._hw[k]
        return np.array(self._crops[k, :h, :w])

    def __getstate__(self):  # never pickle the arrays into DataLoader workers
        return {'path': self.path, '_crops': None, '_hw': None}


def zoom_out(image, boxes, rng, cfg):
    if rng.random() >= cfg['zoom_p']:
        return image, boxes
    size = image.shape[0]
    s = math.exp(rng.uniform(math.log(cfg['zoom_min']), 0.))
    if len(boxes):
        smallest = float((boxes[:, 2:] - boxes[:, :2]).max(1).min())
        if smallest > 0:
            s = max(s, cfg['min_box_px'] / smallest)
    if s >= .999:
        return image, boxes
    n = max(1, round(size * s))
    small = cv2.resize(image, (n, n), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(image, PAD)
    x, y = rng.randrange(size - n + 1), rng.randrange(size - n + 1)
    canvas[y:y + n, x:x + n] = small
    factor = n / size
    return canvas, (boxes * factor + np.array([x, y, x, y], np.float32)).astype(np.float32)


def soft_mask(crop):
    v = crop.max(2).astype(np.float32) / 255.
    t = np.percentile(v, 40)
    alpha = np.clip((v - t) / max(1e-3, .25), 0, 1)
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ellipse = 1 - np.clip(((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) / 2) / (h / 2)) ** 2, 0, 1)
    return cv2.GaussianBlur(alpha * np.sqrt(ellipse), (0, 0), max(.5, min(h, w) / 12))


def paste(image, boxes, rng, cfg, bank):
    if bank is None or rng.random() >= cfg['paste_p']:
        return image, boxes
    size = image.shape[0]
    added = []
    for _ in range(rng.randint(1, cfg['paste_max'])):
        crop = bank.get(rng.randrange(len(bank)))
        side = round(math.exp(rng.uniform(math.log(cfg['paste_min_px']), math.log(cfg['paste_max_px']))))
        scale = side / max(crop.shape[:2])
        w, h = max(2, round(crop.shape[1] * scale)), max(2, round(crop.shape[0] * scale))
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        alpha = soft_mask(crop)
        ys, xs = np.where(alpha > .5)
        if not len(xs):
            continue
        for _ in range(10):
            x, y = rng.randrange(size - w + 1), rng.randrange(size - h + 1)
            box = np.array([x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1], np.float32)
            region = image[y:y + h, x:x + w]
            on_pad = (region == PAD).all(2).mean() > .3
            others = np.concatenate([boxes.reshape(-1, 4), np.asarray(added, np.float32).reshape(-1, 4)])
            overlap = len(others) and ((np.minimum(others[:, 2], x + w) > np.maximum(others[:, 0], x)) &
                                       (np.minimum(others[:, 3], y + h) > np.maximum(others[:, 1], y))).any()
            if on_pad or overlap:
                continue
            a = alpha[:, :, None]
            image[y:y + h, x:x + w] = (crop * a + region * (1 - a)).astype(np.uint8)
            added.append(box)
            break
    if added:
        boxes = np.concatenate([boxes.reshape(-1, 4), np.asarray(added, np.float32)]).astype(np.float32)
    return image, boxes


def degrade(image, rng, cfg):
    size = image.shape[0]
    if rng.random() < cfg['resample_p']:
        f = rng.uniform(.4, .8)
        n = max(8, round(size * f))
        image = cv2.resize(cv2.resize(image, (n, n), interpolation=cv2.INTER_AREA), (size, size),
                           interpolation=cv2.INTER_LINEAR)
    if rng.random() < cfg['blur_p']:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(.3, 1.2))
    if rng.random() < cfg['noise_p']:
        noise = np.random.default_rng(rng.randrange(2 ** 31)).normal(0, rng.uniform(2, 8), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < cfg['jpeg_p']:
        ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 85)])
        if ok:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


class Flames11(Flames):
    def __init__(self, records, size=640, augment=False, seed=42, mosaic=.5, cfg=None, bank=None):
        super().__init__(records, size, augment, seed, mosaic)
        self.cfg, self.bank = cfg, bank

    def __getitem__(self, index):
        if not self.augment or self.cfg is None:
            return super().__getitem__(index)
        rng = random.Random(self.seed + self.epoch * 1000003 + index)          # V10 stream (unchanged order)
        extra = random.Random((self.seed + self.epoch * 1000003 + index) ^ 0x5F1A11)  # V11 stream
        image, boxes, scale = self.load(index)
        if rng.random() < self.mosaic:                                         # identical to data.Flames
            side = self.size
            canvas = np.full((2 * side, 2 * side, 3), PAD, dtype=np.uint8)
            parts = []
            for k, idx in enumerate([index] + [rng.randrange(len(self)) for _ in range(3)]):
                tile, b, _ = self.load(idx)
                x, y = (k % 2) * side, (k // 2) * side
                canvas[y:y + side, x:x + side] = tile
                parts.append(b + np.array([x, y, x, y]))
            x, y = rng.randrange(side + 1), rng.randrange(side + 1)
            image = canvas[y:y + side, x:x + side].copy()
            boxes = np.concatenate(parts) - np.array([x, y, x, y])
            original = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
            boxes = boxes.clip(0, side)
            wh = boxes[:, 2:] - boxes[:, :2]
            keep = (wh.min(1) >= 1) & (wh.prod(1) / np.maximum(original, 1e-6) >= .2)
            boxes = boxes[keep]
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        image, boxes = zoom_out(image, boxes, extra, self.cfg)                 # V11
        image, boxes = paste(np.ascontiguousarray(image), boxes, extra, self.cfg, self.bank)  # V11
        image = degrade(image, extra, self.cfg)                                # V11
        if rng.random() < .5:                                                  # identical to data.Flames
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
            target[:len(boxes), 0] = 0
            target[:len(boxes), 1:] = torch.from_numpy(boxes.astype(np.float32))
        return tensor, target, index, scale


def factory(cfg, bank_path):
    """Drop-in replacement for data.Flames inside train_v10.train (validation calls pass augment=False)."""
    bank = Bank(bank_path) if bank_path and Path(bank_path).exists() else None

    def make(records, size=640, augment=False, seed=42, mosaic=.5):
        return Flames11(records, size, augment, seed, mosaic, cfg=cfg if augment else None, bank=bank)
    return make
