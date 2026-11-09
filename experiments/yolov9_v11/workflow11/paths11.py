"""Locations and constants shared by the V11 steps. V10 code/data are read, never written."""
import os
from pathlib import Path

import numpy as np

from common import work_dir

PACKAGE11 = Path(__file__).resolve().parent.parent
INBOX = PACKAGE11 / 'inbox'
WORK = work_dir()                                   # D:\Honeywell-V10-work (shared; V11 writes *_v11 only)
OUTBOX = WORK / 'outbox'
V10_FROZEN = Path(os.environ.get('V11_V10_CHECKPOINT', WORK / 'FROZEN_V10' / 'selected.pth'))
V10_FROZEN_SHA = 'd046dab613faa72ec650d37dc40d0740ff50ac09df4eb97919d10cfd532b721d'
V9_SHA = '48d8e654ac1e3381e122a52cdfca100d1840e8cf6db0294ac882b52ab1c4a51f'
COCO_WEIGHTS = WORK / 'weights' / 'yolov9_s_ti_lite_coco.pth'
EVAL_MANIFESTS = ('val.json', 'validation_full.json', 'hn_val.json', 'test.json')


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)
