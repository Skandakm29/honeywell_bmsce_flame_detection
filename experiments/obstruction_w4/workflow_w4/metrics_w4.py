"""Two-class evaluation. Flame numbers use the UNCHANGED V10 definitions (metrics.coco_metrics / operating_metrics)
on the clean images, so they are directly comparable with V10/V12. Obstruction adds:
  AP / precision / recall (IoU .5, confidence .40) on the images with an obstruction,
  false-obstruction image rate on clean images (any obstruction box >= .40),
  blind-share error: |predicted blind share - true blind share|, where the predicted share is the union area of the
  obstruction boxes >= .40 divided by the image area (the same number the live viewer shows).
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

from data import collate
from data_w4 import FLAME, OBSTRUCTION, class_records
from metrics import coco_metrics, operating_metrics, postprocess

CLASSES = ('flame', 'obstruction')


def postprocess2(output, record, scale, score_threshold=.001, nms_iou=.65, max_det=100):
    """[N, 4 + C] raw output -> [[x1, y1, x2, y2, score, class], ...] in original pixels, NMS per class."""
    output = output.float()
    rows = []
    sx = torch.as_tensor(scale, device=output.device, dtype=torch.float32)
    for c in range(output.shape[1] - 4):
        score = output[:, 4 + c]
        keep = score >= score_threshold
        if not keep.any():
            continue
        boxes = output[keep, :4] / sx
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, record['width'])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, record['height'])
        s = score[keep]
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, s = boxes[valid], s[valid]
        selected = nms(boxes, s, nms_iou)[:max_det]
        for b, v in zip(boxes[selected].tolist(), s[selected].tolist()):
            rows.append([*b, v, c])
    rows.sort(key=lambda r: -r[4])
    return rows[:max_det]


def postprocess_flame(output, record, scale, score_threshold=.001):
    """The V10 postprocess on the flame column only (so V10 hard-negative counting is unchanged)."""
    return postprocess(output[:, :5], record, scale, score_threshold)


def blind_share(boxes, width, height, grid=160):
    """Union area of xyxy boxes as a share of the image, rasterised on a grid (exact enough for reporting)."""
    if not len(boxes):
        return 0.
    gw, gh = grid, max(1, round(grid * height / width))
    mask = np.zeros((gh, gw), bool)
    for x1, y1, x2, y2 in boxes:
        a, b = int(np.floor(x1 / width * gw)), int(np.floor(y1 / height * gh))
        c, d = int(np.ceil(x2 / width * gw)), int(np.ceil(y2 / height * gh))
        mask[max(0, b):min(gh, d), max(0, a):min(gw, c)] = True
    return float(mask.mean())


def obstruction_metrics(records, detections, confidence=.40):
    """`detections`: per record, rows from postprocess2. Splits clean vs obstructed images."""
    occluded = [i for i, r in enumerate(records) if OBSTRUCTION in r.get('labels', [])]
    clean = [i for i, r in enumerate(records) if OBSTRUCTION not in r.get('labels', [])]
    preds = [dict(image_id=k, category_id=1, bbox=[x, y, x2 - x, y2 - y], score=s)
             for k, i in enumerate(occluded) for x, y, x2, y2, s, c in detections[i] if c == OBSTRUCTION]
    subset = class_records([records[i] for i in occluded], OBSTRUCTION)
    flagged = [any(d[5] == OBSTRUCTION and d[4] >= confidence for d in detections[i]) for i in clean]
    errors = []
    for i in occluded:
        r = records[i]
        boxes = [d[:4] for d in detections[i] if d[5] == OBSTRUCTION and d[4] >= confidence]
        errors.append(abs(blind_share(boxes, r['width'], r['height']) - r.get('blind_fraction', 0.)))
    hits = [any(d[5] == OBSTRUCTION and d[4] >= confidence for d in detections[i]) for i in occluded]
    by_kind = {}
    for k, i in enumerate(occluded):
        spec = records[i].get('occluder') or {}
        key = f'{spec.get("kind", "real")}/{spec.get("shape", "real")}'
        s = by_kind.setdefault(key, [0, 0])
        s[0] += 1
        s[1] += hits[k]
    return dict(
        obstructed_images=len(occluded), clean_images=len(clean),
        coco=coco_metrics(subset, preds) if subset else None,
        at_confidence=operating_metrics(subset, preds, confidence) if subset else None,
        image_detection_rate=float(np.mean(hits)) if hits else None,
        false_obstruction_image_rate=float(np.mean(flagged)) if flagged else None,
        blind_share_mean_abs_error=float(np.mean(errors)) if errors else None,
        detected_by_kind={k: dict(images=v[0], detected=v[1]) for k, v in sorted(by_kind.items())},
        confidence=confidence)


@torch.inference_mode()
def detect(model, dataset, batch=8, workers=2):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=collate,
                        num_workers=workers, pin_memory=device.type == 'cuda')
    detections = [None] * len(dataset)
    for tensors, _, indices, scales in loader:
        outputs = model(tensors.to(device, non_blocking=True))
        if not torch.isfinite(outputs).all():
            raise FloatingPointError('Nonfinite evaluation output')
        for output, index, scale in zip(outputs, indices, scales):
            detections[index] = postprocess2(output, dataset.records[index], scale)
    return detections


def evaluate2(model, dataset, batch=8, workers=2):
    """Drop-in for metrics.evaluate inside train_v10: metrics['coco'] = FLAME on clean images (V10 definition)."""
    started = time.perf_counter()
    detections = detect(model, dataset, batch, workers)
    records = dataset.records
    clean = [i for i, r in enumerate(records) if 'occluder' not in r and OBSTRUCTION not in r.get('labels', [])]
    flame_records = class_records([records[i] for i in clean], FLAME)
    flame_preds = [dict(image_id=k, category_id=1, bbox=[x, y, x2 - x, y2 - y], score=s)
                   for k, i in enumerate(clean) for x, y, x2, y2, s, c in detections[i] if c == FLAME]
    clean_set = set(clean)
    occluded = [i for i in range(len(records)) if i not in clean_set]
    occ_records = class_records([records[i] for i in occluded], FLAME)
    occ_preds = [dict(image_id=k, category_id=1, bbox=[x, y, x2 - x, y2 - y], score=s)
                 for k, i in enumerate(occluded) for x, y, x2, y2, s, c in detections[i] if c == FLAME]
    metrics = {'coco': coco_metrics(flame_records, flame_preds),
               'at_confidence_0.40': operating_metrics(flame_records, flame_preds),
               'images': len(clean), 'boxes': sum(len(r['boxes']) for r in flame_records),
               'flame_on_obstructed_images': dict(at_confidence_0_40=operating_metrics(occ_records, occ_preds),
                                                 images=len(occluded)) if occluded else None,
               'obstruction': obstruction_metrics(records, detections),
               'evaluation_seconds': time.perf_counter() - started,
               'definitions': {'coco / at_confidence_0.40': 'FLAME on clean images only, V10 definitions',
                               'obstruction': 'see metrics_w4.obstruction_metrics'}}
    predictions = [dict(image_id=i, category_id=int(c) + 1, bbox=[x, y, x2 - x, y2 - y], score=s)
                   for i, ds in enumerate(detections) for x, y, x2, y2, s, c in ds]
    return metrics, predictions


@torch.inference_mode()
def negative_false_positives2(model, dataset, batch=8, workers=2):
    """Drop-in for train_v10.negative_false_positives: the V10 flame counts, plus obstruction counts on the same
    no-flame images (lamps, screens, sunsets ...: none of them is an obstruction)."""
    detections = detect(model, dataset, batch, workers)
    out = {}
    for name, cls in (('flame', FLAME), ('obstruction', OBSTRUCTION)):
        counts = {}
        for key, threshold in (('0.20', .2), ('0.40', .4)):
            n = [sum(d[5] == cls and d[4] >= threshold for d in ds) for ds in detections]
            counts[key] = dict(images_with_fp=int(sum(x > 0 for x in n)), fp_boxes=int(sum(n)),
                               fp_image_rate=sum(x > 0 for x in n) / max(1, len(dataset)))
        out[name] = counts
    return out['flame'] | {'images': len(dataset), 'obstruction': out['obstruction']}
