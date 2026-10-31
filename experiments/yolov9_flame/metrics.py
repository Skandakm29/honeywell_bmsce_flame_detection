"""COCO bbox metrics and explicitly defined detection precision/recall/F1."""
import contextlib
import io
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

from data import collate


def postprocess(output, record, scale, score_threshold=.001):
    output = output[output[:, 4] >= score_threshold].float()
    if not len(output):
        return []
    output = output.clone()
    output[:, :4] /= torch.as_tensor(scale, device=output.device)
    output[:, [0, 2]] = output[:, [0, 2]].clamp(0, record['width'])
    output[:, [1, 3]] = output[:, [1, 3]].clamp(0, record['height'])
    output = output[(output[:, 2] > output[:, 0]) & (output[:, 3] > output[:, 1])]
    selected = nms(output[:, :4], output[:, 4], .65)[:100]
    return output[selected].cpu().tolist()


def operating_metrics(records, predictions, confidence=.4):
    grouped = [[] for _ in records]
    for p in predictions:
        grouped[p['image_id']].append(p)
    tp = fp = fn = 0
    for r, candidates in zip(records, grouped):
        matched = set()
        for p in sorted(candidates, key=lambda p: -p['score']):
            if p['score'] < confidence:
                continue
            x, y, w, h = p['bbox']
            best, best_id = 0., None
            for i, (a, b, c, d) in enumerate(r['boxes']):
                if i in matched:
                    continue
                intersection = max(0., min(x+w, c)-max(x, a))*max(0., min(y+h, d)-max(y, b))
                union = w*h + (c-a)*(d-b)-intersection
                overlap = intersection / union if union > 0 else 0.
                if overlap > best:
                    best, best_id = overlap, i
            if best >= .5:
                tp += 1
                matched.add(best_id)
            else:
                fp += 1
        fn += len(r['boxes']) - len(matched)
    precision, recall = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
    return dict(confidence=confidence, matching_iou=.5, TP=tp, FP=fp, FN=fn,
                precision=precision, recall=recall,
                F1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def coco_metrics(records, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    annotations = []
    for i, r in enumerate(records):
        for x, y, x2, y2 in r['boxes']:
            annotations.append(dict(id=len(annotations)+1, image_id=i, category_id=1,
                                    bbox=[x, y, x2-x, y2-y], area=(x2-x)*(y2-y), iscrowd=0))
    gt.dataset = dict(info={}, images=[dict(id=i, width=r['width'], height=r['height']) for i, r in enumerate(records)],
                      annotations=annotations, categories=[dict(id=1, name='flame')])
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        if predictions:
            detections = gt.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {**gt.dataset, 'annotations': []}
            detections.createIndex()
        evaluator = COCOeval(gt, detections, 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ('AP50_95', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')
    return {k: float(v) if v >= 0 else None for k, v in zip(names, evaluator.stats)}


@torch.inference_mode()
def evaluate(model, dataset, batch=8, workers=2):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=collate,
                        num_workers=workers, pin_memory=device.type == 'cuda')
    predictions = []
    started = time.perf_counter()
    for tensors, _, indices, scales in loader:
        outputs = model(tensors.to(device, non_blocking=True))
        if not torch.isfinite(outputs).all():
            raise FloatingPointError('Nonfinite evaluation output')
        for output, index, scale in zip(outputs, indices, scales):
            for x, y, x2, y2, score in postprocess(output, dataset.records[index], scale):
                predictions.append(dict(image_id=index, category_id=1, bbox=[x, y, x2-x, y2-y], score=score))
    metrics = {'coco': coco_metrics(dataset.records, predictions),
               'at_confidence_0.40': operating_metrics(dataset.records, predictions),
               'images': len(dataset), 'boxes': sum(len(r['boxes']) for r in dataset.records),
               'evaluation_seconds': time.perf_counter()-started,
               'definitions': {'AP': 'COCO bbox AP, IoU .50:.95, score floor .001, NMS .65, maxDets 100',
                               'precision_recall_F1': 'score-ordered one-to-one box matching, IoU .50, confidence .40',
                               'area_groups': 'COCO original-image area thresholds; absent groups reported as null',
                               'FP': 'false-positive boxes, not video alarm events'}}
    return metrics, predictions
