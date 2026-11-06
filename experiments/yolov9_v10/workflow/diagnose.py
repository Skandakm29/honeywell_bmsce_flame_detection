"""Why does the detector fire? Run after bench.py for the same model name.

1 False-positive map: FP crops (>= .20) from hn_val and FASDD-val no-flame images, CLIP-embedded, k-means
  clustered, each cluster named by its most common closest nuisance prompt; one contact sheet per cluster.
2 D-RISE (black-box saliency per detection, Petsiuk et al. 2021) for the top FPs and TPs.
3 Perturbation test: confidence change under grayscale, hue +-30 deg, blur, and occluding the box.
  Large drop under grayscale/hue = colour reliance; drop only when occluded = the box content itself.
"""
import argparse
import json

import cv2
import numpy as np
import torch

from common import Clip, FireScorer, contact_sheet, crop, frame_tensor, load_detector, read_json, work_dir, write_json
from temporal import iou


def matched_score(model, image, box, min_iou=.3):
    from metrics import postprocess
    tensor, scale = frame_tensor(image)
    with torch.inference_mode():
        output = model(tensor.to(next(model.parameters()).device))[0]
    dets = postprocess(output, dict(width=image.shape[1], height=image.shape[0]), scale, .01)
    return max([d[4] for d in dets if iou(d, box) >= min_iou] or [0.])


def perturb(image, box):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    shifted = lambda d: cv2.cvtColor(np.dstack([(hsv[:, :, 0].astype(int) + d) % 180, hsv[:, :, 1], hsv[:, :, 2]]).astype(np.uint8),
                                     cv2.COLOR_HSV2BGR)
    occluded = image.copy()
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    occluded[y1:y2, x1:x2] = 114
    return {'grayscale': cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
            'hue_plus30': shifted(15), 'hue_minus30': shifted(-15),
            'blur_sigma3': cv2.GaussianBlur(image, (0, 0), 3), 'occlude_box': occluded}


def drise(model, image, box, masks=400, grid=16, p=.5, batch=16, seed=0):
    from metrics import postprocess
    rng = np.random.default_rng(seed)
    tensor, scale = frame_tensor(image)
    dev = next(model.parameters()).device
    size = tensor.shape[-1]
    cell = int(np.ceil(size / grid))
    saliency = torch.zeros(size, size)
    record = dict(width=image.shape[1], height=image.shape[0])
    for start in range(0, masks, batch):
        n = min(batch, masks - start)
        small = torch.from_numpy((rng.random((n, 1, grid + 1, grid + 1)) < p).astype(np.float32))
        up = torch.nn.functional.interpolate(small, size=((grid + 1) * cell,) * 2, mode='bilinear', align_corners=False)
        shifts = rng.integers(0, cell, (n, 2))
        m = torch.stack([up[i, 0, sy:sy+size, sx:sx+size] for i, (sx, sy) in enumerate(shifts)])
        x = tensor * m[:, None] + (1 - m[:, None]) * (114 / 255)
        with torch.inference_mode():
            outputs = model(x.to(dev))
        for i, output in enumerate(outputs):
            dets = postprocess(output, record, scale, .01)
            weight = max([iou(d, box) * d[4] for d in dets] or [0.])
            saliency += weight * m[i]
    saliency /= masks * p
    h, w = round(image.shape[0] * scale[1]), round(image.shape[1] * scale[0])
    heat = cv2.resize(saliency[:h, :w].numpy(), (image.shape[1], image.shape[0]))
    heat = (255 * (heat - heat.min()) / (np.ptp(heat) + 1e-9)).astype(np.uint8)
    overlay = cv2.addWeighted(image, .55, cv2.applyColorMap(heat, cv2.COLORMAP_JET), .45, 0)
    cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)
    return overlay


def collect(work, name):
    bench = work / 'bench' / name
    fps = []
    for image, dets in read_json(bench / 'hn_val_detections.json').items():
        fps += [(image, d) for d in dets if d[4] >= .2]
    records = read_json(work / 'manifests_v10' / 'validation_full.json')
    predictions = read_json(bench / 'validation_full_predictions.json')
    tps = []
    for pred in predictions:
        r = records[pred['image_id']]
        x, y, w, h = pred['bbox']
        box = [x, y, x + w, y + h, pred['score']]
        if pred['score'] < .2:
            continue
        if not r['boxes']:
            fps.append((r['image'], box))
        elif pred['score'] >= .4 and max(iou(box, g) for g in r['boxes']) >= .5:
            tps.append((r['image'], box))
    fps.sort(key=lambda x: -x[1][4])
    tps.sort(key=lambda x: -x[1][4])
    return fps, tps


def main():
    from sklearn.cluster import KMeans
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--drise', type=int, default=30, help='detections per FP/TP set')
    p.add_argument('--masks', type=int, default=400)
    p.add_argument('--perturb', type=int, default=150)
    a = p.parse_args()
    work = work_dir()
    out = work / 'diagnose' / a.name
    out.mkdir(parents=True, exist_ok=True)
    fps, tps = collect(work, a.name)
    result = dict(false_positives=len(fps), true_positives_considered=len(tps))
    # 1 clusters
    clip = Clip()
    scorer = FireScorer(clip)
    sample = fps[:4000]
    crops = [crop(cv2.imread(i), d) for i, d in sample]
    if len(sample) >= 10:
        embeddings = clip.images(crops)
        _, nearest = scorer.score(embeddings)
        k = min(10, len(sample) // 5)
        labels = KMeans(k, n_init=10, random_state=0).fit_predict(embeddings)
        clusters = []
        for c in range(k):
            members = [i for i in range(len(sample)) if labels[i] == c]
            names = [nearest[i] for i in members]
            title = max(set(names), key=names.count)
            clusters.append(dict(cluster=c, name=title, size=len(members),
                                 mean_score=float(np.mean([sample[i][1][4] for i in members]))))
            contact_sheet([(sample[i][0], sample[i][1], f'{sample[i][1][4]:.2f}') for i in members[:36]],
                          out / f'cluster_{c:02d}_{title}.jpg')
        result['fp_clusters'] = sorted(clusters, key=lambda c: -c['size'])
    # 2 D-RISE
    model = load_detector(a.checkpoint)
    for label, items in (('fp', fps[:a.drise]), ('tp', tps[:a.drise])):
        tiles = [(drise(model, cv2.imread(i), d, a.masks), None, f'{label} {d[4]:.2f}') for i, d in items]
        for s in range(0, len(tiles), 12):
            contact_sheet(tiles[s:s+12], out / f'drise_{label}_{s // 12:02d}.jpg', cols=4, tile=320)
    # 3 perturbations
    perturbation = {}
    for label, items in (('fp', fps[:a.perturb]), ('tp', tps[:a.perturb])):
        deltas = {}
        for image_path, d in items:
            image = cv2.imread(image_path)
            base = matched_score(model, image, d)
            for key, changed in perturb(image, d).items():
                after = matched_score(model, changed, d)
                deltas.setdefault(key, []).append((after - base, base >= .4 and after < .4))
        perturbation[label] = {k: dict(mean_confidence_change=float(np.mean([x for x, _ in v])),
                                       fraction_dropping_below_040=float(np.mean([y for _, y in v])), n=len(v))
                               for k, v in deltas.items()}
    result['perturbation'] = perturbation
    write_json(result, out / 'diagnose.json')
    print(json.dumps(result.get('fp_clusters', []), indent=1))
    print(json.dumps(perturbation, indent=1))


if __name__ == '__main__':
    main()
