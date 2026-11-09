"""Miss autopsy on FASDD validation for each benchmarked model (bench/<name>/validation_full_predictions.json).
Same outcome definitions as the laptop check (investigation_week2/v11/checks/autopsy.py), fixed before results:
detected / merged / misplaced / low_score / weak / blind per ground-truth box, by size at the 640 input.
"""
from collections import Counter, defaultdict

import numpy as np

from common import read_json, write_json
from paths11 import WORK, iou

ORDER = ['detected', 'merged', 'misplaced', 'low_score', 'weak', 'blind']


def load(path, n):
    g = [[] for _ in range(n)]
    for p in read_json(path):
        x, y, w, h = p['bbox']
        g[p['image_id']].append([x, y, x + w, y + h, p['score']])
    return g


def operating_match(gt, preds, conf=.4, thr=.5):
    """Score-ordered one-to-one matching, identical to workflow/metrics.operating_metrics."""
    matched = {}
    for k in sorted(range(len(preds)), key=lambda k: -preds[k][4]):
        if preds[k][4] < conf:
            continue
        ious = iou([preds[k][:4]], gt)[0] if len(gt) else []
        best, best_id = 0., None
        for i, v in enumerate(ious):
            if i not in matched and v > best:
                best, best_id = v, i
        if best_id is not None and best >= thr:
            matched[best_id] = k
    return matched


def classify(records_, grouped, width, height):
    rows = []
    for r, preds in zip(records_, grouped):
        gt = np.asarray(r['boxes'], float).reshape(-1, 4)
        if not len(gt):
            continue
        preds = np.asarray(preds, float).reshape(-1, 5)
        matched = operating_match(gt, preds)
        used = set(matched.values())
        ious = iou(gt, preds[:, :4]) if len(preds) else np.zeros((len(gt), 0))
        ratio = min(width / r['width'], height / r['height'])
        for i, b in enumerate(gt):
            w, h = b[2] - b[0], b[3] - b[1]
            row = dict(area=w * h, side640=max(w, h) * ratio, best_score=0.)
            if i in matched:
                row['outcome'] = 'detected'
            else:
                s = preds[:, 4] if len(preds) else np.zeros(0)
                ov = ious[i]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                inside = ((preds[:, 0] <= cx) & (cx <= preds[:, 2]) & (preds[:, 1] <= cy) & (cy <= preds[:, 3])) if len(preds) else np.zeros(0, bool)
                hi = s >= .4
                covering = hi & ((ov >= .1) | inside)
                if covering.any() and any(k in used or (iou([preds[k, :4]], gt)[0] >= .1).sum() > 1
                                          for k in np.where(covering)[0]):
                    row['outcome'] = 'merged'
                elif (hi & (ov >= .1)).any():
                    row['outcome'] = 'misplaced'
                elif ((s >= .1) & (ov >= .5)).any():
                    row['outcome'] = 'low_score'
                elif ((s >= .05) & (ov >= .3)).any():
                    row['outcome'] = 'weak'
                else:
                    row['outcome'] = 'blind' if not ((s >= .05) & (ov >= .1)).any() else 'weak'
                near = (ov >= .3)
                row['best_score'] = float(s[near].max()) if near.any() else 0.
            rows.append(row)
    return rows


def size_bins(row):
    a, s = row['area'], row['side640']
    coco = 'small' if a < 32 ** 2 else 'medium' if a < 96 ** 2 else 'large'
    px = '<8' if s < 8 else '8-16' if s < 16 else '16-32' if s < 32 else '32-64' if s < 64 else '>=64'
    return coco, px


def table(rows, key):
    groups = defaultdict(Counter)
    for row in rows:
        groups[key(row)][row['outcome']] += 1
        groups['ALL'][row['outcome']] += 1
    lines = ['| group | boxes | ' + ' | '.join(ORDER) + ' |', '|' + '---|' * (len(ORDER) + 2)]
    rank = ['ALL', 'small', 'medium', 'large', '<8', '8-16', '16-32', '32-64', '>=64']
    for g, c in sorted(groups.items(), key=lambda kv: rank.index(kv[0]) if kv[0] in rank else 99):
        n = sum(c.values())
        lines.append(f'| {g} | {n} | ' + ' | '.join(f'{100 * c[o] / n:.1f}%' for o in ORDER) + ' |')
    return '\n'.join(lines)


def threshold_sweep(records_, grouped):
    """Recall at IoU .5 for lower operating confidences (diagnostic only; choosing uses val + nuisance)."""
    out = []
    for conf in (.40, .35, .30, .25, .20):
        tp = n = 0
        for r, preds in zip(records_, grouped):
            gt = np.asarray(r['boxes'], float).reshape(-1, 4)
            n += len(gt)
            if len(gt):
                tp += len(operating_match(gt, np.asarray(preds, float).reshape(-1, 5), conf))
        out.append((conf, tp / n))
    return out


def main():
    recs = read_json(WORK / 'manifests_v11' / 'validation_full.json')
    out, md = {}, ['## Miss autopsy (FASDD validation, conf .40, IoU .50)', '']
    for name in ('v9', 'v10', 'v11'):
        path = WORK / 'bench' / name / 'validation_full_predictions.json'
        if not path.exists():
            continue
        grouped = load(path, len(recs))
        rows = classify(recs, grouped, 640, 640)
        out[name] = {o: sum(r['outcome'] == o for r in rows) for o in ORDER}
        md += [f'### {name}: by longest side at the 640 input', table(rows, lambda r: size_bins(r)[1]), '',
               'recall vs confidence: ' + ' | '.join(f'{c:.2f}: {100 * v:.2f}%' for c, v in threshold_sweep(recs, grouped)), '']
    write_json(out, WORK / 'select_v11' / 'autopsy.json')
    (WORK / 'select_v11' / 'AUTOPSY.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
