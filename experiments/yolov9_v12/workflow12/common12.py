"""Shared V12 code: paths, cached measurement of a checkpoint on VALIDATION ONLY, the size-aware operating rule,
and the frozen selection rule (PROTOCOL_V12.md). V10 code is imported read-only; nothing here touches FASDD test.

Measured per checkpoint (same code for V10 and every candidate):
  AP50, AP50:95 (COCO, full validation), precision / recall at .40 (IoU .5, one-to-one),
  small-flame recall on the fixed 564-image validation slice (V10 definition),
  real-nuisance score on hn_val (V10 definition: per-scene share flagged, synthetic screens excluded).
Operating rule (size-aware threshold): a detection counts if score >= .40, OR if its longest side at the 640 input
  is < S px and score >= t. S and t come from a fixed grid; base rule = .40 for every size.
"""
import json
import os
from pathlib import Path

import numpy as np

from common import detect_records, load_detector, read_json, sha256, work_dir, write_json

HERE = Path(__file__).resolve().parent
PACKAGE12 = HERE.parent
INBOX = PACKAGE12 / 'inbox'
WORK = work_dir()
OUTBOX = WORK / 'outbox'
MANIFESTS11 = WORK / 'manifests_v11'          # V10 train + label fixes + dense frames + WhatsApp (V11 step 2)
MANIFESTS12 = WORK / 'manifests_v12'          # MANIFESTS11 + reviewed recordings (step 4b)
EVAL = ('val.json', 'validation_full.json', 'hn_val.json', 'test.json')
V10_FROZEN = Path(os.environ.get('V12_V10_CHECKPOINT', WORK / 'FROZEN_V10' / 'selected.pth'))
V10_FROZEN_SHA = 'd046dab613faa72ec650d37dc40d0740ff50ac09df4eb97919d10cfd532b721d'
CONFIG = json.loads((HERE / 'config_v12.json').read_text())
CACHE = WORK / 'measure_v12'
_records = {}


def records(name):
    if name not in _records:
        _records[name] = read_json(MANIFESTS11 / name)
    return _records[name]


def measure(path):
    """Full validation + hn_val for one checkpoint; cached by checkpoint SHA256 (json + raw detections >= .10)."""
    from data import Flames
    from metrics import evaluate
    digest = sha256(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    row_path, raw_path = CACHE / f'{digest}.json', CACHE / f'{digest}.npz'
    if row_path.exists() and raw_path.exists():
        return read_json(row_path)
    import torch
    model = load_detector(path)
    metrics, predictions = evaluate(model, Flames(records('validation_full.json'), 640), 16, 4)
    hn = detect_records(model, records('hn_val.json'), floor=.10)
    val_raw = np.array([[p['image_id'], *p['bbox'], p['score']] for p in predictions if p['score'] >= .10],
                       np.float32).reshape(-1, 6)
    hn_raw = np.array([[i, *d[:5]] for i, ds in enumerate(hn) for d in ds], np.float32).reshape(-1, 6)
    np.savez_compressed(raw_path, val=val_raw, hn=hn_raw)
    epoch = torch.load(path, map_location='cpu', weights_only=False).get('epoch', 0)
    row = dict(checkpoint=str(path), epoch=epoch, sha256=digest, AP50=metrics['coco']['AP50'],
               AP50_95=metrics['coco']['AP50_95'], **rule_metrics(digest, 0, 1.))
    write_json(row, row_path)
    del model
    torch.cuda.empty_cache()
    return row


def _longest_val(raw):
    recs = records('validation_full.json')
    ids = raw[:, 0].astype(int)
    dims = np.array([max(recs[i]['width'], recs[i]['height']) for i in ids], np.float32)
    return np.maximum(raw[:, 3], raw[:, 4]) * 640 / np.maximum(dims, 1)


def _longest_hn(raw):
    recs = records('hn_val.json')
    ids = raw[:, 0].astype(int)
    dims = np.array([max(recs[i]['width'], recs[i]['height']) for i in ids], np.float32)
    return np.maximum(raw[:, 3] - raw[:, 1], raw[:, 4] - raw[:, 2]) * 640 / np.maximum(dims, 1)


def keep_mask(score, longest, side, threshold, base=None):
    base = CONFIG['operating_rule_grid']['base_threshold'] if base is None else base
    return (score >= base) | ((longest < side) & (score >= threshold))


def rule_metrics(digest, side, threshold):
    """Precision / recall / small recall / nuisance under the rule (side=0 -> plain .40)."""
    from metrics import operating_metrics
    from select_v10 import nuisance, small_slice
    raw = np.load(CACHE / f'{digest}.npz')
    val, hn = raw['val'], raw['hn']
    recs = records('validation_full.json')
    kv = keep_mask(val[:, 5], _longest_val(val), side, threshold) if len(val) else np.zeros(0, bool)
    preds = [dict(image_id=int(v[0]), category_id=1, bbox=[float(x) for x in v[1:5]], score=float(v[5])) for v in val[kv]]
    full = operating_metrics(recs, preds, confidence=0.)
    index = small_slice(recs)
    position = {i: k for k, i in enumerate(index)}
    small = operating_metrics([recs[i] for i in index],
                              [dict(p, image_id=position[p['image_id']]) for p in preds if p['image_id'] in position],
                              confidence=0.)
    hn_recs = records('hn_val.json')
    kh = keep_mask(hn[:, 5], _longest_hn(hn), side, threshold) if len(hn) else np.zeros(0, bool)
    flagged = [[] for _ in hn_recs]
    for h in hn[kh]:
        flagged[int(h[0])].append([*map(float, h[1:5]), 1.])      # kept -> flagged
    score, _ = nuisance(hn_recs, flagged)
    return dict(precision=full['precision'], recall=full['recall'], FP=full['FP'], small_recall=small['recall'],
                small_tp=small['TP'], small_boxes=small['TP'] + small['FN'], nuisance=score)


def best_rule(digest, reference):
    """Search the fixed grid on VALIDATION: highest small recall with precision >= base precision - 1 pt and
    nuisance <= 1.5 x the reference (frozen V10 at .40). Ties: higher precision. Returns (side, threshold, row)."""
    base = rule_metrics(digest, 0, 1.)
    grid = CONFIG['operating_rule_grid']
    best = (0, CONFIG['operating_rule_grid']['base_threshold'], base)
    rows = []
    for side in grid['small_side_px']:
        for t in grid['small_threshold']:
            m = rule_metrics(digest, side, t)
            ok = (m['precision'] >= base['precision'] - .01 and
                  m['nuisance']['score'] <= CONFIG['selection']['nuisance_max_ratio'] * reference['nuisance']['score'])
            rows.append(dict(side=side, threshold=t, eligible=ok, **m))
            if ok and (m['small_recall'], m['precision']) > (best[2]['small_recall'], best[2]['precision']):
                best = (side, t, m)
    return dict(side=best[0], threshold=best[1], chosen=best[2], base=base, grid=rows)


def select(candidates, reference):
    """Frozen rule. Eligible: AP50 >= ref - 1 pt, recall >= ref - 1 pt, nuisance <= 1.5 x ref.
    Best: highest small recall; within .5 pt -> higher recall, then higher AP50:95.
    Adopted over V10 only if eligible and small recall >= ref + 1 pt."""
    s = CONFIG['selection']
    floors = dict(AP50=reference['AP50'] - s['ap50_floor_pts'] / 100, recall=reference['recall'] - s['recall_floor_pts'] / 100,
                  nuisance_max=s['nuisance_max_ratio'] * reference['nuisance']['score'])
    for r in candidates:
        r['eligible'] = (r['AP50'] >= floors['AP50'] and r['recall'] >= floors['recall'] and
                         r['nuisance']['score'] <= floors['nuisance_max'])
    eligible = [r for r in candidates if r['eligible']]
    pool = eligible or candidates
    top = max(r['small_recall'] for r in pool)
    best = max((r for r in pool if r['small_recall'] >= top - .005), key=lambda r: (r['recall'], r['AP50_95']))
    adopt = bool(eligible) and best['small_recall'] >= reference['small_recall'] + s['adopt_small_recall_gain_pts'] / 100
    reason = ('adopted: best eligible checkpoint beats V10 small-flame recall by >= 1 point' if adopt else
              'V10 retained: ' + ('no checkpoint met the AP50 / recall / nuisance floors' if not eligible else
                                  'best eligible checkpoint did not beat V10 small-flame recall by 1 point'))
    return dict(floors=floors, best=best, best_is_eligible=best in eligible, adopted=adopt, reason=reason)


def pct(x):
    return '-' if x is None else f'{100 * x:.2f}%'
