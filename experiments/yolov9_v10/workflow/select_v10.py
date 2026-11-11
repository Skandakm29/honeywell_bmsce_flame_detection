"""Checkpoint selection with the rule frozen before training (FASDD test never used).

Candidates: V9 (initial.pth, epoch 0) and every epoch_XX.pth.
Eligible: full-val AP50 >= V9 - .01, recall@.40 >= V9 - .01, and box recall@.40 (IoU .50) on the fixed
564-image small-flame val slice >= V9 - .01 (unrounded V9 values from this same evaluation).
Real-nuisance score: on hn_val WITHOUT synthetic screens, the share of images flagged at .40 within each scene
(near-duplicate group / clip), averaged with equal weight per scene. Lower is better.
Switch from V9 only if the best eligible epoch scores <= 0.90 x V9's score (V9 kept if its score is already 0);
ties broken by higher AP50:95. Scenes improved/worsened and synthetic screens are reported separately.
The selected model is exported to ONNX with parity checks.
"""
import csv
import gzip
import shutil
from pathlib import Path

import torch

from common import PACKAGE, detect_records, load_detector, read_json, sha256, work_dir, write_json


def small_slice(records):
    keep = set()
    with gzip.open(PACKAGE / 'refs' / 'segments_exact.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            if row['split'] == 'val' and row['flame_size'] == 'small':
                keep.add(Path(row['name']).name)
    return [i for i, r in enumerate(records) if Path(r['image']).name in keep]


def small_recall(records, predictions, index):
    from metrics import operating_metrics
    position = {i: k for k, i in enumerate(index)}
    preds = [dict(p, image_id=position[p['image_id']]) for p in predictions if p['image_id'] in position]
    m = operating_metrics([records[i] for i in index], preds)
    return m['recall'], m['TP'], m['TP'] + m['FN']


def nuisance(records, detections):
    scenes, screens = {}, {'images': 0, 'flag_020': 0, 'flag_040': 0}
    for r, d in zip(records, detections):
        f20, f40 = any(x[4] >= .2 for x in d), any(x[4] >= .4 for x in d)
        if r['category'] == 'synthetic_screen':
            screens['images'] += 1
            screens['flag_020'] += f20
            screens['flag_040'] += f40
            continue
        s = scenes.setdefault(r.get('group', r['image']), [0, 0, 0])
        s[0] += 1
        s[1] += f20
        s[2] += f40
    fractions = {g: s[2] / s[0] for g, s in scenes.items()}
    return dict(score=sum(fractions.values()) / max(1, len(fractions)), scenes=len(fractions),
                images=sum(s[0] for s in scenes.values()), flag_020=sum(s[1] for s in scenes.values()),
                flag_040=sum(s[2] for s in scenes.values()), synthetic_screens=screens), fractions


def main():
    from data import Flames
    from finish import export
    from metrics import evaluate
    work = work_dir()
    run = work / 'train_v10'
    manifests = work / 'manifests_v10'
    config = read_json(Path(__file__).with_name('config_v10.json'))
    validation_records = read_json(manifests / 'validation_full.json')
    validation = Flames(validation_records, 640)
    hn_val = read_json(manifests / 'hn_val.json')
    small = small_slice(validation_records)
    epochs = sorted(run.glob('epoch_*.pth'))
    warnings = [] if len(epochs) == config['epochs'] else [f'only {len(epochs)} of {config["epochs"]} epoch files exist']
    cache = work / 'select' / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    rows, scene_fractions = [], {}
    for path in [run / 'initial.pth'] + epochs:
        digest = sha256(path)
        cached = cache / f'{digest}.json'
        if cached.exists():
            row = read_json(cached)
        else:
            model = load_detector(path)
            metrics, predictions = evaluate(model, validation, 16, 4)
            recall_small, tp_small, boxes_small = small_recall(validation_records, predictions, small)
            score, fractions = nuisance(hn_val, detect_records(model, hn_val, floor=.2))
            state = torch.load(path, map_location='cpu', weights_only=False)
            row = dict(checkpoint=path.name, epoch=0 if path.name == 'initial.pth' else state['epoch'], sha256=digest,
                       AP50=metrics['coco']['AP50'], AP50_95=metrics['coco']['AP50_95'],
                       precision=metrics['at_confidence_0.40']['precision'], recall=metrics['at_confidence_0.40']['recall'],
                       small_recall=recall_small, small_tp=tp_small, small_boxes=boxes_small, nuisance=score,
                       scene_fractions=fractions)
            write_json(row, cached)
            del model
        scene_fractions[row['checkpoint']] = row.pop('scene_fractions')
        rows.append(row)
        print({k: v for k, v in row.items() if k != 'nuisance'}, 'nuisance score', round(row['nuisance']['score'], 4), flush=True)
    v9 = rows[0]
    floors = dict(AP50=v9['AP50'] - .01, recall=v9['recall'] - .01, small_recall=v9['small_recall'] - .01)
    for row in rows:
        row['eligible'] = (row['AP50'] >= floors['AP50'] and row['recall'] >= floors['recall']
                           and row['small_recall'] >= floors['small_recall'])
        base = scene_fractions[v9['checkpoint']]
        mine = scene_fractions[row['checkpoint']]
        row['scenes_improved'] = sum(mine[g] < base[g] for g in base)
        row['scenes_worsened'] = sum(mine[g] > base[g] for g in base)
    candidates = [r for r in rows[1:] if r['eligible']]
    best = min(candidates, key=lambda r: (r['nuisance']['score'], -r['AP50_95'])) if candidates else None
    v9_score = v9['nuisance']['score']
    switch = best is not None and v9_score > 0 and best['nuisance']['score'] <= .90 * v9_score
    chosen = best if switch else v9
    reason = ('best eligible epoch cuts the real-nuisance score by >= 10%' if switch else
              'V9 kept: ' + ('no epoch met the AP50 / recall / small-flame floors' if not candidates else
                             'V9 nuisance score already 0' if v9_score == 0 else
                             'best eligible epoch did not cut the real-nuisance score by >= 10%'))
    selected = run / chosen['checkpoint']
    shutil.copyfile(selected, work / 'select' / 'selected.pth')
    write_json(dict(rule=__doc__.strip(), floors=floors, candidates=rows, best_eligible=best, selected=chosen,
                    v9_retained=not switch, reason=reason, warnings=warnings), work / 'select' / 'selection.json')
    for w in warnings:
        print('WARNING:', w)
    print('SELECTED', chosen['checkpoint'], '|', reason)
    model = load_detector(work / 'select' / 'selected.pth').cpu()
    export(model, validation, work / 'select' / 'export')


if __name__ == '__main__':
    main()
