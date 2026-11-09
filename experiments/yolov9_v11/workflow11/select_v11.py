"""Step 5a: V11 checkpoint selection. Rule fixed in PROTOCOL_V11.md before training; FASDD test never used.

Reference: frozen V10, evaluated here with the same code and data as every candidate.
Candidates: EMA checkpoints of epochs 21-40 (train_v11/epoch_XX.pth).
Measured per checkpoint: full-val AP50, AP50:95, P/R @.40 (IoU .5), recall @.40 on the fixed 564-image small-flame
val slice (same slice as V10 selection), and the real-nuisance score on hn_val (V10 definition).
Eligible: AP50 >= V10 - .01, recall >= V10 - .01, real-nuisance score <= 1.5 x V10.
Best eligible: highest small-flame recall; within .005 of the top -> higher full-val recall; then higher AP50:95.
Adopt V11 only if best eligible small-flame recall >= V10 + .02 (2 points). Otherwise V10 stays the selected model.
The best eligible checkpoint (or, if none is eligible, the highest small-flame recall one, flagged) is exported
to ONNX 640x640 with parity checks and benchmarked as "v11" either way, so the owner can see it.
"""
import shutil

import torch

from common import detect_records, load_detector, read_json, sha256, write_json
from paths11 import V10_FROZEN, V10_FROZEN_SHA, WORK
from select_v10 import nuisance, small_recall, small_slice


def measure(path, validation, validation_records, hn_val, small, cache):
    from metrics import evaluate
    digest = sha256(path)
    cached = cache / f'{digest}.json'
    if cached.exists():
        return read_json(cached)
    model = load_detector(path)
    metrics, predictions = evaluate(model, validation, 16, 4)
    r_small, tp, n = small_recall(validation_records, predictions, small)
    score, _ = nuisance(hn_val, detect_records(model, hn_val, floor=.2))
    epoch = torch.load(path, map_location='cpu', weights_only=False).get('epoch', 0)
    row = dict(checkpoint=str(path), epoch=epoch, sha256=digest, AP50=metrics['coco']['AP50'],
               AP50_95=metrics['coco']['AP50_95'], precision=metrics['at_confidence_0.40']['precision'],
               recall=metrics['at_confidence_0.40']['recall'], small_recall=r_small, small_tp=tp, small_boxes=n,
               nuisance=score)
    write_json(row, cached)
    return row


def main():
    from data import Flames
    from finish import export
    run, out = WORK / 'train_v11', WORK / 'select_v11'
    (out / 'cache').mkdir(parents=True, exist_ok=True)
    manifests = WORK / 'manifests_v11'
    validation_records = read_json(manifests / 'validation_full.json')
    validation = Flames(validation_records, 640)
    hn_val = read_json(manifests / 'hn_val.json')
    small = small_slice(validation_records)
    if sha256(V10_FROZEN) != V10_FROZEN_SHA:
        raise RuntimeError('frozen V10 checksum mismatch')
    v10 = measure(V10_FROZEN, validation, validation_records, hn_val, small, out / 'cache')
    paths = [run / f'epoch_{e:02d}.pth' for e in range(21, 41)]
    missing = [p.name for p in paths if not p.exists()]
    rows = [measure(p, validation, validation_records, hn_val, small, out / 'cache') for p in paths if p.exists()]
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('nuisance', 'sha256')},
              'nuisance', round(r['nuisance']['score'], 4), flush=True)
    floors = dict(AP50=v10['AP50'] - .01, recall=v10['recall'] - .01, nuisance_max=1.5 * v10['nuisance']['score'])
    for r in rows:
        r['eligible'] = (r['AP50'] >= floors['AP50'] and r['recall'] >= floors['recall']
                         and r['nuisance']['score'] <= floors['nuisance_max'])
    eligible = [r for r in rows if r['eligible']]
    pool = eligible or rows
    top = max(r['small_recall'] for r in pool)
    best = max((r for r in pool if r['small_recall'] >= top - .005), key=lambda r: (r['recall'], r['AP50_95']))
    adopt = bool(eligible) and best['small_recall'] >= v10['small_recall'] + .02
    reason = ('V11 adopted: best eligible epoch beats V10 small-flame recall by >= 2 points' if adopt else
              'V10 retained: ' + ('no V11 epoch met the AP50 / recall / nuisance floors' if not eligible else
                                  'best eligible epoch did not beat V10 small-flame recall by 2 points'))
    shutil.copyfile(best['checkpoint'], out / 'v11_best.pth')
    write_json(dict(rule=__doc__.strip(), v10_reference=v10, floors=floors, candidates=rows, missing=missing,
                    best=best, best_is_eligible=best in eligible, v11_adopted=adopt, reason=reason),
               out / 'selection.json')
    print('BEST', best['checkpoint'], '| eligible', best in eligible, '|', reason, flush=True)
    model = load_detector(out / 'v11_best.pth').cpu()
    export(model, validation, out / 'export')


if __name__ == '__main__':
    main()
