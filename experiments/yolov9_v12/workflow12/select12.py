"""Steps 3a and 4d: selection with the rule frozen in PROTOCOL_V12.md (validation only; FASDD test never read).

  python select12.py variants   candidates = epochs 1-8 of a_data, b_zoom, c_zoom_paste      -> select_v12/
  python select12.py rec        candidates = epochs 1-4 of the recordings run + its start model -> select_v12_rec/

Reference = frozen V10 measured with the same code. Rule: common12.select. For the best checkpoint the size-aware
operating rule is chosen on validation (common12.best_rule). The best checkpoint is exported to ONNX 640x640.
start_for_recordings (variants mode) = best eligible checkpoint, or frozen V10 if none is eligible.
"""
import shutil
import sys

from common import load_detector, read_json, sha256, write_json
from common12 import CONFIG, MANIFESTS11, V10_FROZEN, V10_FROZEN_SHA, WORK, best_rule, measure, select


def main():
    from data import Flames
    from finish import export
    mode = sys.argv[1]
    if sha256(V10_FROZEN) != V10_FROZEN_SHA:
        raise RuntimeError('frozen V10 checksum mismatch')
    v10 = dict(measure(V10_FROZEN), variant='v10')
    if mode == 'variants':
        out = WORK / 'select_v12'
        candidates, missing = [], []
        for name in CONFIG['variants']:
            for e in range(1, CONFIG['base']['epochs'] + 1):
                path = WORK / 'train_v12' / name / f'epoch_{e:02d}.pth'
                if path.exists():
                    candidates.append(dict(measure(path), variant=name))
                else:
                    missing.append(str(path))
    elif mode == 'rec':
        out = WORK / 'select_v12_rec'
        start = read_json(WORK / 'select_v12' / 'selection.json')['start_for_recordings']
        candidates = [dict(measure(start['checkpoint']), variant=start['variant'] + ' (start, no recordings)')]
        missing = []
        for e in range(1, CONFIG['recordings']['epochs'] + 1):
            path = WORK / 'train_v12' / 'rec' / f'epoch_{e:02d}.pth'
            if path.exists():
                candidates.append(dict(measure(path), variant='rec'))
            else:
                missing.append(str(path))
    else:
        raise SystemExit(__doc__)
    if missing:
        raise RuntimeError(f'missing checkpoints (training incomplete?): {missing[:4]}')
    for r in candidates:
        print(r['variant'], r['epoch'], {k: round(r[k], 4) for k in ('AP50', 'recall', 'small_recall')},
              'nuisance', round(r['nuisance']['score'], 4), flush=True)
    result = select(candidates, v10)
    best = result['best']
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best['checkpoint'], out / 'best.pth')
    rule = best_rule(best['sha256'], v10)
    rule_v10 = best_rule(v10['sha256'], v10)
    doc = dict(rule=__doc__.strip(), protocol='PROTOCOL_V12.md', v10_reference=v10, candidates=candidates, **result,
               operating_rule=rule, operating_rule_v10=rule_v10, best_checkpoint=str(out / 'best.pth'))
    if mode == 'variants':
        eligible = [r for r in candidates if r['eligible']]
        start = result['best'] if result['best_is_eligible'] else v10
        doc['start_for_recordings'] = dict(checkpoint=start['checkpoint'], variant=start['variant'], epoch=start['epoch'],
                                           sha256=start['sha256'], eligible_count=len(eligible))
    write_json(doc, out / 'selection.json')
    print('BEST', best['variant'], 'epoch', best['epoch'], '| eligible', result['best_is_eligible'], '|', result['reason'], flush=True)
    model = load_detector(out / 'best.pth').cpu()
    export(model, Flames(read_json(MANIFESTS11 / 'validation_full.json'), 640), out / 'export')


if __name__ == '__main__':
    main()
