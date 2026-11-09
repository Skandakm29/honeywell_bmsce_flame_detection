"""Step 5: the ONE test run, after every selection is final (nothing is changed after it).

Final candidate = select_v12_rec/best.pth (after the recordings run), or select_v12/best.pth if the owner wrote
inbox/skip_recordings.txt. Its adoption over V10 was already decided on validation (selection.json).
  1. FASDD test via V10 bench.py --test (refuses a rerun): AP50, P/R @.40, segments.
  2. The same test predictions under the operating rule chosen on validation (and V10's test predictions under
     V10's own chosen rule, if bench/v10/test_predictions.json exists).
  3. Distance test (corridor recordings, never trained on): per clip, share of reviewed flame frames with a
     detection overlapping the box (IoU >= .1), and share of no-flame frames with any detection, at .40 and under the rule.
Output: test_v12/test.json
"""
import csv
import gzip
from pathlib import Path

import numpy as np

from common import PACKAGE, detect_records, load_detector, read_json, write_json
from common12 import INBOX, MANIFESTS11, V10_FROZEN, WORK, keep_mask


def final_selection():
    rec = WORK / 'select_v12_rec' / 'selection.json'
    if rec.exists():
        return read_json(rec), 'after recordings'
    if (INBOX / 'skip_recordings.txt').exists():
        return read_json(WORK / 'select_v12' / 'selection.json'), 'without recordings (owner skipped them)'
    raise RuntimeError('recordings selection missing and inbox/skip_recordings.txt not written by the owner')


def rule_on_predictions(predictions, records, side, threshold):
    from metrics import operating_metrics
    segments = {}
    with gzip.open(PACKAGE / 'refs' / 'segments_exact.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            if row['split'] == 'test':
                segments[Path(row['name']).name] = row.get('flame_size')

    def longest(p):
        r = records[p['image_id']]
        return max(p['bbox'][2], p['bbox'][3]) * 640 / max(r['width'], r['height'])
    kept = [p for p in predictions if keep_mask(np.float32(p['score']), np.float32(longest(p)), side, threshold)]
    full = operating_metrics(records, kept, confidence=0.)
    index = [i for i, r in enumerate(records) if segments.get(Path(r['image']).name) == 'small']
    pos = {i: k for k, i in enumerate(index)}
    small = operating_metrics([records[i] for i in index], [dict(p, image_id=pos[p['image_id']]) for p in kept if p['image_id'] in pos],
                              confidence=0.)
    return dict(side=side, threshold=threshold, precision=full['precision'], recall=full['recall'], FP=full['FP'],
                small_recall=small['recall'], small_boxes=small['TP'] + small['FN'])


def distance(model, items, side, threshold):
    dets = detect_records(model, items, floor=.10)
    clips = {}
    for r, ds in zip(items, dets):
        c = clips.setdefault(r['clip'], dict(flame=r.get('flame'), distance_m=r.get('distance_m'), frames=0,
                                            hit_040=0, hit_rule=0))
        c['frames'] += 1
        for key, s, t in (('hit_040', 0, 1.), ('hit_rule', side, threshold)):
            kept = []
            for d in ds:
                long_side = max(d[2] - d[0], d[3] - d[1]) * 640 / max(r['width'], r['height'])
                if keep_mask(np.float32(d[4]), np.float32(long_side), s, t):
                    kept.append(d)
            if r['boxes']:
                b = r['boxes'][0]
                # hit: overlaps the reviewed box, or its centre lies inside it (cluster boxes hold several diyas)
                ok = any(_iou(d, b) >= .1 or (b[0] <= (d[0] + d[2]) / 2 <= b[2] and b[1] <= (d[1] + d[3]) / 2 <= b[3])
                         for d in kept)
            else:
                ok = bool(kept)                      # no-flame clip: a "hit" is a false alarm frame
            c[key] += ok
    for c in clips.values():
        c['rate_040'], c['rate_rule'] = c['hit_040'] / c['frames'], c['hit_rule'] / c['frames']
    return clips


def _iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def main():
    import subprocess
    import sys
    sel, how = final_selection()
    final = Path(sel['best_checkpoint'])
    out = WORK / 'test_v12'
    out.mkdir(parents=True, exist_ok=True)
    bench_dir = WORK / 'bench' / 'v12_final'
    if not (bench_dir / 'test_summary.json').exists():
        flow = Path(sys.modules['common'].__file__).parent
        subprocess.run([sys.executable, '-u', str(flow / 'bench.py'), '--name', 'v12_final', '--checkpoint', str(final), '--test'],
                       check=True)
    records = read_json(MANIFESTS11 / 'test.json')
    rule, rule10 = sel['operating_rule'], sel['operating_rule_v10']
    result = dict(final_checkpoint=str(final), selected=how, adopted=sel['adopted'], reason=sel['reason'],
                  fasdd_test_v12=read_json(bench_dir / 'test_summary.json'))
    preds = read_json(bench_dir / 'test_predictions.json')
    result['fasdd_test_v12_rule'] = dict(at_040=rule_on_predictions(preds, records, 0, 1.),
                                         chosen=rule_on_predictions(preds, records, rule['side'], rule['threshold']))
    v10_preds = WORK / 'bench' / 'v10' / 'test_predictions.json'
    if v10_preds.exists():
        p10 = read_json(v10_preds)
        result['fasdd_test_v10_rule'] = dict(at_040=rule_on_predictions(p10, records, 0, 1.),
                                             chosen=rule_on_predictions(p10, records, rule10['side'], rule10['threshold']))
    test_items = WORK / 'rec_v12' / 'distance_test.json'
    items = read_json(test_items) if test_items.exists() else []
    if items:
        result['distance_test'] = dict(
            v10=distance(load_detector(V10_FROZEN), items, rule10['side'], rule10['threshold']),
            v12=distance(load_detector(final), items, rule['side'], rule['threshold']))
    write_json(result, out / 'test.json')
    print('test done ->', out / 'test.json', flush=True)


if __name__ == '__main__':
    main()
