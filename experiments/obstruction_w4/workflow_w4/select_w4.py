"""Step 3: choose the 2-class checkpoint with the rule in PROTOCOL_W4.md, export ONNX, write the report.

Reference = initial.pth = the start model grown to 2 classes (identical flame scores, obstruction untrained),
measured by the same code during training. Candidates = epoch_01 .. epoch_NN. Metrics are the monitor metrics
train_v10 saved with every checkpoint (validation only; FASDD test is never read here).
Eligible if ALL hold:
  flame AP50 (clean monitor images)        >= reference - 1 pt
  flame recall @ .40 (clean monitor images) >= reference - 1 pt
  flame false-positive image rate on hn_val @ .40 <= 1.5 x reference
  obstruction false-alarm image rate @ .40: clean monitor images <= 2 %, hn_val <= 2 %
Best: highest obstruction AP50 on the synthetic validation obstructions; within .5 pt -> higher flame AP50.
No eligible checkpoint -> nothing is adopted (the report says why); the flame model stays as it was.
"""
import json
import shutil
import zipfile
from pathlib import Path

import torch

from common import read_json, sha256, write_json
from common_w4 import CONFIG, MANIFESTS_W4, OUTBOX, SELECT_DIR, TENSORLAB, TRAIN_DIR


def row(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    m = state['metrics']
    hn = state.get('hn_val') or m.get('hn_val')
    o = m['obstruction']
    return dict(checkpoint=str(path), epoch=state['epoch'], sha256=sha256(path),
                flame_AP50=m['coco']['AP50'], flame_AP50_95=m['coco']['AP50_95'],
                flame_recall=m['at_confidence_0.40']['recall'], flame_precision=m['at_confidence_0.40']['precision'],
                flame_hn_rate=hn['0.40']['fp_image_rate'],
                obstruction_AP50=(o['coco'] or {}).get('AP50'), obstruction_AP50_95=(o['coco'] or {}).get('AP50_95'),
                obstruction_recall=(o['at_confidence'] or {}).get('recall'),
                obstruction_precision=(o['at_confidence'] or {}).get('precision'),
                obstruction_image_rate=o['image_detection_rate'],
                obstruction_false_rate_clean=o['false_obstruction_image_rate'],
                obstruction_false_rate_hn=hn['obstruction']['0.40']['fp_image_rate'],
                blind_share_error=o['blind_share_mean_abs_error'], detected_by_kind=o['detected_by_kind'])


def select(reference, candidates, rule=CONFIG['selection']):
    floors = dict(flame_AP50=reference['flame_AP50'] - rule['flame_ap50_floor_pts'] / 100,
                  flame_recall=reference['flame_recall'] - rule['flame_recall_floor_pts'] / 100,
                  flame_hn_rate_max=reference['flame_hn_rate'] * rule['flame_hn_fp_max_ratio'],
                  obstruction_false_rate_clean_max=rule['obstruction_false_rate_clean_max'],
                  obstruction_false_rate_hn_max=rule['obstruction_false_rate_hn_max'])
    for r in candidates:
        failed = []
        if r['flame_AP50'] < floors['flame_AP50']:
            failed.append('flame AP50')
        if r['flame_recall'] < floors['flame_recall']:
            failed.append('flame recall')
        if r['flame_hn_rate'] > floors['flame_hn_rate_max'] + 1e-12:
            failed.append('flame hn_val false positives')
        if (r['obstruction_false_rate_clean'] or 0) > floors['obstruction_false_rate_clean_max']:
            failed.append('obstruction false alarms (clean)')
        if (r['obstruction_false_rate_hn'] or 0) > floors['obstruction_false_rate_hn_max']:
            failed.append('obstruction false alarms (hn_val)')
        if r['obstruction_AP50'] is None:
            failed.append('no obstruction metrics')
        r['eligible'], r['failed'] = not failed, failed
    eligible = [r for r in candidates if r['eligible']]
    if not eligible:
        return dict(floors=floors, best=None, adopted=False,
                    reason='no checkpoint met every floor; flame model unchanged')
    top = max(r['obstruction_AP50'] for r in eligible)
    best = max((r for r in eligible if r['obstruction_AP50'] >= top - rule['tie_pts'] / 100),
               key=lambda r: (r['flame_AP50'], r['obstruction_AP50']))
    return dict(floors=floors, best=best, adopted=True, reason='best eligible obstruction AP50')


def export_two_class(checkpoint, out):
    from data_w4 import Flames2
    from finish import export
    from model import Detector
    model = Detector(str(TENSORLAB), classes=len(CONFIG['classes']))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    result = export(model.eval(), Flames2(read_json(MANIFESTS_W4 / 'val.json'), 640), out)
    contract_path = Path(out) / 'model_contract.json'
    contract = read_json(contract_path)
    contract['architecture'] = 'TI YOLOv9-S-Lite two-class adapter (flame, obstruction)'
    contract['output']['fields'] = ['x1', 'y1', 'x2', 'y2', 'flame_probability', 'obstruction_probability']
    contract['postprocessing']['nms'] = 'per class'
    contract['postprocessing']['thresholds'] = CONFIG['operating']
    contract['blind_share'] = 'union area of obstruction boxes >= threshold / image area'
    write_json(contract, contract_path)
    return result


def report(doc, path):
    def p(x):
        return '-' if x is None else f'{100 * x:.2f}%'
    lines = ['# Week 4: flame + obstruction (two classes in the same YOLOv9-S-TI-Lite)', '',
             'Validation only (monitor set + synthetic obstructions). Rule: PROTOCOL_W4.md. Real-camera test: step 4.', '',
             f'**{doc["reason"]}**', '',
             '| epoch | eligible | flame AP50 | flame recall | flame hn FP | obstr. AP50 | obstr. recall | obstr. false (clean / hn) | blind-share error | failed |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in [doc['reference']] + doc['candidates']:
        lines.append(f'| {r["epoch"]}{" (start)" if r is doc["reference"] else ""} | {r.get("eligible", "-")} | {p(r["flame_AP50"])} | '
                     f'{p(r["flame_recall"])} | {p(r["flame_hn_rate"])} | {p(r["obstruction_AP50"])} | {p(r["obstruction_recall"])} | '
                     f'{p(r["obstruction_false_rate_clean"])} / {p(r["obstruction_false_rate_hn"])} | {p(r["blind_share_error"])} | '
                     f'{", ".join(r.get("failed", [])) or "-"} |')
    if doc['best']:
        lines += ['', f'Selected: epoch {doc["best"]["epoch"]}, `{doc["best_checkpoint"]}`.', '',
                  'Obstruction found per kind/shape (synthetic validation, confidence .40):', '',
                  '| kind/shape | images | found |', '|---|---|---|']
        for k, v in doc['best']['detected_by_kind'].items():
            lines.append(f'| {k} | {v["images"]} | {v["detected"]} |')
    lines += ['', 'Limits: obstruction numbers are on SYNTHETIC obstructions; real-camera numbers come from step 4.',
              'AM62A speed, int8 accuracy and TIDL compilation are not measured.', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main():
    reference = row(TRAIN_DIR / 'initial.pth')
    epochs = sorted(TRAIN_DIR.glob('epoch_*.pth'))
    if len(epochs) < CONFIG['base']['epochs']:
        raise RuntimeError(f'only {len(epochs)} epoch checkpoints; training incomplete')
    candidates = [row(e) for e in epochs]
    result = select(reference, candidates)
    SELECT_DIR.mkdir(parents=True, exist_ok=True)
    doc = dict(rule=__doc__.strip(), protocol='PROTOCOL_W4.md', reference=reference, candidates=candidates, **result)
    if result['best']:
        best = SELECT_DIR / 'best.pth'
        shutil.copyfile(result['best']['checkpoint'], best)
        doc['best_checkpoint'] = str(best)
        doc['export'] = export_two_class(best, SELECT_DIR / 'export')
    write_json(doc, SELECT_DIR / 'selection.json')
    report(doc, SELECT_DIR / 'REPORT_W4.md')
    OUTBOX.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTBOX / 'w4_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in [SELECT_DIR / 'selection.json', SELECT_DIR / 'REPORT_W4.md', TRAIN_DIR / 'history.json',
                  MANIFESTS_W4 / 'build_report.json', MANIFESTS_W4 / 'preview_val.jpg',
                  SELECT_DIR / 'export' / 'model_contract.json', SELECT_DIR / 'export' / 'export.json',
                  SELECT_DIR / 'export' / 'flame.onnx']:
            if f.exists():
                z.write(f, f.name)
    print((SELECT_DIR / 'REPORT_W4.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
