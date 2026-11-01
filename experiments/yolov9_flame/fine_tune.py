"""Isolated A5000 refinement: unchanged data, strict initialization, no LR pilots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def canonical(records):
    # Relocating the dataset is allowed; image bytes, order, IDs and boxes are not changed.
    return [{k: v for k, v in r.items() if k != 'image'} for r in records]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    from data import sha256, Flames
    from train import atomic_json, train
    for name, expected in json.loads((root/'package_manifest.json').read_text()).items():
        if sha256(root/name) != expected:
            raise RuntimeError(f'Package checksum mismatch: {name}')
    lock = (root/'training.lock').open('a+b')
    if not lock.tell():
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    if os.name == 'nt':
        import ctypes
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        run(root)
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


def run(root):
    import torch
    from bootstrap import dataset
    from data import prepare, sha256, Flames
    from train import atomic_json, train
    from model import Detector
    from finish import export
    if not torch.cuda.is_available() or 'A5000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This verified batch configuration requires the NVIDIA RTX A5000; no CPU fallback.')
    workflow, ti = root/'workflow', root/'tensorlab'
    output = root/'outputs'; output.mkdir(exist_ok=True)
    cfg = json.loads((workflow/'finetune_config.json').read_text())
    weights = root/'initial_best.pth'
    if sha256(weights) != cfg['initial_checkpoint_sha256']:
        raise RuntimeError('Wrong trained checkpoint')
    print('Checking short-run loading, original-model fallback and interrupted recovery.', flush=True)
    subprocess.run([sys.executable, '-u', str(workflow/'test_finetune.py')], check=True)
    os.environ['KAGGLEHUB_CACHE'] = str(root/'.runtime/kaggle-cache')
    manifests = output/'manifests'
    if not (output/'data_verified.json').exists():
        dataset(root/'data', archive=root/'dataset.zip' if (root/'dataset.zip').exists() else None)
        expected = json.loads((workflow/'dataset_source.json').read_text())['annotation_sha256']
        prepare(root/'data', manifests, workers=8, expected_annotations=expected)
        for name in ('train', 'val', 'validation_full', 'test'):
            old = json.loads((root/'original_manifests'/f'{name}.json').read_text())
            new = json.loads((manifests/f'{name}.json').read_text())
            if canonical(old) != canonical(new):
                raise RuntimeError(f'{name} image bytes/order/labels differ from the original run')
        atomic_json({name: sha256(manifests/f'{name}.json') for name in
                    ('train', 'val', 'validation_full', 'test')}, output/'data_verified.json')
    else:
        for name, expected in json.loads((output/'data_verified.json').read_text()).items():
            if sha256(manifests/f'{name}.json') != expected:
                raise RuntimeError('Prepared manifest changed after verification')
    print('Data verified unchanged. Starting at trained epoch20; at most 3 NEW epochs, no mosaic/LR pilots.', flush=True)
    train(cfg, ti, weights, manifests, output/'full')

    def evaluate_once(checkpoint, destination):
        report = destination/'metrics.json'
        if report.exists():
            m = json.loads(report.read_text())
            if m['checkpoint_sha256'] == sha256(checkpoint) and m['manifest_sha256'] == sha256(manifests/'validation_full.json'):
                return m
        subprocess.run([sys.executable, '-u', str(workflow/'finish.py'), '--tensorlab', str(ti),
                        '--checkpoint', str(checkpoint), '--manifests', str(manifests),
                        '--batch', str(cfg['batch']), '--workers', str(cfg['workers']),
                        '--output', str(destination), '--skip-export'], check=True)
        return json.loads(report.read_text())

    initial = output/'full/initial.pth'
    candidate = output/'full/best.pth'
    base_metrics = evaluate_once(initial, output/'baseline_validation')
    candidate_metrics = (base_metrics if sha256(initial) == sha256(candidate) else
                         evaluate_once(candidate, output/'candidate_validation'))
    # Full validation selects the candidate; the original remains the conservative fallback.
    improved = candidate_metrics['coco']['AP50_95'] > base_metrics['coco']['AP50_95']
    selected = candidate if improved else initial
    selection = dict(rule='strictly higher full-validation AP50:95; ties retain original',
                     candidate_improved_AP=improved, selected_checkpoint=str(selected),
                     selected_sha256=sha256(selected), baseline=base_metrics, candidate=candidate_metrics,
                     caveat='Check precision/recall/small-flame and independent video regressions; AP alone is not deployment acceptance.')
    atomic_json(selection, output/'comparison.json')
    state = torch.load(selected, map_location='cpu', weights_only=False)
    model = Detector(ti).eval(); model.load_state_dict(state['model'], strict=True)
    validation = Flames(json.loads((manifests/'validation_full.json').read_text()), cfg['size'])
    export(model, validation, output/'selected_export')
    print('Full validation and ONNX parity completed. Original/candidate metrics saved. Test split not used for selection.', flush=True)
    archive = root/'Honeywell-finetune-results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for folder in (output, workflow, ti):
            for path in sorted(folder.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    z.write(path, path.relative_to(root))
        for name in ('package_manifest.json', 'initial_best.pth', 'setup.log'):
            if (root/name).is_file(): z.write(root/name, name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise RuntimeError('Result ZIP integrity failed; keep originals.')
    digest = sha256(archive)
    (root/'Honeywell-finetune-results.sha256').write_text(digest+'  '+archive.name+'\n')
    atomic_json(dict(complete=True, archive=str(archive), sha256=digest,
                     note='Validation/export complete; independent video/test and board checks pending'), output/'SUCCESS.json')
    print(f'Copy and verify {archive} and its SHA256 file before any cleanup.', flush=True)


if __name__ == '__main__':
    main()
