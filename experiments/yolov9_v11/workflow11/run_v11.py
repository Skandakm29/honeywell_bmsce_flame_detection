"""V11 entry point on the lab PC. Every step is resumable; finished steps are skipped (use --force to redo).

  0  preflight   package + V10 code hashes, GPU, disk, V10 work files, frozen V10, COCO weights, 1 GPU training step
  1  labelfix    unboxed flames in positive training images -> outbox/v11_step1_labelfix.zip          GATE A
  1b dense      propose boxes every 0.33 s in the 4 train small-flame clips -> outbox/v11_step1b_dense.zip  GATE B
  1c dense-ok   needs inbox/dense_review.csv (every proposal ok|bad) -> dense_v11/extra.json
  2  build       needs inbox/labelfix_spotcheck.json -> manifests_v11, flame bank, augmentation preview
  3  pilot       200 real batches: timing + plumbing (nothing written to train_v11)
  4  train       40 epochs from COCO weights (~16-20 h; rerun the same command to resume)
  5  results     selection (frozen rule) + ONNX export + bench/temporal/diagnose v11 + autopsy + report   STOP
  6  test        FASDD test once for V11 (after selection is final; never rerun)
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V10_PACKAGE = Path(os.environ.get('V10_PACKAGE', r'D:\Honeywell-V10'))
V10_FLOW = V10_PACKAGE / 'workflow'
sys.path[:0] = [str(HERE), str(V10_FLOW)]
os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE), str(V10_FLOW), os.environ.get('PYTHONPATH', '')])

from common import read_json, work_dir, write_json  # noqa: E402  (V10 code, read-only)

STEPS = ['0', '1', '1b', '1c', '2', '3', '4', '5', '6']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def py(script, *args, folder=HERE):
    command = [sys.executable, '-u', str(folder / script), *map(str, args)]
    print('\n>>', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def check_packages():
    bad = [n for n, d in read_json(HERE.parent / 'package_manifest.json').items() if sha(HERE.parent / n) != d]
    bad += ['V10:' + n for n, d in read_json(V10_PACKAGE / 'package_manifest.json').items() if sha(V10_PACKAGE / n) != d]
    if bad:
        raise RuntimeError(f'Package files changed: {bad}. Re-copy the package; never edit package files.')


def preflight(work):
    import torch
    from paths11 import COCO_WEIGHTS, V10_FROZEN, V10_FROZEN_SHA
    check_packages()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not visible to PyTorch')
    free = shutil.disk_usage(work).free / 1e9
    if free < 60:
        raise RuntimeError(f'Only {free:.0f} GB free at {work}; need >= 60 GB (40 epoch checkpoints + caches)')
    need = [work / 'manifests_v10' / n for n in ('train.json', 'val.json', 'validation_full.json', 'hn_val.json')]
    need += [work / 'bench' / n / 'bench.json' for n in ('v9', 'v10')] + [work / 'sources' / 'videos.json']
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise RuntimeError(f'Missing V10 work files: {missing}')
    if not V10_FROZEN.exists() or sha(V10_FROZEN) != V10_FROZEN_SHA:
        raise RuntimeError(f'Frozen V10 checkpoint missing or changed: {V10_FROZEN} (set $env:V11_V10_CHECKPOINT)')
    from bootstrap import weights
    weights(COCO_WEIGHTS)                                    # downloads the official TI COCO file, SHA256 checked
    inputs = {str(p.relative_to(work)): sha(p) for p in need[:4]}
    write_json(dict(inputs=inputs, v10_frozen=V10_FROZEN_SHA, coco=sha(COCO_WEIGHTS), free_gb=free,
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__), work / 'status_v11' / 'preflight.json')
    from model import Detector, load_pretrained
    from common import PACKAGE
    model = Detector(str(PACKAGE / 'tensorlab')).cuda().train()
    load_pretrained(model, COCO_WEIGHTS)
    images = torch.rand(2, 3, 640, 640, device='cuda')
    target = torch.full((2, 1, 5), -1., device='cuda')
    target[:, :, 1:] = 0
    target[0, 0] = torch.tensor([0., 100., 100., 108., 112.])     # a small box
    with torch.autocast('cuda', dtype=torch.float16):
        loss = sum(model(images, target).values())
    loss.backward()
    print(f'Preflight OK: {torch.cuda.get_device_name(0)}, {free:.0f} GB free, loss {float(loss):.3f}')


def verify_inputs(work):
    """Every later step: the V10 manifests V11 depends on must be exactly what preflight recorded."""
    recorded = read_json(work / 'status_v11' / 'preflight.json')['inputs']
    changed = [n for n, d in recorded.items() if sha(work / n) != d]
    if changed:
        raise RuntimeError(f'V10 inputs changed since preflight: {changed}. Stop and report; do not continue.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=STEPS)
    p.add_argument('--extra', help='step 2: optional folder with extra.json (small-flame frames, training locations)')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    work = work_dir()
    status = work / 'status_v11'
    status.mkdir(exist_ok=True)
    done = status / f'step_{a.step}.done'
    if done.exists() and not a.force:
        print(f'Step {a.step} already done ({done.read_text().strip()}); use --force to rerun.')
        return
    previous = STEPS[STEPS.index(a.step) - 1] if a.step != '0' else None
    if previous and not (status / f'step_{previous}.done').exists():
        raise RuntimeError(f'Run step {previous} first')
    if a.step != '0':
        check_packages()
        verify_inputs(work)
    started = time.time()
    if a.step == '0':
        preflight(work)
    elif a.step == '1':
        py('labelfix.py')
    elif a.step == '1b':
        py('dense.py', 'propose')
    elif a.step == '1c':
        py('dense.py', 'finalize')
    elif a.step == '2':
        py('build_v11.py', '--extra', work / 'dense_v11', *(['--extra', a.extra] if a.extra else []))
    elif a.step == '3':
        py('train_v11.py', 'pilot')
    elif a.step == '4':
        py('train_v11.py', 'train')
        summary = work / 'train_v11' / 'training_summary.json'
        if not summary.exists() or read_json(summary).get('stop_reason') != 'epoch_or_patience_limit':
            raise RuntimeError('Training stopped before epoch 40 (time budget reached). STOP and report to the owner; '
                               'do not edit the config or delete files.')
    elif a.step == '5':
        py('select_v11.py')
        best = work / 'select_v11' / 'v11_best.pth'
        py('bench.py', '--name', 'v11', '--checkpoint', best, folder=V10_FLOW)
        py('temporal.py', '--model', 'v11', folder=V10_FLOW)
        py('diagnose.py', '--name', 'v11', '--checkpoint', best, folder=V10_FLOW)
        py('autopsy11.py')
        py('report_v11.py')
    elif a.step == '6':
        # Owner decision 6 Nov: run end to end. The selection in step 5 is final before the test is read;
        # the test is run exactly once and never used to change anything.
        if (work / 'bench' / 'v11' / 'test_summary.json').exists():
            raise RuntimeError('FASDD test for V11 already run once; never rerun it')
        py('bench.py', '--name', 'v11', '--checkpoint', work / 'select_v11' / 'v11_best.pth', '--test', folder=V10_FLOW)
    done.write_text(time.strftime('%Y-%m-%d %H:%M:%S') + f' ({(time.time() - started) / 60:.1f} min)\n')
    print(f'\nV11 step {a.step} finished in {(time.time() - started) / 60:.1f} min.')


if __name__ == '__main__':
    main()
