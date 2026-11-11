"""Week-4 entry point on the lab PC. Every step is resumable; finished steps are skipped (use --force to redo).

  0  preflight  GPU, disk, V10/V12 code present, source manifests, start checkpoint checksum
  1  build      manifests_w4 + preview_val.jpg (GATE A: a person looks at the preview before step 2)
  2  train      flame -> flame + obstruction fine-tune, 10 epochs (rerun the same command to resume)
  3  select     frozen selection rule + ONNX export + report -> outbox/w4_results.zip
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_W4 = HERE.parent
V10_PACKAGE = Path(os.environ.get('V10_PACKAGE', PACKAGE_W4.parent / 'yolov9_v10'))
V12_PACKAGE = Path(os.environ.get('V12_PACKAGE', PACKAGE_W4.parent / 'yolov9_v12'))
PATHS = [str(HERE), str(V10_PACKAGE / 'workflow'), str(V12_PACKAGE / 'workflow12')]
sys.path[:0] = PATHS
os.environ['PYTHONPATH'] = os.pathsep.join(PATHS + [os.environ.get('PYTHONPATH', '')])

from common import write_json  # noqa: E402  (V10 code, read-only)
from common_w4 import CONFIG, MANIFESTS_W4, SOURCE_MANIFESTS, START, START_SHA256, WORK  # noqa: E402

STEPS = ['0', '1', '2', '3']


def py(script, *args):
    command = [sys.executable, '-u', str(HERE / script), *map(str, args)]
    print('\n>>', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def preflight():
    import torch
    from data import sha256
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not visible to PyTorch')
    for path in (V10_PACKAGE / 'workflow' / 'train_v10.py', V10_PACKAGE / 'tensorlab', V12_PACKAGE / 'workflow12' / 'data11.py'):
        if not path.exists():
            raise RuntimeError(f'missing {path} (set V10_PACKAGE / V12_PACKAGE)')
    missing = [n for n in ('train.json', 'val.json', 'validation_full.json', 'hn_val.json', 'test.json')
               if not (SOURCE_MANIFESTS / n).exists()]
    if missing:
        raise RuntimeError(f'missing source manifests in {SOURCE_MANIFESTS}: {missing}')
    if not START.exists() or sha256(START) != START_SHA256:
        raise RuntimeError(f'start checkpoint missing or checksum differs: {START} (expected {START_SHA256[:12]}...)')
    free = shutil.disk_usage(WORK).free / 1e9
    if free < 30:
        raise RuntimeError(f'only {free:.0f} GB free at {WORK}; need >= 30 GB (12 checkpoints + ONNX)')
    write_json(dict(start=str(START), start_sha256=START_SHA256, source_manifests=str(SOURCE_MANIFESTS),
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__, free_gb=free,
                    config=CONFIG), WORK / 'status_w4' / 'preflight.json')
    print(f'Preflight OK: {torch.cuda.get_device_name(0)}, start {START.name} ({START_SHA256[:12]}), {free:.0f} GB free')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=STEPS)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    status = WORK / 'status_w4'
    status.mkdir(parents=True, exist_ok=True)
    done = status / f'step_{a.step}.done'
    if done.exists() and not a.force:
        print(f'Step {a.step} already done ({done.read_text().strip()}); use --force to rerun.')
        return
    if a.step != '0':
        previous = STEPS[STEPS.index(a.step) - 1]
        if not (status / f'step_{previous}.done').exists():
            raise RuntimeError(f'Run step {previous} first')
    if a.step == '2' and not (status / 'gate_a.txt').exists():
        raise RuntimeError(f'GATE A: look at {MANIFESTS_W4 / "preview_val.jpg"}. If the obstructions and boxes look right, '
                           f'write your name and the date into {status / "gate_a.txt"}, then rerun step 2.')
    started = time.time()
    if a.step == '0':
        preflight()
    elif a.step == '1':
        py('build_w4.py')
    elif a.step == '2':
        py('train_w4.py')
    elif a.step == '3':
        py('select_w4.py')
    done.write_text(time.strftime('%Y-%m-%d %H:%M:%S') + f' ({(time.time() - started) / 60:.1f} min)\n')
    print(f'\nW4 step {a.step} finished in {(time.time() - started) / 60:.1f} min.')


if __name__ == '__main__':
    main()
