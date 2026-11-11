"""V12 entry point on the lab PC. Every step is resumable; finished steps are skipped (use --force to redo).

  0   preflight   package + V10 code hashes, GPU, disk, V11 finished, manifests_v11, frozen V10
  1   opcheck     size-aware threshold on frozen V10 (validation only, ~30-45 min)
  2   train       3 fine-tunes from V10, 8 epochs each: a_data, b_zoom, c_zoom_paste (~9-11 h; rerun resumes)
  3   results     selection (frozen rule) + ONNX + bench / temporal / diagnose for v12 + report     -> outbox/v12_results.zip
  4a  rec-propose (any time after step 0, also during step 2) needs D:\\Honeywell-V10-work\\recordings_v12 (owner)  -> outbox/v12_step4a_recordings.zip   GATE C
  4b  rec-build   needs inbox/rec_review.csv -> manifests_v12, rec_v12/distance_test.json
  4c  rec-train   fine-tune 4 epochs from the step-3 start model on manifests_v12 (~1.5-2 h)
  4d  rec-select  selection among the start model + 4 epochs; bench/temporal for v12r if it changed
  5   test        FASDD test + distance test ONCE (after 4d, or after 3 if the owner wrote inbox/skip_recordings.txt)
"""
import argparse
import hashlib
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

STEPS = ['0', '1', '2', '3', '4a', '4b', '4c', '4d', '5']
VARIANTS = ['a_data', 'b_zoom', 'c_zoom_paste']
INPUTS = ['manifests_v11/' + n for n in ('train.json', 'val.json', 'validation_full.json', 'hn_val.json', 'test.json',
                                         'flame_bank.npz')]


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
        raise RuntimeError(f'Package files changed: {bad}. Bug fixes need workflow12/rehash.py; never edit rules/config.')


def preflight(work):
    import torch
    from common12 import V10_FROZEN, V10_FROZEN_SHA
    check_packages()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not visible to PyTorch')
    if not (work / 'status_v11' / 'step_6.done').exists():
        raise RuntimeError('V11 has not finished (status_v11/step_6.done missing). Wait for V11; do not run both at once.')
    free = shutil.disk_usage(work).free / 1e9
    if free < 80:
        raise RuntimeError(f'Only {free:.0f} GB free at {work}; need >= 80 GB (32 checkpoints + caches + frames)')
    missing = [n for n in INPUTS if not (work / n).exists()]
    if missing:
        raise RuntimeError(f'Missing V11 data: {missing}')
    for n in ('val.json', 'validation_full.json', 'hn_val.json', 'test.json'):
        if sha(work / 'manifests_v11' / n) != sha(work / 'manifests_v10' / n):
            raise RuntimeError(f'manifests_v11/{n} differs from manifests_v10: evaluation sets must be identical to V10')
    if not V10_FROZEN.exists() or sha(V10_FROZEN) != V10_FROZEN_SHA:
        raise RuntimeError(f'Frozen V10 checkpoint missing or changed: {V10_FROZEN}')
    write_json(dict(inputs={n: sha(work / n) for n in INPUTS}, v10_frozen=V10_FROZEN_SHA, free_gb=free,
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__), work / 'status_v12' / 'preflight.json')
    print(f'Preflight OK: {torch.cuda.get_device_name(0)}, {free:.0f} GB free')


def verify_inputs(work):
    recorded = read_json(work / 'status_v12' / 'preflight.json')['inputs']
    changed = [n for n, d in recorded.items() if sha(work / n) != d]
    if changed:
        raise RuntimeError(f'V11 inputs changed since preflight: {changed}. Stop and report; do not continue.')


def results(name, checkpoint):
    py('bench.py', '--name', name, '--checkpoint', checkpoint, folder=V10_FLOW)
    py('temporal.py', '--model', name, folder=V10_FLOW)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=STEPS)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    work = work_dir()
    status = work / 'status_v12'
    status.mkdir(exist_ok=True)
    done = status / f'step_{a.step}.done'
    if done.exists() and not a.force:
        print(f'Step {a.step} already done ({done.read_text().strip()}); use --force to rerun.')
        return
    if a.step == '5':
        if (work / 'test_v12' / 'test.json').exists():
            raise RuntimeError('V12 test already run once; never rerun it')
        ready = (status / 'step_4d.done').exists()
        from common12 import INBOX
        if not ready and not ((status / 'step_3.done').exists() and (INBOX / 'skip_recordings.txt').exists()):
            raise RuntimeError('Step 5 needs step 4d, or step 3 plus inbox/skip_recordings.txt written by the OWNER')
    elif a.step != '0':
        # 4a / 4b (recordings) only need step 0, so they can run while step 2 trains; 4c needs both 3 and 4b.
        needs = {'4a': ['0'], '4b': ['4a'], '4c': ['3', '4b']}.get(a.step, [STEPS[STEPS.index(a.step) - 1]])
        for previous in needs:
            if not (status / f'step_{previous}.done').exists():
                raise RuntimeError(f'Run step {previous} first')
    if a.step != '0':
        check_packages()
        verify_inputs(work)
    started = time.time()
    if a.step == '0':
        preflight(work)
    elif a.step == '1':
        py('opcheck12.py')
    elif a.step == '2':
        for v in VARIANTS:
            py('train_v12.py', 'variant', v)
    elif a.step == '3':
        py('select12.py', 'variants')
        best = work / 'select_v12' / 'best.pth'
        results('v12', best)
        py('diagnose.py', '--name', 'v12', '--checkpoint', best, folder=V10_FLOW)
        py('report12.py')
    elif a.step == '4a':
        py('record12.py', 'propose')
    elif a.step == '4b':
        py('record12.py', 'finalize')
    elif a.step == '4c':
        py('train_v12.py', 'rec')
    elif a.step == '4d':
        py('select12.py', 'rec')
        if sha(work / 'select_v12_rec' / 'best.pth') != sha(work / 'select_v12' / 'best.pth'):
            results('v12r', work / 'select_v12_rec' / 'best.pth')
        py('report12.py')
    elif a.step == '5':
        py('test12.py')
        py('report12.py')
    done.write_text(time.strftime('%Y-%m-%d %H:%M:%S') + f' ({(time.time() - started) / 60:.1f} min)\n')
    print(f'\nV12 step {a.step} finished in {(time.time() - started) / 60:.1f} min.')


if __name__ == '__main__':
    main()
