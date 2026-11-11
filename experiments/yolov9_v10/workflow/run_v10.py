"""Single entry point for the lab PC. Every step is resumable; finished steps are skipped.

  0  preflight            GPU, disk, package hashes, V9 weights, 1 training step, CLIP + downloads reachable
  1  data                 FASDD, D-Fire, Open Images, own clips (+MIVIA/BoWFire if given), synthetic screens
  2  mine                 V9 hard-negative mining -> outbox/step2_review.zip                     STOP 1
  2b finalize-review      needs inbox/review_answers.csv (pulled from GitHub)
  3  manifests            dedup + manifests_v10 -> outbox/step3_manifest_report.zip             STOP 2
  4  baseline             V9 benchmarks, temporal layer, diagnosis
  5  train                V10 fine-tune (overnight; rerun the same command to resume)
  6  results              selection + V10 benchmarks/temporal/diagnosis -> outbox/step6_results.zip  STOP 3
  7  test                 needs inbox/APPROVE_TEST.txt: FASDD test once for V9 and the selected model
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from common import HERE, PACKAGE, V9_SHA256, read_json, sha256, work_dir, write_json

STEPS = ['0', '1', '2', '2b', '3', '4', '5', '6', '7']


def py(script, *args):
    command = [sys.executable, '-u', str(HERE / script), *map(str, args)]
    print('\n>>', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def preflight(a, work):
    import torch
    manifest = read_json(PACKAGE / 'package_manifest.json')
    bad = [name for name, digest in manifest.items() if sha256(PACKAGE / name) != digest]
    if bad:
        raise RuntimeError(f'Package files changed: {bad}. Re-pull from GitHub; do not edit package files.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not visible to PyTorch')
    gpu = torch.cuda.get_device_name(0)
    free = shutil.disk_usage(work).free / 1e9
    if free < 120:
        raise RuntimeError(f'Only {free:.0f} GB free at {work}; need >= 120 GB')
    if not a.v9 or sha256(a.v9) != V9_SHA256:
        raise RuntimeError('Pass --v9 <path to the original V9 best.pth> (SHA256 48d8e654...)')
    shutil.copyfile(a.v9, work / 'v9_best.pth')
    from model import Detector
    model = Detector(str(PACKAGE / 'tensorlab')).cuda().train()
    images = torch.rand(2, 3, 640, 640, device='cuda')
    target = torch.full((2, 1, 5), -1., device='cuda')
    target[:, :, 1:] = 0
    target[0, 0] = torch.tensor([0., 100., 100., 300., 300.])
    with torch.autocast('cuda', dtype=torch.float16):
        loss = sum(model(images, target).values())
    loss.backward()
    import kagglehub, onnxruntime, open_clip, sklearn, pandas, yaml  # noqa: F401 (dependency check)
    from common import Clip
    Clip()  # downloads CLIP weights once
    import urllib.request
    urllib.request.urlopen('https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv', timeout=30).read()
    write_json(dict(gpu=gpu, free_gb=free, loss=float(loss), torch=torch.__version__), work / 'status' / 'preflight.json')
    print(f'Preflight OK: {gpu}, {free:.0f} GB free, one training step loss {float(loss):.3f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=STEPS)
    p.add_argument('--v9', help='step 0: original V9 best.pth')
    p.add_argument('--fasdd-zip', help='step 1: existing FASDD dataset.zip (else Kaggle download)')
    p.add_argument('--fasdd-root', help='step 1: existing extracted FASDD folder, e.g. E:/Honeywell-Lab/data')
    p.add_argument('--clips', help='step 1: folder with recorded clips + clips.csv')
    p.add_argument('--mivia', help='step 1: optional MIVIA folder with mivia.csv')
    p.add_argument('--onfire', help='step 1: optional ONFIRE folder with onfire.csv (test only)')
    p.add_argument('--bowfire', help='step 1: optional BoWFire image folder')
    p.add_argument('--per-class', type=int, default=800)
    p.add_argument('--download-only', action='store_true',
                   help='step 1: only download FASDD, D-Fire and Open Images (no clips needed); rerun step 1 with --clips later')
    p.add_argument('--force', action='store_true', help='rerun a step that is marked done')
    a = p.parse_args()
    work = work_dir()
    status = work / 'status'
    status.mkdir(exist_ok=True)
    done = status / f'step_{a.step}.done'
    if done.exists() and not a.force:
        print(f'Step {a.step} already done ({done.read_text().strip()}); use --force to rerun.')
        return
    previous = STEPS[STEPS.index(a.step) - 1] if a.step != '0' else None
    if previous and not (status / f'step_{previous}.done').exists():
        raise RuntimeError(f'Run step {previous} first')
    started = time.time()
    v9 = work / 'v9_best.pth'
    if a.step == '0':
        preflight(a, work)
    elif a.step == '1':
        py('sources.py', 'fasdd', *(['--archive', a.fasdd_zip] if a.fasdd_zip else []),
           *(['--root', a.fasdd_root] if a.fasdd_root else []))
        py('sources.py', 'dfire')
        py('sources.py', 'openimages', '--per-class', a.per_class)
        py('sources.py', 'onfire')                      # test-only videos; always fetched
        if os.environ.get('ROBOFLOW_API_KEY'):
            py('sources.py', 'roboflow')                # heater / glowing objects (optional, needs key)
        if a.download_only:
            print('Downloads finished. Step 1 is NOT marked done: rerun it with --clips once the clips are here.')
            return
        if not a.clips:
            raise RuntimeError('Pass --clips <folder with recorded clips and clips.csv>')
        onfire = a.onfire or str(work / 'data' / 'onfire')
        py('frames.py', '--clips', a.clips, *(['--mivia', a.mivia] if a.mivia else []), '--onfire', onfire)
        if a.bowfire:
            py('sources.py', 'bowfire', '--folder', a.bowfire)
        py('screens.py')
    elif a.step == '2':
        py('mine.py', 'run', '--checkpoint', v9)
    elif a.step == '2b':
        py('mine.py', 'finalize')
    elif a.step == '3':
        if not (work / 'dedup' / 'dedup.json').exists():
            py('dedup.py')
        py('negcheck.py')
        py('build_manifest.py')
    elif a.step == '4':
        py('bench.py', '--name', 'v9', '--checkpoint', v9)
        py('temporal.py', '--model', 'v9')
        py('diagnose.py', '--name', 'v9', '--checkpoint', v9)
    elif a.step == '5':
        from train_v10 import train
        config = json.loads((HERE / 'config_v10.json').read_text())
        train(config, str(PACKAGE / 'tensorlab'), str(v9), str(work / 'manifests_v10'), str(work / 'train_v10'))
    elif a.step == '6':
        py('select_v10.py')
        selected = work / 'select' / 'selected.pth'
        py('bench.py', '--name', 'v10', '--checkpoint', selected)
        py('temporal.py', '--model', 'v10')
        py('diagnose.py', '--name', 'v10', '--checkpoint', selected)
        py('report.py')
    elif a.step == '7':
        if not (PACKAGE / 'inbox' / 'APPROVE_TEST.txt').exists():
            raise RuntimeError('FASDD test runs only after approval: inbox/APPROVE_TEST.txt must be pulled from GitHub')
        py('bench.py', '--name', 'v9', '--checkpoint', v9, '--test')
        py('bench.py', '--name', 'v10', '--checkpoint', work / 'select' / 'selected.pth', '--test')
    done.write_text(time.strftime('%Y-%m-%d %H:%M:%S') + f' ({(time.time() - started) / 60:.1f} min)\n')
    print(f'\nStep {a.step} finished in {(time.time() - started) / 60:.1f} min.')


if __name__ == '__main__':
    main()
