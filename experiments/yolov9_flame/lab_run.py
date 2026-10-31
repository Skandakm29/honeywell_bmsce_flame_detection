"""Single-process lab orchestration. No credentials or cloud mounts are needed."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    for relative, expected in json.loads((root/'package_manifest.json').read_text()).items():
        actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Package source changed: {relative}; re-extract the original ZIP.')
    workflow, ti, output = root/'workflow', root/'tensorlab', root/'outputs'
    output.mkdir(exist_ok=True)
    # The OS releases this lock even if the terminal crashes. A second launcher
    # cannot create a competing training process for this directory.
    lock = (root/'training.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('Another training launcher is already running here.')
        # Keep this system awake only for this process; no power settings changed.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['KAGGLEHUB_CACHE'] = str(root/'.runtime'/'kaggle-cache')
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    with (root/'run.log').open('a', encoding='utf-8', buffering=1) as log:
        def report(text):
            print(text, flush=True)
            log.write(text+'\n')

        def run(name, *arguments):
            report(f'\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] {name}')
            command = [sys.executable, '-u', *map(str, arguments)]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
                for line in process.stdout:
                    report(line.rstrip())
                if process.wait():
                    raise RuntimeError(f'{name} failed (exit {process.returncode}); see run.log')

        try:
            run('Verify CUDA', '-c',
                'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
                'print(torch.__version__, torch.cuda.get_device_name(), '
                'torch.cuda.get_device_properties(0).total_memory)')
            weights, data, manifests = root/'weights.pth', root/'data', output/'manifests'
            run('Verified official weights', workflow/'bootstrap.py', 'weights', '--destination', weights)
            run('Pinned dataset download', workflow/'bootstrap.py', 'dataset', '--destination', data)
            run('Dataset audit', workflow/'data.py', '--root', data, '--output', manifests)
            common = ['--tensorlab', ti, '--checkpoint', weights]
            run('Engineering tests', workflow/'test_workflow.py', *common, '--export')
            run('GPU preflight', workflow/'preflight.py', *common,
                '--manifests', manifests, '--output', output/'preflight')
            config = output/'preflight'/'selected_config.json'
            speed_path = output/'speed.json'
            run('Real-data GPU timing including EMA', workflow/'speed_check.py', *common,
                '--config', config, '--manifests', manifests, '--output', speed_path)
            cfg, speed = json.loads(config.read_text()), json.loads(speed_path.read_text())
            records = json.loads((manifests/'train.json').read_text())
            images = len(records)*cfg['epochs'] + min(len(records), cfg['pilot_images'])*cfg['pilot_epochs']*len(cfg['pilot_lrs'])
            hours = images/speed['images_per_second']/3600
            report(f'ROUGH compute-only estimate for the complete fresh recipe: {hours:.1f} hours. '
                   'Add download/setup, validation, checkpoint writes and export; '
                   'first complete epoch is needed for a stronger estimate. '
                   'This is not a morning-completion guarantee.')
            run('LR trials and full training (automatic matching-checkpoint resume)',
                workflow/'train.py', *common, '--config', config,
                '--manifests', manifests, '--output', output, '--tune')
            best = output/'full'/'best.pth'
            evaluation = ['--tensorlab', ti, '--checkpoint', best, '--manifests', manifests,
                          '--batch', str(cfg['batch']), '--workers', str(cfg['workers'])]
            run('Full validation and trained ONNX export/parity', workflow/'finish.py',
                *evaluation, '--output', output/'validation')
            run('Final test after model selection', workflow/'finish.py', *evaluation,
                '--split', 'test', '--final-test', '--skip-export', '--output', output/'test')
            report('SUCCESS: training, full validation, final test and ONNX export/parity all completed.')
            # Archive outputs AND source, never the data/cache/environment. CRC-test
            # the archive and record SHA256 before suggesting any cleanup.
            archive = root/'Honeywell-results.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
                for folder in (output, workflow, ti):
                    for path in sorted(folder.rglob('*')):
                        if path.is_file() and '__pycache__' not in path.parts:
                            bundle.write(path, path.relative_to(root))
                bundle.write(root/'run.log', 'run.log')
                for name in ('lab_setup.ps1', 'START_TRAINING.cmd', 'START_HERE.md', 'package_manifest.json'):
                    if (root/name).exists():
                        bundle.write(root/name, name)
            with zipfile.ZipFile(archive) as bundle:
                if bundle.testzip() is not None:
                    raise RuntimeError('Results archive integrity check failed; preserve all original outputs.')
            with archive.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            (root/'Honeywell-results.sha256').write_text(digest+'  Honeywell-results.zip\n')
            (output/'SUCCESS.json').write_text(json.dumps(dict(complete=True, archive=str(archive), sha256=digest), indent=2))
            report(f'Verified results archive: {archive}\nSHA256: {digest}\n'
                   'Copy the ZIP and SHA256 file to your own machine/Drive. Keep all lab files until that copy is verified.')
        except Exception as error:
            report(f'FAILED: {type(error).__name__}: {error}')
            raise
        finally:
            if os.name == 'nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.close()


if __name__ == '__main__':
    main()
