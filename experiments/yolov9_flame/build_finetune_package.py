"""Build private lab handoff with the verified trained weights and original manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from build_lab_package import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensorlab', required=True)
    p.add_argument('--backup', required=True, help='Verified original lab backup directory')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    backup = Path(a.backup)
    weights = (backup/'outputs/full/best.pth').read_bytes()
    expected = json.loads((here/'finetune_config.json').read_text())['initial_checkpoint_sha256']
    if hashlib.sha256(weights).hexdigest() != expected: raise RuntimeError('Wrong initial trained weights')
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'source.zip'
        build(a.tensorlab, path)
        with zipfile.ZipFile(path) as z:
            files = {n.removeprefix('Honeywell-Lab/'): z.read(n) for n in z.namelist()}
    files.pop('package_manifest.json')
    files['initial_best.pth'] = weights
    for name in ('train', 'val', 'validation_full', 'test'):
        files[f'original_manifests/{name}.json'] = (backup/f'outputs/manifests/{name}.json').read_bytes()
    # Same pinned isolated Python/CUDA installation; change only the launch target.
    setup = files['lab_setup.ps1'].decode('utf-8-sig').replace('workflow\\lab_run.py', 'workflow\\fine_tune.py')
    files['lab_setup.ps1'] = setup.encode('utf-8')
    files['START_HERE.md'] = (here/'FINETUNE_START.md').read_bytes()
    files['package_manifest.json'] = json.dumps({n: hashlib.sha256(b).hexdigest() for n,b in files.items()}, indent=2).encode()
    destination = Path(a.output); destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr('Honeywell-Refine/'+name,data)
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise RuntimeError('Package integrity failed')
    print(destination, destination.stat().st_size, hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == '__main__':main()
