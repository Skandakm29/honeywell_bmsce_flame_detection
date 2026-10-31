"""Produce a self-contained Windows source ZIP; no credentials/data/weights."""
import hashlib
import json
from pathlib import Path
import zipfile


def build(tensorlab, destination):
    here = Path(__file__).resolve().parent
    vendor = json.loads((here/'vendor.json').read_text())
    sources = {}
    for path in sorted(here.iterdir()):
        if path.suffix in ('.py', '.json', '.txt'):
            sources['workflow/'+path.name] = path.read_bytes().replace(b'\r\n', b'\n')
    for name in ('lab_setup.ps1', 'START_TRAINING.cmd', 'START_HERE.md'):
        sources[name] = (here/name).read_bytes()
    for relative, expected in vendor['files'].items():
        data = (Path(tensorlab)/'edgeai-mmdetection'/'mmdet'/relative).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'TI source checksum mismatch: {relative}')
        sources['tensorlab/edgeai-mmdetection/mmdet/'+relative] = data
    sources['tensorlab/LICENSE'] = (Path(tensorlab)/'edgeai-mmdetection/LICENSE').read_bytes()
    sources['package_manifest.json'] = json.dumps({name: hashlib.sha256(data).hexdigest()
                                                for name, data in sources.items()}, indent=2).encode()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sources.items():
            archive.writestr('Honeywell-Lab/'+name, data)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    print(destination, destination.stat().st_size, hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorlab', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.tensorlab, args.output)
