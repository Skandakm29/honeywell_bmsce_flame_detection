"""Restore only hash-verified frozen manifests/initialization from the asset carrier."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', required=True)
    args = parser.parse_args()
    inventory = json.loads((ROOT / 'configs/assets.json').read_text())
    carrier = Path(args.handoff)
    with carrier.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != inventory['handoff_sha256']:
            raise ValueError('Handoff ZIP differs from the frozen baseline asset carrier')
    with zipfile.ZipFile(carrier) as archive:
        verified = {}
        for name, expected in inventory['files_sha256'].items():
            destination = (ROOT / name).resolve()
            if not destination.is_relative_to(ROOT.resolve()):
                raise ValueError('Asset path leaves project')
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Asset hash mismatch: {name}')
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError(f'Refusing to overwrite different asset: {name}')
            verified[destination] = content
        for destination, content in verified.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    print(f'Restored {len(verified)} verified assets; no bundled code/docs copied')


if __name__ == '__main__':
    main()
