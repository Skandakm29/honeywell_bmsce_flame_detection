"""Materialize the frozen manifests on another PC without repeating the audit."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_MD5='da5e1410e13113fc9d605567aea54508'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',default='data/raw/FASDD_CV.zip')
    parser.add_argument('--data',default='data/fasdd')
    parser.add_argument('--splits',nargs='+',choices=['train','val','validation_full','test'],default=['train','val'])
    args=parser.parse_args()
    archive_path=Path(args.archive)
    if not archive_path.is_absolute():
        archive_path=ROOT/archive_path
    with archive_path.open('rb') as stream:
        actual=hashlib.file_digest(stream,'md5').hexdigest()
    if actual!=EXPECTED_MD5:
        raise ValueError(f'Wrong/incomplete archive: MD5 {actual}')
    records={}
    for split in args.splits:
        for record in json.loads((ROOT/args.data/f'{split}.json').read_text()):
            existing=records.get(record['image'])
            if existing and existing['sha256']!=record['sha256']:
                raise ValueError('Conflicting manifest image hashes')
            records[record['image']]=record
    with zipfile.ZipFile(archive_path) as archive:
        names={Path(n).name:n for n in archive.namelist() if '/images/' in n and n.lower().endswith(('.jpg','.jpeg','.png'))}
        for i,record in enumerate(records.values(),1):
            destination=(ROOT/record['image']).resolve()
            if not destination.is_relative_to(ROOT.resolve()):
                raise ValueError('Manifest path leaves project')
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==record['sha256']:
                continue
            raw=archive.read(names[destination.name])
            if hashlib.sha256(raw).hexdigest()!=record['sha256']:
                raise ValueError(f'Image differs from audited copy: {destination.name}')
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(raw)
            if i%1000==0:
                print(f'Extracted/verified {i}/{len(records)}',flush=True)
    print(f'Ready: {len(records)} unique images for {args.splits}; no re-splitting or training performed',flush=True)


if __name__=='__main__':
    main()
