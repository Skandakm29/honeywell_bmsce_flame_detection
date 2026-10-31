"""Colab setup/download helpers. Dataset originals and existing runs are preserved."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from model import PRETRAINED_SHA256

PRETRAINED_URL = ('https://software-dl.ti.com/jacinto7/esd/modelzoo/10_01_00/models/vision/detection/'
                  'coco/edgeai-mmdet/yolov9_s_coco_lite_640x640_20250219_checkpoint.pth')


def checksum(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def weights(destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and checksum(path) == PRETRAINED_SHA256:
        return
    temporary = path.with_suffix('.download')
    urllib.request.urlretrieve(PRETRAINED_URL, temporary)
    if checksum(temporary) != PRETRAINED_SHA256:
        raise RuntimeError('Official pretrained checkpoint checksum mismatch')
    os.replace(temporary, path)


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as stream:
        for info in stream.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Unsafe archive member: {info.filename}')
        stream.extractall(destination)


def copy_dataset(source, destination, workers=16):
    # Colab's cached Kaggle files can be on a remote filesystem. Bound concurrent
    # reads avoid paying one network round-trip per image in a serial copytree.
    source, destination = Path(source), Path(destination)
    files = []
    for folder in ('annotations', 'images'):
        for parent, _, names in os.walk(source/folder):
            parent = Path(parent)
            target = destination/parent.relative_to(source)
            target.mkdir(parents=True, exist_ok=True)
            files.extend((parent/name, target/name) for name in names)
    print(f'Copying {len(files)} dataset files with {workers} readers', flush=True)
    def copy(pair):
        origin, target = pair
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(pool.map(copy, files), 1):
            if count % 10000 == 0:
                print(f'Dataset copy: {count}/{len(files)} files', flush=True)


def dataset(destination, drive_source=None, archive=None):
    destination = Path(destination)
    marker = destination / 'source.json'
    if marker.exists() and all((destination/'annotations'/f'{s}.json').exists() for s in ('train', 'val', 'test')):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    if drive_source and (Path(drive_source)/'annotations/train.json').exists():
        source = Path(drive_source)
        copy_dataset(source, destination)
        provenance = {'source': str(source), 'method': 'copied existing teammate Drive dataset'}
    elif archive:
        extract(archive, destination)
        provenance = {'source': str(archive), 'sha256': checksum(archive), 'method': 'existing ZIP'}
    else:
        import kagglehub
        source = Path(kagglehub.dataset_download('yuulind/fasdd-cv-coco/versions/2'))
        copy_dataset(source, destination)
        provenance = {'source': str(source), 'dataset': 'yuulind/fasdd-cv-coco',
                      'method': 'kagglehub; downloaded version is recorded in cache path'}
    marker.write_text(json.dumps(provenance, indent=2))
    return destination


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['weights', 'dataset'])
    p.add_argument('--destination', required=True)
    p.add_argument('--drive-source')
    p.add_argument('--archive')
    a = p.parse_args()
    if a.action == 'weights':
        weights(a.destination)
    else:
        dataset(a.destination, a.drive_source, a.archive)
