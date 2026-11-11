"""Download/ingest training and evaluation sources into V9-format records.

fasdd       FASDD-CV-COCO v2 (annotation hashes checked) -> manifests_fasdd/
dfire       D-Fire (Kaggle mirror of gaiasd/DFireDataset): fire boxes kept, smoke dropped
openimages  Open Images V7 train images for fire-like nuisance labels (human-verified), fire-tagged images excluded
roboflow    heater / glowing-object images from Roboflow Universe (needs $env:ROBOFLOW_API_KEY)
onfire      ONFIRE 2023 contest videos (public Drive copy), evaluation only
bowfire     optional evaluation stills (manual download)
"""
import argparse
import concurrent.futures
import random
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common import HERE, make_record, read_json, work_dir, write_json

OI_BASE = 'https://storage.googleapis.com/openimages/'
OI_FILES = {'classes': 'v7/oidv7-class-descriptions.csv',
            'labels': 'v7/oidv7-train-annotations-human-imagelabels.csv',
            'boxes': 'v6/oidv6-train-annotations-bbox.csv'}
OI_IMAGE = 'https://open-images-dataset.s3.amazonaws.com/train/{}.jpg'
# Names verified against oidv7-class-descriptions.csv on 3 Nov 2026.
OI_INCLUDE = ['Traffic light', 'Street light', 'Lamp', 'Light bulb', 'Television', 'Computer monitor',
              'Mobile phone', 'Laptop', 'Tablet computer', 'Christmas tree', 'Pumpkin', 'Jacket', 'Flashlight',
              'Heater', 'Space heater', 'Sunset', 'Sunlight', 'Lens flare', 'Afterglow', 'Welding', 'Neon',
              'Neon sign', 'Reflection', 'Fireworks', 'Sparkler', 'Headlamp', 'Automotive lighting',
              'Christmas lights', 'Traffic cone', 'High-visibility clothing', 'Lighting']
OI_EXCLUDE = ['Fire', 'Flame', 'Bonfire', 'Campfire', 'Barbecue grill', 'Candle', 'Fireplace', 'Torch',
              'Lantern', 'Explosion', 'Wildfire', 'Smoke', 'Diya', 'Oil lamp', 'Gas stove', 'Stove']
OI_SCREENS = ['Television', 'Computer monitor', 'Mobile phone', 'Laptop', 'Tablet computer']


def fasdd(archive=None, root=None):
    from bootstrap import dataset
    from data import prepare
    work = work_dir()
    out = work / 'manifests_fasdd'
    if (out / 'prepare_report.json').exists():
        print('FASDD manifests already prepared')
        return
    # an already-extracted copy (e.g. E:\Honeywell-Lab\data from the V9 run) is used in place, read-only
    root = Path(root) if root else dataset(work / 'data' / 'fasdd', archive=archive)
    expected = read_json(HERE / 'dataset_source.json')['annotation_sha256']
    report = prepare(root, out, workers=8, expected_annotations=expected)
    counts = {k: v['images'] for k, v in report['splits'].items()}
    if counts != {'train': 63544, 'val': 15884, 'test': 15884}:
        raise RuntimeError(f'FASDD split sizes differ from V9: {counts}')


def dfire():
    import kagglehub
    import yaml
    work = work_dir()
    out = work / 'sources' / 'dfire.json'
    if out.exists():
        print('D-Fire already ingested')
        return
    root = Path(kagglehub.dataset_download('sayedgamal99/smoke-fire-detection-yolo'))
    names = None
    for spec in root.rglob('*.yaml'):
        content = yaml.safe_load(spec.read_text())
        if isinstance(content, dict) and 'names' in content:
            names = content['names']
            names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
            break
    if not names or 'fire' not in [n.lower() for n in names]:
        raise RuntimeError(f'Cannot find D-Fire class names (data.yaml) under {root}')
    fire_id = [n.lower() for n in names].index('fire')
    images = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and 'images' in p.parts)

    def one(path):
        label = Path(*[('labels' if part == 'images' else part) for part in path.parts]).with_suffix('.txt')
        picture = cv2.imread(str(path))
        if picture is None:
            return None
        h, w = picture.shape[:2]
        fire, smoke = [], 0
        if label.exists():
            for line in label.read_text().split('\n'):
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, x, y, bw, bh = int(parts[0]), *map(float, parts[1:])
                if c == fire_id:
                    fire.append([(x - bw/2) * w, (y - bh/2) * h, (x + bw/2) * w, (y + bh/2) * h])
                else:
                    smoke += 1
        category = 'dfire_fire' if fire else ('dfire_smoke' if smoke else 'dfire_none')
        return make_record(path, fire, source='dfire', category=category, group='dfire:' + path.stem)

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        records = [r for r in pool.map(one, images) if r]
    counts = {}
    for r in records:
        counts[r['category']] = counts.get(r['category'], 0) + 1
    print('D-Fire', counts, 'class names', names)
    write_json(dict(root=str(root), class_names=names, fire_class=fire_id, counts=counts, records=records), out)


def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_suffix('.part')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)


def openimages(per_class=800, seed=42):
    import pandas as pd
    work = work_dir()
    out = work / 'sources' / 'openimages.json'
    if out.exists():
        print('Open Images already ingested')
        return
    csv = work / 'data' / 'oi' / 'csv'
    for key, name in OI_FILES.items():
        print('downloading', name, flush=True)
        _download(OI_BASE + name, csv / Path(name).name)
    classes = pd.read_csv(csv / Path(OI_FILES['classes']).name)
    mid = dict(zip(classes.DisplayName, classes.LabelName))
    missing = [n for n in OI_INCLUDE + OI_EXCLUDE if n not in mid]
    if missing:
        raise RuntimeError(f'Open Images class names not found: {missing}')
    name_of = {mid[n]: n for n in OI_INCLUDE + OI_EXCLUDE}
    include, exclude = {mid[n] for n in OI_INCLUDE}, {mid[n] for n in OI_EXCLUDE}
    by_class, banned = {n: set() for n in OI_INCLUDE}, set()
    for chunk in pd.read_csv(csv / Path(OI_FILES['labels']).name, usecols=['ImageID', 'LabelName', 'Confidence'],
                             chunksize=5_000_000):
        chunk = chunk[chunk.Confidence == 1]
        banned.update(chunk.ImageID[chunk.LabelName.isin(exclude)])
        for label, ids in chunk[chunk.LabelName.isin(include)].groupby('LabelName').ImageID:
            by_class[name_of[label]].update(ids)
    screens = {}
    screen_mids = {mid[n] for n in OI_SCREENS}
    for chunk in pd.read_csv(csv / Path(OI_FILES['boxes']).name,
                             usecols=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf', 'IsDepiction'],
                             chunksize=5_000_000):
        chunk = chunk[chunk.LabelName.isin(screen_mids) & (chunk.IsGroupOf == 0) & (chunk.IsDepiction == 0)]
        chunk = chunk[(chunk.XMax - chunk.XMin) * (chunk.YMax - chunk.YMin) >= .02]
        for row in chunk.itertuples():
            screens.setdefault(row.ImageID, []).append([row.XMin, row.YMin, row.XMax, row.YMax])
    rng = random.Random(seed)
    chosen = {}
    for name in OI_INCLUDE:
        pool = sorted(by_class[name] - banned)
        if name in OI_SCREENS:  # prefer images with usable screen boxes (needed for synthetic screens)
            pool = sorted(set(pool) & set(screens)) or pool
        for image_id in rng.sample(pool, min(per_class, len(pool))):
            chosen.setdefault(image_id, name)
    print(f'Open Images: {len(chosen)} images selected; {len(banned)} fire-tagged images excluded', flush=True)
    folder = work / 'data' / 'oi' / 'img'
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        image_id, name = item
        path = folder / f'{image_id}.jpg'
        if not path.exists():
            try:
                data = urllib.request.urlopen(OI_IMAGE.format(image_id), timeout=60).read()
            except OSError:
                return None
            picture = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if picture is None:
                return None
            scale = 1024 / max(picture.shape[:2])
            if scale < 1:
                picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 92])
        record = make_record(path, [], source='openimages', category='oi:' + name, group='oi:' + image_id)
        if record and image_id in screens:
            w, h = record['width'], record['height']
            record['screens'] = [[a*w, b*h, c*w, d*h] for a, b, c, d in screens[image_id]]
        return record

    records = []
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for done, record in enumerate(pool.map(fetch, sorted(chosen.items())), 1):
            if record:
                records.append(record)
            if done % 1000 == 0:
                print(f'Open Images download {done}/{len(chosen)}', flush=True)
    counts = {}
    for r in records:
        counts[r['category']] = counts.get(r['category'], 0) + 1
    write_json(dict(per_class=per_class, counts=counts, excluded_fire_tagged=len(banned), records=records), out)


# Checked 4 Nov: hiromitsu-onoue/heater2 -> 404 (removed); terior/scene2_heater has no exportable version.
ROBOFLOW_PROJECTS = ['rex-jy68d/heater-detection', 'isik/glowing-object-detection', 'hp-cylinders/final_11-iob6n']


def roboflow():
    """Heater / glowing-object images from Roboflow Universe as candidate negatives (key from $env:ROBOFLOW_API_KEY,
    never stored). Their own boxes are ignored; some heaters may be gas heaters with real flame, so these go through
    the same V9 mining + CLIP + review as Open Images (wild source)."""
    import json as _json
    import os
    import zipfile
    key = os.environ.get('ROBOFLOW_API_KEY')
    out = work_dir() / 'sources' / 'roboflow.json'
    if out.exists():
        print('Roboflow already ingested')
        return
    if not key:
        raise RuntimeError('Set $env:ROBOFLOW_API_KEY in this PowerShell window first')
    folder = work_dir() / 'data' / 'roboflow'
    records, counts = [], {}
    for project in ROBOFLOW_PROJECTS:
        target = folder / project.replace('/', '__')
        if not target.exists():
            def get(url):   # never let the key reach a traceback or log
                try:
                    return _json.loads(urllib.request.urlopen(url, timeout=300).read())
                except Exception as error:
                    raise RuntimeError(f'Roboflow request failed for {project}: {type(error).__name__}') from None
            try:
                info = get(f'https://api.roboflow.com/{project}?api_key={key}')
            except RuntimeError as error:   # private / removed project: skip it, keep the others
                print('skipped:', error)
                continue
            versions = info.get('versions') or []
            if not versions:
                print('no public version for', project)
                continue
            version = max(int(str(v['id']).rsplit('/', 1)[-1]) for v in versions)
            import time
            for _ in range(30):   # Roboflow may still be generating the export
                link = get(f'https://api.roboflow.com/{project}/{version}/coco?api_key={key}').get('export', {}).get('link')
                if link:
                    break
                time.sleep(10)
            if not link:
                print('no export link for', project)
                continue
            archive = folder / (project.replace('/', '__') + '.zip')
            _download(link, archive)
            with zipfile.ZipFile(archive) as z:
                z.extractall(target)
            archive.unlink()
        for path in sorted(target.rglob('*')):
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                record = make_record(path, [], source='roboflow', category='rf:' + project.split('/')[1],
                                     group='rf:' + path.stem.split('.rf.')[0])
                if record:
                    records.append(record)
                    counts[record['category']] = counts.get(record['category'], 0) + 1
    print('Roboflow', counts)
    write_json(dict(projects=ROBOFLOW_PROJECTS, counts=counts, records=records), out)


ONFIRE_DRIVE = {'videos': '1tEz2wVQjPp1MjVHZLa-Z3uyVBnwljgGF', 'gt': '123AcAQCldRNE6iKpXuCaVtsaR3uHIOeN'}


def onfire():
    """ONFIRE 2023 training set (public Drive copy linked from github.com/CFG-Official/MachineLearning), TEST ONLY.
    GT per video: '<start frame>,<Fire|Smoke>' (CL1) or empty (CL0 = no fire, no smoke). Flame-only scope:
    'Fire' videos -> flame videos with start frame; CL0 -> non-fire videos; 'Smoke'-first videos are left out
    (they may or may not show flame and have no flame start time)."""
    import csv
    import zipfile
    folder = work_dir() / 'data' / 'onfire'
    table = folder / 'onfire.csv'
    if table.exists():
        print('ONFIRE already prepared:', table)
        return
    for key, file_id in ONFIRE_DRIVE.items():
        archive = folder / f'{key}.zip'
        if not archive.exists():
            print('downloading ONFIRE', key, flush=True)
            _download(f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t', archive)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                archive.unlink()
                raise RuntimeError(f'ONFIRE {key}.zip is corrupt and was deleted; rerun to download again')
            z.extractall(folder / key)
    videos = {p.stem.lower(): p for p in (folder / 'videos').rglob('*')
              if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')}
    rows, counts = [], {}
    for gt in sorted((folder / 'gt').rglob('*.rtf')):
        text = gt.read_text(encoding='latin-1')
        match = re.findall(r'(\d+)\s*,\s*([A-Za-z]+)\s*\}?\s*$', text.strip())
        positive_class = 'CL1' in str(gt.parent.name).upper()
        kind = match[0][1].lower() if match else ('unknown' if positive_class else 'none')
        counts[kind] = counts.get(kind, 0) + 1
        video = videos.get(gt.stem.lower())
        if video is None or kind not in ('fire', 'none'):
            continue
        rows.append(dict(clip=video.relative_to(folder).as_posix(), category='onfire_fire' if kind == 'fire' else 'onfire_nonfire',
                         fire_start_frame=match[0][0] if kind == 'fire' else ''))
    with open(table, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['clip', 'category', 'fire_start_frame'])
        writer.writeheader()
        writer.writerows(rows)
    print('ONFIRE annotations', counts, '| used:', sum(r['fire_start_frame'] != '' for r in rows), 'flame +',
          sum(r['fire_start_frame'] == '' for r in rows), 'non-fire videos (test only)')


def bowfire(folder):
    """Expects BoWFire images (e.g. fire*.png / not_fire*.png). Label comes from the file name."""
    records = []
    for path in sorted(Path(folder).rglob('*')):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png') or re.search(r'(gt|mask)', path.stem, re.I):
            continue
        fire = not path.stem.lower().startswith(('not', 'nofire', 'no_fire', 'non'))
        record = make_record(path, [], source='bowfire', category='fire' if fire else 'not_fire', fire=fire)
        if record:
            records.append(record)
    print('BoWFire', sum(r['fire'] for r in records), 'fire /', sum(not r['fire'] for r in records), 'not fire')
    write_json(dict(records=records), work_dir() / 'sources' / 'bowfire.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['fasdd', 'dfire', 'openimages', 'roboflow', 'onfire', 'bowfire'])
    p.add_argument('--archive', help='existing FASDD dataset.zip (skips Kaggle download)')
    p.add_argument('--root', help='existing extracted FASDD folder with annotations/ and images/ (used read-only)')
    p.add_argument('--folder')
    p.add_argument('--per-class', type=int, default=800)
    a = p.parse_args()
    if a.action == 'fasdd':
        fasdd(a.archive, a.root)
    elif a.action == 'dfire':
        dfire()
    elif a.action == 'openimages':
        openimages(a.per_class)
    elif a.action == 'onfire':
        onfire()
    elif a.action == 'roboflow':
        roboflow()
    else:
        bowfire(a.folder)
