"""Build manifests from the Kaggle FASDD-CV-COCO copy, keeping its own train/val/test splits.

Reads <root>/annotations/{train,val,test}.json and <root>/images/; never writes there.
Only 'fire' boxes become flame targets. Smoke annotations are ignored, so smoke-only images
and images without annotations are negatives (no boxes). Records use the format read by
scripts/train.py: absolute image path, width, height, [x1,y1,x2,y2] boxes in original pixels.

Images are quarantined (reported, left out of the manifest) when the file is missing, its
header size differs from the annotation, it has an EXIF orientation tag and cv2 (which applies
valid rotations) decodes it at a different size, which train.Flames would reject, or a box
fails the baseline audit rule (validated_boxes).
Exact duplicates across splits are reported, not removed.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_local import validated_boxes  # noqa: E402

SPLITS = ('train','val','test')
COCO_AREAS = [('small',0,32**2),('medium',32**2,96**2),('large',96**2,math.inf)]
INPUT_SIDES = [('<8px',0,8),('8-16px',8,16),('16-32px',16,32),('32-64px',32,64),('>=64px',64,math.inf)]


def bucket(value,table):
    return next(name for name,low,high in table if low<=value<high)


def inspect(task,hash_files):
    image,boxes_or_error,path = task
    if path is None:
        return None,'file not found'
    if isinstance(boxes_or_error,Exception):
        return None,f'invalid box: {boxes_or_error}'
    try:
        with Image.open(path) as picture:
            size = picture.size
            orientation = picture.getexif().get(0x0112,1)
    except Exception as error:
        return None,f'unreadable header: {error}'
    if size != (image['width'],image['height']):
        return None,f'header size {size} differs from annotation'
    record = {'image':str(path),'width':image['width'],'height':image['height'],
              'boxes':boxes_or_error,'source_image_id':image['id']}
    if orientation not in (None,1):
        # cv2.imread applies valid EXIF rotations (2-8); keep the image only if cv2 decodes it
        # at the annotated size, which is what train.Flames requires.
        decoded = cv2.imread(str(path))
        shape = None if decoded is None else decoded.shape[:2]
        if shape != (image['height'],image['width']):
            return None,f'EXIF orientation {orientation}: cv2 decodes {shape}, annotation {(image["height"],image["width"])}'
        record['exif_orientation_tag'] = orientation
    if hash_files:
        record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record,None


def statistics(records,smoke_ids,sizes=(416,640)):
    stats = {'images':len(records),'flame_images':sum(bool(r['boxes']) for r in records),
             'flame_boxes':sum(len(r['boxes']) for r in records)}
    stats['negative_images'] = stats['images']-stats['flame_images']
    stats['smoke_only_images'] = sum(1 for r in records if not r['boxes'] and r['source_image_id'] in smoke_ids)
    stats['no_annotation_images'] = stats['negative_images']-stats['smoke_only_images']
    stats['flame_with_smoke_images'] = sum(1 for r in records if r['boxes'] and r['source_image_id'] in smoke_ids)
    areas = collections.Counter()
    sides = {size:collections.Counter() for size in sizes}
    for r in records:
        for x1,y1,x2,y2 in r['boxes']:
            area = (x2-x1)*(y2-y1)
            areas[bucket(area,COCO_AREAS)] += 1
            for size in sizes:
                ratio = min(size/r['height'],size/r['width'])
                sides[size][bucket(math.sqrt(area)*ratio,INPUT_SIDES)] += 1
    stats['flame_boxes_by_coco_area'] = {name:areas[name] for name,_,_ in COCO_AREAS}
    stats['flame_boxes_by_input_side'] = {str(size):{name:sides[size][name] for name,_,_ in INPUT_SIDES} for size in sizes}
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='/kaggle/input/fasdd-cv-coco')
    parser.add_argument('--output',required=True)
    parser.add_argument('--monitor-images',type=int,default=2000,help='Fixed random subset of val checked every epoch')
    parser.add_argument('--smoke-images',type=int,default=64)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--skip-hash',action='store_true',help='Skip SHA-256 (no duplicate report)')
    args = parser.parse_args()
    root,output = Path(args.root),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    files = collections.defaultdict(list)
    for path in (root/'images').rglob('*'):
        if path.is_file():
            files[path.name].append(path)
    report = {'root':str(root),'image_files':sum(len(v) for v in files.values()),
              'duplicate_file_names':sum(len(v)>1 for v in files.values()),
              'policy':{'target':'fire annotations only, renamed flame',
                        'smoke':'smoke annotations ignored; smoke-only images are negatives',
                        'splits':'Kaggle annotations/{train,val,test}.json used unchanged',
                        'quarantine':'missing file, header size mismatch, EXIF rotation, or box failing validated_boxes'},
              'splits':{},'quarantined':[]}
    all_records = {}
    for split in SPLITS:
        coco = json.loads((root/'annotations'/f'{split}.json').read_text())
        names = {c['id']:c['name'].strip().lower() for c in coco['categories']}
        fire = [i for i,n in names.items() if n in ('fire','flame')]
        smoke = {i for i,n in names.items() if n=='smoke'}
        if len(fire)!=1 or set(names)-set(fire)-smoke:
            raise ValueError(f'{split}: unexpected categories {coco["categories"]}')
        ids = [image['id'] for image in coco['images']]
        if len(set(ids))!=len(ids):
            raise ValueError(f'{split}: duplicate image ids')
        known = set(ids)
        fire_annotations = collections.defaultdict(list)
        smoke_ids = set()
        orphans = collections.Counter()
        for annotation in coco['annotations']:
            if annotation['image_id'] not in known:
                orphans[names[annotation['category_id']]] += 1
            elif annotation['category_id']==fire[0]:
                fire_annotations[annotation['image_id']].append(annotation)
            else:
                smoke_ids.add(annotation['image_id'])
        tasks = []
        outside_split_folder = 0
        for image in coco['images']:
            name = Path(image['file_name']).name
            expected = root/'images'/split/image['file_name']
            if expected.is_file():
                path = expected
            elif len(files.get(name,[]))==1:
                path = files[name][0]
                outside_split_folder += 1
            else:
                path = None
            try:
                boxes = validated_boxes(image,fire_annotations[image['id']])
            except ValueError as error:
                boxes = error
            tasks.append((image,boxes,path))
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda task:inspect(task,not args.skip_hash),tasks))
        records = []
        for (image,_,_),(record,reason) in zip(tasks,results):
            if reason:
                report['quarantined'].append({'split':split,'file_name':image['file_name'],'reason':reason})
            else:
                records.append(record)
        all_records[split] = records
        stats = statistics(records,smoke_ids)
        stats.update({'annotation_images':len(coco['images']),'annotations':len(coco['annotations']),
                      'fire_annotations':sum(len(v) for v in fire_annotations.values()),
                      'smoke_annotations':len(coco['annotations'])-sum(len(v) for v in fire_annotations.values())-sum(orphans.values()),
                      'categories':coco['categories'],'orphan_annotations':dict(orphans),
                      'quarantined':len(coco['images'])-len(records),
                      'kept_with_exif_orientation_tag':sum('exif_orientation_tag' in r for r in records),
                      'files_found_outside_split_folder':outside_split_folder})
        report['splits'][split] = stats
        print(f'{split}: kept {len(records)}/{len(coco["images"])}',flush=True)
    if not args.skip_hash:
        owners = collections.defaultdict(set)
        for split,records in all_records.items():
            for record in records:
                owners[record['sha256']].add(split)
        pairs = collections.Counter('+'.join(sorted(s)) for s in owners.values() if len(s)>1)
        within = {split:len(records)-len({r['sha256'] for r in records}) for split,records in all_records.items()}
        report['exact_duplicates'] = {'cross_split_hashes':dict(pairs),'within_split_extra_copies':within,
                                      'action':'reported only; Kaggle splits unchanged'}
    monitor = random.Random(42).sample(all_records['val'],min(args.monitor_images,len(all_records['val'])))
    manifests = {'train.json':all_records['train'],'validation_full.json':all_records['val'],
                 'val.json':monitor,'test.json':all_records['test']}
    for name,records in manifests.items():
        (output/name).write_text(json.dumps(records))
    report['monitor_val'] = statistics(monitor,set())|{'note':'seed-42 random subset of val; smoke counts not tracked'}
    smoke = output/'smoke'
    smoke.mkdir(exist_ok=True)
    sampler = random.Random(7)
    positives = [r for r in all_records['train'] if r['boxes']]
    negatives = [r for r in all_records['train'] if not r['boxes']]
    half = args.smoke_images//2
    (smoke/'train.json').write_text(json.dumps(sampler.sample(positives,min(half,len(positives)))+
                                               sampler.sample(negatives,min(half,len(negatives)))))
    (smoke/'val.json').write_text(json.dumps(sampler.sample(all_records['val'],min(half,len(all_records['val'])))))
    report['manifest_sha256'] = {name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in manifests}
    (output/'prepare_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='quarantined'}|{'quarantined_total':len(report['quarantined'])},indent=2))


if __name__=='__main__':
    main()
