"""Verify the official archive, decode images, and build leakage-screened manifests.

Uses the original split membership. Test/validation take precedence when a
duplicate crosses splits. Perceptual screening is conservative, not proof of
independent scene/video sources (the archive does not provide that provenance).
"""
import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import math
import random
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MD5 = 'da5e1410e13113fc9d605567aea54508'


def image_fingerprint(image):
    gray = ImageOps.grayscale(image)
    small = list(gray.resize((9,8),Image.Resampling.BILINEAR).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value<<1) | (small[y*9+x] > small[y*9+x+1])
    thumbnail = bytes(gray.resize((16,16),Image.Resampling.BILINEAR).getdata())
    return value,thumbnail


class OverlapIndex:
    def __init__(self):
        self.exact = {}
        self.bands = collections.defaultdict(list)

    @staticmethod
    def keys(value):
        # Five disjoint bands cover 64 bits. Distance <=4 shares a band.
        for start,width in [(0,13),(13,13),(26,13),(39,13),(52,12)]:
            yield start,(value>>start)&((1<<width)-1)

    def match(self,record,split):
        if record['sha256'] in self.exact:
            return 'exact',self.exact[record['sha256']]
        candidates = {}
        for key in self.keys(record['dhash']):
            for other in self.bands[key]:
                if other['split'] != split:
                    candidates[other['image']] = other
        for other in candidates.values():
            ratio = (record['width']/record['height'])/(other['width']/other['height'])
            if .95 <= ratio <= 1.05 and (record['dhash']^other['dhash']).bit_count() <= 4:
                difference = sum(abs(a-b) for a,b in zip(record['thumbnail'],other['thumbnail']))/256
                if difference <= 5:
                    return 'near',other
        return None,None

    def add(self,record,split):
        stored = {k:record[k] for k in ['image','width','height','sha256','dhash','thumbnail']}
        stored['split'] = split
        self.exact[record['sha256']] = stored
        for key in self.keys(record['dhash']):
            self.bands[key].append(stored)


def annotations_for(coco):
    fire_ids = {c['id'] for c in coco['categories'] if c['name'].lower() in ('fire','flame')}
    if len(fire_ids) != 1:
        raise ValueError(f'Unexpected classes: {coco["categories"]}')
    annotations = collections.defaultdict(list)
    for annotation in coco['annotations']:
        if annotation['category_id'] in fire_ids:
            annotations[annotation['image_id']].append(annotation)
    return annotations


def validated_boxes(image,annotations):
    boxes = []
    width,height = image['width'],image['height']
    for annotation in annotations:
        x,y,w,h = annotation['bbox']
        if not all(math.isfinite(v) for v in (x,y,w,h)):
            raise ValueError('Nonfinite box')
        if annotation.get('iscrowd',0):
            raise ValueError('Crowd annotation needs dedicated handling')
        if min(x,y) < 0 or min(w,h) <= 0 or x+w > width+1 or y+h > height+1:
            raise ValueError('Out-of-bounds or nonpositive box')
        # Allow only the documented <=1-pixel rounding overflow.
        boxes.append([x,y,min(x+w,width),min(y+h,height)])
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive',default='data/raw/FASDD_CV.zip')
    parser.add_argument('--output',default='data/fasdd')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--monitor-images',type=int,default=2000)
    args = parser.parse_args()
    archive_path = ROOT/args.archive
    output = ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    print('Verifying full archive MD5',flush=True)
    digest = hashlib.file_digest(archive_path.open('rb'),'md5').hexdigest()
    if digest != EXPECTED_MD5:
        raise ValueError(f'Archive MD5 mismatch: {digest}')
    report = {'archive':args.archive,'md5':digest,'archive_bytes':archive_path.stat().st_size,
              'archive_verified':True,'split_policy':'original memberships; deduplicate with test > val > train priority',
              'near_duplicate_rule':'dHash Hamming <=4 AND gray thumbnail MAE <=5/255 AND aspect ratio within 5%',
              'source_independence':'not established; source/video provenance missing',
              'seed':42,'invalid':[],'duplicates':[],'splits':{}}
    (ROOT/'reports/archive_verified.json').write_text(json.dumps({k:report[k] for k in ['archive','md5','archive_bytes','archive_verified']},indent=2))
    start = time.perf_counter()
    index = OverlapIndex()
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for name in archive.namelist():
            if '/images/' in name and name.lower().endswith(('.jpg','.jpeg','.png')):
                filename = Path(name).name
                if filename in members:
                    raise ValueError(f'Duplicate archive filename: {filename}')
                members[filename] = name
        report['archive_image_count'] = len(members)
        cache_path = output/'decoded_index.jsonl'
        decoded_cache = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                item = json.loads(line)
                decoded_cache[item['image']] = item

        def decode(task):
            image,boxes = task
            filename = Path(image['file_name']).name
            if filename not in members:
                raise ValueError(f'Missing image: {filename}')
            relative = (output/'images'/filename).relative_to(ROOT).as_posix()
            destination = ROOT/relative
            cached = decoded_cache.get(relative)
            if cached and destination.exists() and cached.get('archive_md5') == digest:
                return cached
            raw = archive.read(members[filename])  # CRC checked by ZipFile.
            with Image.open(io.BytesIO(raw)) as picture:
                if picture.size != (image['width'],image['height']):
                    raise ValueError('Decoded dimensions differ from annotation')
                picture.load()  # Decode pixels, not only the header.
                dhash,thumbnail = image_fingerprint(picture)
            destination.write_bytes(raw)
            return {'image':relative,'source_image_id':image['id'],'width':image['width'],
                    'height':image['height'],'boxes':boxes,'sha256':hashlib.sha256(raw).hexdigest(),
                    'dhash':dhash,'thumbnail_hex':thumbnail.hex(),'archive_md5':digest}

        (output/'images').mkdir(exist_ok=True)
        with cache_path.open('a',encoding='utf8') as cache_file:
            for split in ['test','val','train']:
                member = next(n for n in archive.namelist() if '/COCO_CV/' in n and n.endswith('/'+split+'.json'))
                coco = json.loads(archive.read(member))
                annotations = annotations_for(coco)
                tasks = []
                for image in coco['images']:
                    try:
                        tasks.append((image,validated_boxes(image,annotations[image['id']])))
                    except Exception as error:
                        report['invalid'].append({'split':split,'image':image['file_name'],'reason':str(error)})
                records = []
                counters = collections.Counter()
                def guarded(task):
                    try:
                        return decode(task),None
                    except Exception as error:
                        return None,{'split':split,'image':task[0]['file_name'],'reason':str(error)}
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    # Bounded batches prevent futures holding the entire dataset.
                    for base in range(0,len(tasks),128):
                        for item,error in pool.map(guarded,tasks[base:base+128]):
                            if error:
                                report['invalid'].append(error)
                                continue
                            if item['image'] not in decoded_cache:
                                cache_file.write(json.dumps(item)+'\n')
                            record = dict(item)
                            record['thumbnail'] = bytes.fromhex(record['thumbnail_hex'])
                            kind,other = index.match(record,split)
                            if kind:
                                counters[kind]+=1
                                report['duplicates'].append({'kind':kind,'removed_split':split,
                                    'removed_image':record['image'],'retained_split':other['split'],'retained_image':other['image']})
                                continue
                            index.add(record,split)
                            records.append({k:record[k] for k in ['image','source_image_id','width','height','boxes','sha256']})
                        cache_file.flush()
                        if base % 1024 == 0 or base+128 >= len(tasks):
                            print(f'{split}: audited {min(base+128,len(tasks))}/{len(tasks)}, kept={len(records)}, duplicates={dict(counters)}',flush=True)
                name = 'validation_full' if split=='val' else split
                (output/f'{name}.json').write_text(json.dumps(records,indent=2))
                report['splits'][split] = {'original':len(coco['images']),'kept':len(records),
                    'flame_images':sum(bool(r['boxes']) for r in records),
                    'flame_boxes':sum(len(r['boxes']) for r in records),'duplicates_removed':dict(counters)}
                if split=='val':
                    monitor = random.Random(42).sample(records,min(args.monitor_images,len(records)))
                    (output/'val.json').write_text(json.dumps(monitor,indent=2))
                    report['monitor_validation_images'] = len(monitor)
                (output/'audit.json').write_text(json.dumps(report,indent=2))
    report['audit_seconds'] = time.perf_counter()-start
    report['remaining'] = ['human label review','source/video provenance','industrial negative footage','external/video/hardware evaluation']
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('invalid','duplicates')},indent=2),flush=True)


if __name__=='__main__':
    main()
