"""Evaluate a saved checkpoint on a frozen validation manifest and collect failures for review.

The held-out test split is deliberately not selectable: it stays unused during tuning
(docs/PROJECT_STATUS.md). Matching follows train.operating_metrics: one-to-one,
score-ordered, IoU 0.5, single flame class.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from flame_model import FlameDetector, ROOT
from train import Flames, collate, configure_worker, detections, operating_metrics

# Flame size as sqrt(box area) in model-input pixels, i.e. after the letterbox resize.
# COCO's small/medium/large use original-image pixels, which mislead for 10k-pixel sources.
SIZE_BUCKETS = [('<8px',0,8),('8-16px',8,16),('16-32px',16,32),('32-64px',32,64),('>=64px',64,math.inf)]


def size_bucket(side):
    return next(name for name,low,high in SIZE_BUCKETS if low<=side<high)


def iou(xywh,box):
    x,y,w,h = xywh
    a,b,c,d = box
    intersection = max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b))
    union = w*h+(c-a)*(d-b)-intersection
    return intersection/union if union>0 else 0


def match(boxes,candidates,threshold):
    """Same rule as train.operating_metrics; also returns which boxes/predictions failed."""
    matched = [False]*len(boxes)
    outcomes = []
    for prediction in sorted((p for p in candidates if p['score']>=threshold),key=lambda p:-p['score']):
        best_iou,best_index = 0,None
        for index,box in enumerate(boxes):
            if matched[index]:
                continue
            overlap = iou(prediction['bbox'],box)
            if overlap>best_iou:
                best_iou,best_index = overlap,index
        hit = best_index is not None and best_iou>=.5
        if hit:
            matched[best_index] = True
        outcomes.append((prediction,hit))
    return matched,outcomes


def false_detection_kind(prediction,boxes):
    if not boxes:
        return 'negative_image'
    overlap = max(iou(prediction['bbox'],box) for box in boxes)
    if overlap>=.5:
        return 'duplicate'
    if overlap>=.1:
        return 'poor_localisation'
    return 'background_on_flame_image'


def coco_metrics(records,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    images,annotations = [],[]
    for i,record in enumerate(records):
        images.append({'id':i,'width':record['width'],'height':record['height']})
        for x,y,x2,y2 in record['boxes']:
            annotations.append({'id':len(annotations)+1,'image_id':i,'category_id':1,
                                'bbox':[x,y,x2-x,y2-y],'area':(x2-x)*(y2-y),'iscrowd':0})
    gt.dataset = {'info':{},'images':images,'annotations':annotations,'categories':[{'id':1,'name':'flame'}]}
    gt.createIndex()
    if not predictions:
        return {'AP50_95':0.0,'AP50':0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation = COCOeval(gt,gt.loadRes(predictions),'bbox')
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    names = ['AP50_95','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100']
    return {name:float(value) for name,value in zip(names,evaluation.stats)}


def contact_sheets(items,records,predictions_by_image,confidence,stem,directory,per_sheet=25,columns=5):
    """Green=label, red=prediction at the operating confidence, yellow=the reviewed failure."""
    for page in range(0,len(items),per_sheet):
        tiles = []
        for item in items[page:page+per_sheet]:
            record = records[item['image_index']]
            image = cv2.imread(str(ROOT/record['image']))
            thickness = max(2,round(max(image.shape[:2])/300))
            for box in record['boxes']:
                cv2.rectangle(image,tuple(map(int,box[:2])),tuple(map(int,box[2:])),(0,255,0),thickness)
            for prediction in predictions_by_image[item['image_index']]:
                if prediction['score']>=confidence:
                    x,y,w,h = map(int,prediction['bbox'])
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),thickness)
            x,y,x2,y2 = map(int,item['box_xyxy'])
            cv2.rectangle(image,(x,y),(x2,y2),(0,255,255),thickness*2)
            ratio = min(320/image.shape[1],240/image.shape[0])
            resized = cv2.resize(image,(max(1,int(image.shape[1]*ratio)),max(1,int(image.shape[0]*ratio))))
            tile = np.zeros((260,320,3),np.uint8)
            tile[20:20+resized.shape[0],:resized.shape[1]] = resized
            cv2.putText(tile,item['caption'],(4,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            tiles.append(tile)
        while len(tiles)%columns:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)]
        cv2.imwrite(str(directory/f'{stem}_{page//per_sheet+1:02d}.jpg'),np.vstack(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True,help='best.pth (or last.pth) from a training run')
    parser.add_argument('--data',default='data/fasdd')
    parser.add_argument('--split',choices=['val','validation_full'],default='validation_full',
                        help='Held-out test is deliberately not selectable during tuning')
    parser.add_argument('--output',required=True)
    parser.add_argument('--size',type=int,default=416,help='Must match the training run')
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--confidence',type=float,default=.4,help='Operating point for the failure breakdown')
    parser.add_argument('--review',type=int,default=100,help='Misses and false detections sampled for human review')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    output = ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = ROOT/args.checkpoint
    model = FlameDetector()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    model.to(device).eval()
    manifest = ROOT/args.data/f'{args.split}.json'
    dataset = Flames(manifest,args.size)
    records = dataset.records
    worker_options = {'multiprocessing_context':'spawn'} if args.workers else {}
    loader = DataLoader(dataset,batch_size=args.batch,collate_fn=collate,num_workers=args.workers,
                        worker_init_fn=configure_worker,pin_memory=device.type=='cuda',**worker_options)
    predictions = []
    start = time.perf_counter()
    with torch.inference_mode():
        for done,(images,_,indices) in enumerate(loader,1):
            outputs = model(images.to(device)).cpu()
            for prediction,index in zip(outputs,indices):
                boxes,scores = detections(prediction,records[index],dataset.size)
                for (x,y,x2,y2),score in zip(boxes.tolist(),scores.tolist()):
                    predictions.append({'image_id':index,'category_id':1,'bbox':[x,y,x2-x,y2-y],'score':score})
            if done%50==0 or done==len(loader):
                elapsed = time.perf_counter()-start
                print(f'batch {done}/{len(loader)}, eta_min={elapsed/done*(len(loader)-done)/60:.1f}',flush=True)
    metrics = coco_metrics(records,predictions)
    print(json.dumps(metrics),flush=True)
    by_image = [[] for _ in records]
    for prediction in predictions:
        by_image[prediction['image_id']].append(prediction)
    sizes = {name:{'boxes':0,'detected':0} for name,_,_ in SIZE_BUCKETS}
    kinds = {}
    misses,false_detections = [],[]
    for index,record in enumerate(records):
        ratio = min(args.size/record['height'],args.size/record['width'])
        matched,outcomes = match(record['boxes'],by_image[index],args.confidence)
        for box,hit in zip(record['boxes'],matched):
            side = math.sqrt((box[2]-box[0])*(box[3]-box[1]))*ratio
            bucket = size_bucket(side)
            sizes[bucket]['boxes'] += 1
            sizes[bucket]['detected'] += hit
            if not hit:
                misses.append({'image_index':index,'image':record['image'],'box_xyxy':box,
                               'input_side_px':round(side,1),'caption':f'miss {bucket} #{index}'})
        for prediction,hit in outcomes:
            if hit:
                continue
            kind = false_detection_kind(prediction,record['boxes'])
            kinds[kind] = kinds.get(kind,0)+1
            x,y,w,h = prediction['bbox']
            false_detections.append({'image_index':index,'image':record['image'],'box_xyxy':[x,y,x+w,y+h],
                                     'score':round(prediction['score'],3),'kind':kind,
                                     'caption':f'{kind} {prediction["score"]:.2f} #{index}'})
    for row in sizes.values():
        row['recall'] = row['detected']/row['boxes'] if row['boxes'] else None
    sampler = random.Random(42)
    review_misses = sampler.sample(misses,min(args.review,len(misses)))
    review_false = sampler.sample(false_detections,min(args.review,len(false_detections)))
    contact_sheets(review_misses,records,by_image,args.confidence,'review_misses',output)
    contact_sheets(review_false,records,by_image,args.confidence,'review_false_detections',output)
    summary = {
        'checkpoint':str(args.checkpoint),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'split':args.split,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'size':args.size,'images':len(records),'flame_boxes':sum(len(r['boxes']) for r in records),
        'negative_images':sum(not r['boxes'] for r in records),'device':str(device),
        'coco':metrics,
        'coco_area_note':'AP_small/medium/large use original-image pixel areas; see recall_by_input_size for model-input scale',
        'at_confidence':{'confidence':args.confidence,'misses':len(misses),'false_detections':len(false_detections),
                         'false_detections_by_kind':kinds,'recall_by_input_size':sizes},
        'threshold_sweep':operating_metrics(records,predictions),
        'evaluation_seconds':time.perf_counter()-start}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'review_misses.json').write_text(json.dumps(review_misses,indent=2))
    (output/'review_false_detections.json').write_text(json.dumps(review_false,indent=2))
    # Keep the dump reviewable in size; AP above used every prediction down to 0.001.
    (output/'predictions_ge_0.05.json').write_text(json.dumps([p for p in predictions if p['score']>=.05]))
    print(json.dumps({k:summary[k] for k in ('coco','at_confidence')},indent=2),flush=True)


if __name__=='__main__':
    main()
