"""Single-class flame evaluation for the baseline (--arch nano) or YOLOX-S-ti-lite (--arch s).

Definitions (written into metrics.json as well):
- Predictions: model output -> score = objectness x class score -> keep score >= 0.001 ->
  NMS IoU 0.65 -> at most 100 per image (train.detections, unchanged from the baseline).
- AP: pycocotools COCOeval on boxes, maxDets 100. COCO small/medium/large use the original-image
  box area (< 32^2, < 96^2, rest).
- Matching for every threshold-dependent metric: per image, predictions at or above the
  confidence threshold in descending score order; each takes the unmatched ground-truth box with
  the highest IoU and is a TP if that IoU >= 0.5, otherwise an FP; a ground-truth box matches at
  most once. This is train.operating_metrics' rule. Because it is greedy in score order, the
  matches above any threshold are a prefix of the full run, so one pass serves every threshold.
- Negative image: no flame box (includes smoke-only images). Image-level false alarm: a negative
  image with at least one prediction at or above the threshold.
- Mean matched IoU: over TP predictions only.

Thresholds are selected on validation (maximum F1), or fixed in advance with --confidence (e.g. 0.40,
the baseline's reported operating point; not tuned on any split). The test split requires
--final-test and either --threshold-from <validation metrics.json> or --confidence; no threshold is
chosen on test. The confidence sweep and curves are still produced on test for reporting only.
Every split also reports the fixed confidences 0.10/0.25/0.40/0.50/0.75.
- Latency: batch 1, float32, CUDA-synchronised; 'inference' is the GPU-only network time,
  'end_to_end' adds decode, letterbox, transfer and postprocessing (NMS). Throughput is also
  measured GPU-only at batch 16 on random input.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

EXP = Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
from s_model import ROOT, build  # noqa: E402
from train import Flames, collate, configure_worker, detections, preprocess  # noqa: E402
from evaluate import coco_metrics, contact_sheets, false_detection_kind, iou  # noqa: E402

GRID = [round(float(t),2) for t in np.arange(.01,1.0,.01)]  # plain floats: numpy scalars break json
COCO_AREAS = [('small',0,32**2),('medium',32**2,96**2),('large',96**2,math.inf)]
FIXED_THRESHOLDS = (.10,.25,.40,.50,.75)
RECALL_TARGETS = (.5,.6,.7,.8,.9)
PRECISION_TARGETS = (.5,.6,.7,.8,.9,.95)


def area_class(box):
    area = (box[2]-box[0])*(box[3]-box[1])
    return next(name for name,low,high in COCO_AREAS if low<=area<high)


def greedy(boxes,candidates):
    """Full score-ordered matching. Returns per-prediction (prediction,hit,iou) and, per box,
    None or (score,iou) of the prediction that matched it."""
    matched = [None]*len(boxes)
    outcomes = []
    for prediction in sorted(candidates,key=lambda p:-p['score']):
        best_iou,best_index = 0,None
        for index,box in enumerate(boxes):
            if matched[index] is not None:
                continue
            overlap = iou(prediction['bbox'],box)
            if overlap>best_iou:
                best_iou,best_index = overlap,index
        hit = best_index is not None and best_iou>=.5
        if hit:
            matched[best_index] = (prediction['score'],best_iou)
        outcomes.append((prediction,hit,best_iou))
    return outcomes,matched


def at_threshold(records,matches,threshold):
    """Every threshold-dependent count, read off the single greedy pass."""
    tp=fp=fn=fp_negative=fp_positive=0
    negative=negative_alarm=positive=positive_detected=positive_any=0
    ious = []
    for record,(outcomes,matched) in zip(records,matches):
        kept = [(p,hit,overlap) for p,hit,overlap in outcomes if p['score']>=threshold]
        hits = [overlap for _,hit,overlap in kept if hit]
        misses = len(kept)-len(hits)
        tp += len(hits)
        fp += misses
        ious += hits
        fn += sum(match is None or match[0]<threshold for match in matched)
        if record['boxes']:
            positive += 1
            positive_detected += bool(hits)
            positive_any += bool(kept)
            fp_positive += misses
        else:
            negative += 1
            negative_alarm += bool(kept)
            fp_negative += misses
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {'confidence':threshold,'TP':tp,'FP':fp,'FN':fn,'precision':precision,'recall':recall,
            'F1':2*precision*recall/(precision+recall) if precision+recall else 0.0,
            'missed_flame_rate':fn/(tp+fn) if tp+fn else None,
            'mean_matched_iou':float(np.mean(ious)) if ious else None,
            'images':len(records),'flame_images':positive,'negative_images':negative,
            'negative_images_with_false_alarm':negative_alarm,
            'image_level_false_positive_rate':negative_alarm/negative if negative else None,
            'false_positive_boxes_on_negative_images':fp_negative,
            'false_positive_boxes_on_flame_images':fp_positive,
            'false_alarms_per_image':fp/len(records) if records else None,
            'false_alarms_per_negative_image':fp_negative/negative if negative else None,
            'flame_images_with_a_true_positive':positive_detected,
            'flame_images_with_any_detection':positive_any,
            'flame_images_fully_missed':positive-positive_detected}


def detail(records,matches,threshold):
    """Size, IoU and error breakdowns at the operating threshold."""
    by_area = {name:{'boxes':0,'detected':0,'matched_ious':[]} for name,_,_ in COCO_AREAS}
    kinds = {}
    ious = []
    failures = {'misses':[],'false_detections':[],'correct_detections':[]}
    for index,(record,(outcomes,matched)) in enumerate(zip(records,matches)):
        for p,hit,overlap in outcomes:
            if p['score']>=threshold:
                if hit:
                    ious.append(overlap)
                    x,y,w,h = p['bbox']
                    failures['correct_detections'].append({'image_index':index,'image':record['image'],
                        'box_xyxy':[x,y,x+w,y+h],'score':round(p['score'],3),'iou':round(overlap,3),
                        'caption':f'TP {p["score"]:.2f} IoU {overlap:.2f} #{index}'})
                else:
                    kind = false_detection_kind(p,record['boxes'])
                    kinds[kind] = kinds.get(kind,0)+1
                    x,y,w,h = p['bbox']
                    failures['false_detections'].append({'image_index':index,'image':record['image'],
                        'box_xyxy':[x,y,x+w,y+h],'score':round(p['score'],3),'kind':kind,
                        'caption':f'{kind} {p["score"]:.2f} #{index}'})
        for b,box in enumerate(record['boxes']):
            name = area_class(box)
            by_area[name]['boxes'] += 1
            hit = matched[b] is not None and matched[b][0]>=threshold
            by_area[name]['detected'] += hit
            if hit:
                by_area[name]['matched_ious'].append(matched[b][1])
            if not hit:
                failures['misses'].append({'image_index':index,'image':record['image'],'box_xyxy':box,
                    'coco_area':name,'caption':f'miss {name} #{index}'})
    for row in by_area.values():
        row['recall'] = row['detected']/row['boxes'] if row['boxes'] else None
        values = row.pop('matched_ious')
        row['mean_matched_iou'] = float(np.mean(values)) if values else None
    histogram,edges = np.histogram(ious,bins=np.arange(.5,1.0001,.05))
    return {'recall_by_coco_area':by_area,'false_detections_by_kind':kinds,
            'matched_iou_histogram':{f'{edges[i]:.2f}-{edges[i+1]:.2f}':int(c) for i,c in enumerate(histogram)}},failures


def targets(sweep):
    best_precision = {}
    for target in RECALL_TARGETS:
        rows = [r for r in sweep if r['recall']>=target]
        best = max(rows,key=lambda r:r['precision']) if rows else None
        best_precision[str(target)] = {'precision':best['precision'],'confidence':best['confidence']} if best else None
    best_recall = {}
    for target in PRECISION_TARGETS:
        rows = [r for r in sweep if r['precision']>=target and r['TP']>0]
        best = max(rows,key=lambda r:r['recall']) if rows else None
        best_recall[str(target)] = {'recall':best['recall'],'confidence':best['confidence']} if best else None
    return {'precision_at_recall':best_precision,'recall_at_precision':best_recall,
            'note':'best value over the 0.01-0.99 confidence grid; None if unreachable'}


def plots(sweep,operating,image_confusion,directory,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = [r['confidence'] for r in sweep]
    figure,axes = plt.subplots(2,2,figsize=(11,9))
    axes[0,0].plot([r['recall'] for r in sweep],[r['precision'] for r in sweep])
    axes[0,0].scatter([operating['recall']],[operating['precision']],color='red',zorder=3,label=f'conf {operating["confidence"]}')
    axes[0,0].set(xlabel='Recall',ylabel='Precision',title='Precision-recall (IoU 0.5, confidence sweep)',xlim=(0,1),ylim=(0,1))
    axes[0,0].legend()
    axes[0,1].plot(t,[r['F1'] for r in sweep],label='F1')
    axes[0,1].plot(t,[r['precision'] for r in sweep],label='Precision')
    axes[0,1].plot(t,[r['recall'] for r in sweep],label='Recall')
    axes[0,1].axvline(operating['confidence'],color='red',ls='--')
    axes[0,1].set(xlabel='Confidence',title='F1 / precision / recall vs confidence',ylim=(0,1))
    axes[0,1].legend()
    axes[1,0].plot(t,[r['FP'] for r in sweep],label='FP boxes')
    axes[1,0].plot(t,[r['FN'] for r in sweep],label='FN boxes')
    axes[1,0].plot(t,[r['negative_images_with_false_alarm'] for r in sweep],label='Negative images with alarm')
    axes[1,0].axvline(operating['confidence'],color='red',ls='--')
    axes[1,0].set(xlabel='Confidence',ylabel='Count',title='Errors vs confidence',yscale='log')
    axes[1,0].legend()
    matrix = np.array([[image_confusion['flame_image']['alarm'],image_confusion['flame_image']['no_alarm']],
                       [image_confusion['negative_image']['alarm'],image_confusion['negative_image']['no_alarm']]])
    axes[1,1].imshow(matrix,cmap='Blues')
    for (i,j),value in np.ndenumerate(matrix):
        axes[1,1].text(j,i,str(value),ha='center',va='center',color='black')
    axes[1,1].set(xticks=[0,1],xticklabels=['alarm','no alarm'],yticks=[0,1],yticklabels=['flame image','negative image'],
                  title=f'Image-level confusion at conf {operating["confidence"]}')
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(directory/'curves.png',dpi=110)
    plt.close(figure)


def latency(model,records,size,device,count):
    """Batch 1, float32, 10 warm-up images; per stage wall time, CUDA synchronised."""
    stages = {'decode':[],'preprocess':[],'inference':[],'postprocess':[]}
    sync = (lambda: torch.cuda.synchronize(device)) if device.type=='cuda' else (lambda: None)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    subset = records[:count+10]
    with torch.inference_mode():
        for n,record in enumerate(subset):
            t0 = time.perf_counter()
            image = cv2.imread(str(ROOT/record['image']))
            t1 = time.perf_counter()
            tensor,_ = preprocess(image,size)
            tensor = tensor.unsqueeze(0).to(device)
            sync()
            t2 = time.perf_counter()
            output = model(tensor)
            sync()
            t3 = time.perf_counter()
            detections(output[0].cpu(),record,size)
            t4 = time.perf_counter()
            if n>=10:
                for name,value in zip(stages,(t1-t0,t2-t1,t3-t2,t4-t3)):
                    stages[name].append(value*1000)
    total = [sum(v) for v in zip(*stages.values())]
    summary = {name:{'mean_ms':float(np.mean(v)),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))}
               for name,v in stages.items()}
    summary['end_to_end'] = {'mean_ms':float(np.mean(total)),'median_ms':float(np.median(total)),'p95_ms':float(np.percentile(total,95))}
    summary['fps_from_mean_end_to_end'] = 1000/float(np.mean(total))
    summary['method'] = (f'batch 1, float32, {len(total)} timed images after 10 warm-up, wall clock; '
                         'decode = cv2.imread; preprocess = letterbox + host-to-device copy; '
                         'postprocess = score, threshold 0.001, NMS 0.65, box restore on CPU')
    if device.type=='cuda':
        summary['peak_gpu_memory_allocated_MiB'] = torch.cuda.max_memory_allocated(device)/2**20
    return summary


def throughput(model,size,device,batch=16,warmup=10,iterations=50):
    """GPU-only (or CPU-only) network throughput on random input; no decode or NMS."""
    if device.type!='cuda':
        return None
    data = torch.rand(batch,3,size,size,device=device)*255
    with torch.inference_mode():
        for _ in range(warmup):
            model(data)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(data)
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-start
    return {'batch':batch,'images_per_second':batch*iterations/seconds,'ms_per_batch':seconds/iterations*1000,
            'method':f'random float32 input, {warmup} warm-up + {iterations} timed batches, CUDA-synchronised'}


def plain(value):
    """json fallback for numpy scalars."""
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',choices=['nano','s'],required=True)
    parser.add_argument('--checkpoint',required=True,help='best.pth from a training run')
    parser.add_argument('--data',required=True,help='Manifest directory')
    parser.add_argument('--split',choices=['val','validation_full','test'],required=True)
    parser.add_argument('--size',type=int,required=True,help='Input size the model was trained with')
    parser.add_argument('--output',required=True)
    parser.add_argument('--threshold-from',help='metrics.json of a validation run; uses its selected confidence')
    parser.add_argument('--confidence',type=float,help='Operating confidence fixed in advance (not selected on any split)')
    parser.add_argument('--final-test',action='store_true',help='Required for the test split')
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--latency-images',type=int,default=100)
    parser.add_argument('--review',type=int,default=50)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    if args.threshold_from and args.confidence is not None:
        raise SystemExit('Use either --threshold-from or --confidence, not both')
    if args.split=='test' and not (args.final_test and (args.threshold_from or args.confidence is not None)):
        raise SystemExit('Test split needs --final-test and --threshold-from <validation metrics.json> or --confidence')
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = Path(args.checkpoint)
    model = build(args.arch)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    model.to(device).eval()
    manifest = Path(args.data)/f'{args.split}.json'
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
                boxes,scores = detections(prediction,records[index],args.size)
                for (x,y,x2,y2),score in zip(boxes.tolist(),scores.tolist()):
                    predictions.append({'image_id':index,'category_id':1,'bbox':[x,y,x2-x,y2-y],'score':score})
            if done%50==0 or done==len(loader):
                elapsed = time.perf_counter()-start
                print(f'batch {done}/{len(loader)}, eta_min={elapsed/done*(len(loader)-done)/60:.1f}',flush=True)
    inference_seconds = time.perf_counter()-start
    coco = coco_metrics(records,predictions)
    by_image = [[] for _ in records]
    for prediction in predictions:
        by_image[prediction['image_id']].append(prediction)
    matches = [greedy(record['boxes'],[p for p in candidates if p['score']>=GRID[0]])
               for record,candidates in zip(records,by_image)]
    if args.threshold_from:
        source = json.loads(Path(args.threshold_from).read_text())
        if source['split']=='test':
            raise SystemExit('Threshold must come from a validation run')
        threshold = float(source['operating_point']['confidence'])
        selection = f'fixed from {args.threshold_from} ({source["split"]}, {source["threshold_selection"]})'
    if args.confidence is not None:
        threshold = args.confidence
        selection = f'fixed in advance at {args.confidence:.2f} (not selected on any split)'
    sweep = [at_threshold(records,matches,t) for t in GRID]  # curves; selection below only without a given threshold
    if not args.threshold_from and args.confidence is None:
        best = max(sweep,key=lambda r:r['F1'])
        threshold = best['confidence']
        selection = f'maximum F1 on {args.split} over confidence 0.01-0.99 (step 0.01)'
    operating = at_threshold(records,matches,threshold)
    breakdown,failures = detail(records,matches,threshold)
    image_confusion = {'flame_image':{'alarm':operating['flame_images_with_any_detection'],
                                      'no_alarm':operating['flame_images']-operating['flame_images_with_any_detection']},
                       'negative_image':{'alarm':operating['negative_images_with_false_alarm'],
                                         'no_alarm':operating['negative_images']-operating['negative_images_with_false_alarm']},
                       'definition':'alarm = at least one prediction >= operating confidence anywhere in the image'}
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1,3,args.size,args.size,device=device))
    timing = latency(model,records,args.size,device,args.latency_images)
    batched = throughput(model,args.size,device)
    fixed = [at_threshold(records,matches,t) for t in FIXED_THRESHOLDS]
    metrics = {
        'arch':args.arch,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'split':args.split,'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'input_size':args.size,'device':str(device),
        'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu','torch':torch.__version__,
        'dataset':{'images':len(records),'flame_images':operating['flame_images'],'negative_images':operating['negative_images'],
                   'flame_boxes':sum(len(r['boxes']) for r in records),
                   'boxes_by_coco_area':{k:v['boxes'] for k,v in breakdown['recall_by_coco_area'].items()}},
        'measured':{'coco':coco,'operating_point':operating,**breakdown,'image_level_confusion':image_confusion,
                    'fixed_thresholds':fixed,'latency_on_evaluation_device':timing,'gpu_throughput':batched,
                    'evaluation_seconds':inference_seconds},
        'calculated':{'parameters':sum(p.numel() for p in model.parameters()),
                      'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'checkpoint_file_bytes':checkpoint.stat().st_size,
                      'gmacs':counter.get_total_flops()/2e9,
                      'gmacs_convention':'torch.utils.flop_counter FLOPs / 2 at batch 1; conv/matmul only'},
        'not_measured':['AM62A latency/FPS/memory (no board)','TIDL compilation','validation loss (train.py does not compute it)'],
        'operating_point':operating,'threshold_selection':selection,
        'definitions':__doc__.strip()}
    if args.split!='test':
        metrics['measured']['targets'] = targets(sweep)
    write_csv(output/'threshold_sweep.csv',sweep)
    write_csv(output/'fixed_thresholds.csv',fixed)
    note = ' - curves for reporting only, threshold not chosen on test' if args.split=='test' else ''
    plots(sweep,operating,image_confusion,output,f'{args.arch} {args.split} ({len(records)} images){note}')
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,default=plain))
    flat = [{'metric':f'coco.{k}','value':v,'label':'measured'} for k,v in coco.items()]
    flat += [{'metric':f'operating.{k}','value':v,'label':'measured'} for k,v in operating.items()]
    flat += [{'metric':f'recall_{k}','value':v['recall'],'label':'measured'} for k,v in breakdown['recall_by_coco_area'].items()]
    flat += [{'metric':f'conf_{row["confidence"]:.2f}.{k}','value':row[k],'label':'measured (fixed threshold)'}
             for row in fixed for k in ('precision','recall','F1','FP','FN','false_alarms_per_image')]
    flat += [{'metric':f'latency.{k}.mean_ms','value':v['mean_ms'],'label':f'measured on {metrics["device_name"]}'}
             for k,v in timing.items() if isinstance(v,dict)]
    flat += [{'metric':k,'value':v,'label':'calculated'} for k,v in metrics['calculated'].items() if k!='gmacs_convention']
    write_csv(output/'metrics.csv',flat)
    sampler = np.random.default_rng(42)
    for name,items in failures.items():
        chosen = [items[i] for i in sampler.permutation(len(items))[:args.review]]
        (output/f'review_{name}.json').write_text(json.dumps(chosen,indent=2,default=plain))
        if chosen:
            contact_sheets(chosen,records,by_image,threshold,f'review_{name}',output)
    (output/'predictions_ge_0.01.json').write_text(json.dumps([p for p in predictions if p['score']>=.01]))
    print(json.dumps({'coco':coco,'operating_point':operating,'threshold_selection':selection},indent=2,default=plain))


if __name__=='__main__':
    main()
