"""CPU-first fine-tuning, validation and ONNX export of the official TI detector."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from flame_model import FlameDetector, ROOT, load_pretrained


def preprocess(image, size=416):
    h,w = image.shape[:2]
    ratio = min(size/h,size/w)
    resized = cv2.resize(image,(int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size,size,3),114,dtype=np.uint8)
    canvas[:resized.shape[0],:resized.shape[1]] = resized
    # Historical TI/YOLOX non-legacy path: BGR, CHW, raw float32 0..255.
    return torch.from_numpy(np.ascontiguousarray(canvas.transpose(2,0,1))).float(),ratio


class Flames(Dataset):
    def __init__(self, manifest, size=416, augment=False):
        self.records = json.loads(Path(manifest).read_text())
        self.size,self.augment = size,augment
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        record = self.records[index]
        image = cv2.imread(str(ROOT/record['image']))
        if image is None:
            raise ValueError(f'Cannot decode {record["image"]}')
        if list(image.shape[:2]) != [record['height'],record['width']]:
            raise ValueError(f'Image/annotation dimensions disagree: {record["image"]}')
        boxes = np.asarray(record['boxes'],dtype=np.float32).reshape(-1,4).copy()
        rng = random.Random(42 + self.epoch*1000003 + index)
        if self.augment and rng.random() < .5:
            image = image[:,::-1].copy()
            boxes[:,[0,2]] = image.shape[1]-boxes[:,[2,0]]
        tensor,ratio = preprocess(image,self.size)
        if self.augment:
            # Operate on the resized content, avoiding huge HSV temporaries for
            # 10k-pixel source images. Leave the 114-valued padding unchanged.
            content=tensor.numpy().transpose(1,2,0)[:int(image.shape[0]*ratio),:int(image.shape[1]*ratio)].astype(np.uint8)
            hsv = cv2.cvtColor(content,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:,:,0] = (hsv[:,:,0]+rng.uniform(-.01,.01)*180)%180
            hsv[:,:,1] = np.clip(hsv[:,:,1]*rng.uniform(.7,1.3),0,255)
            hsv[:,:,2] = np.clip(hsv[:,:,2]*rng.uniform(.7,1.3),0,255)
            converted=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            tensor[:,:converted.shape[0],:converted.shape[1]]=torch.from_numpy(converted.transpose(2,0,1))
        boxes *= ratio
        targets = torch.zeros((max(1,len(boxes)),5),dtype=torch.float32)
        if len(boxes):
            targets[:,1:3] = torch.from_numpy((boxes[:,:2]+boxes[:,2:])/2)
            targets[:,3:5] = torch.from_numpy(boxes[:,2:]-boxes[:,:2])
        return tensor,targets,index


def collate(batch):
    targets = torch.zeros((len(batch),max(len(b[1]) for b in batch),5))
    for i,(_,target,_) in enumerate(batch):
        targets[i,:len(target)] = target
    return torch.stack([b[0] for b in batch]),targets,[b[2] for b in batch]


def configure_worker(_):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def detections(prediction, record, size=416, threshold=.001):
    score = prediction[:,4]*prediction[:,5]
    keep = score >= threshold
    prediction,score = prediction[keep],score[keep]
    boxes = torch.cat((prediction[:,:2]-prediction[:,2:4]/2,
                       prediction[:,:2]+prediction[:,2:4]/2),1)
    selected = torchvision.ops.nms(boxes,score,.65)[:100]
    boxes,score = boxes[selected],score[selected]
    ratio = min(size/record['height'],size/record['width'])
    boxes /= ratio
    boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,record['width'])
    boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,record['height'])
    return boxes,score


def operating_metrics(records,predictions,thresholds=(.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9)):
    """One-to-one score-ordered matching at IoU=.5, single flame class."""
    grouped = [[] for _ in records]
    for prediction in predictions:
        grouped[prediction['image_id']].append(prediction)
    rows = []
    for threshold in thresholds:
        tp=fp=fn=false_positive_images=negative_images=0
        for record,candidates in zip(records,grouped):
            matched = set()
            candidates = sorted((p for p in candidates if p['score']>=threshold),key=lambda p:-p['score'])
            if not record['boxes']:
                negative_images+=1
                false_positive_images+=bool(candidates)
            for prediction in candidates:
                x,y,w,h = prediction['bbox']
                best_iou,best_index = 0,None
                for index,(a,b,c,d) in enumerate(record['boxes']):
                    if index in matched:
                        continue
                    intersection = max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b))
                    union = w*h+(c-a)*(d-b)-intersection
                    iou = intersection/union if union>0 else 0
                    if iou>best_iou:
                        best_iou,best_index = iou,index
                if best_index is not None and best_iou>=.5:
                    tp+=1
                    matched.add(best_index)
                else:
                    fp+=1
            fn+=len(record['boxes'])-len(matched)
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        rows.append({'confidence':threshold,'matching_iou':.5,'TP':tp,'FP':fp,'FN':fn,
                     'precision':precision,'recall':recall,'F1':f1,
                     'negative_images':negative_images,'false_positive_images':false_positive_images,
                     'negative_image_false_positive_rate':false_positive_images/negative_images if negative_images else None})
    return {'selected_by':'maximum F1 on validation only; provisional, not an alarm acceptance criterion',
            'selected':max(rows,key=lambda row:row['F1']),'sweep':rows}


def atomic_checkpoint(state,path):
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, dataset, threads, batch_size=4):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval()
    device=next(model.parameters()).device
    gt = COCO()
    images,annotations,results = [],[],[]
    for i,record in enumerate(dataset.records):
        images.append({'id':i,'width':record['width'],'height':record['height']})
        for box in record['boxes']:
            x,y,x2,y2 = box
            annotations.append({'id':len(annotations)+1,'image_id':i,'category_id':1,
                                'bbox':[x,y,x2-x,y2-y],'area':(x2-x)*(y2-y),'iscrowd':0})
    gt.dataset = {'info':{},'images':images,'annotations':annotations,'categories':[{'id':1,'name':'flame'}]}
    gt.createIndex()
    loader = DataLoader(dataset,batch_size=batch_size,collate_fn=collate,num_workers=0)
    for tensors,_,indices in loader:
        predictions = model(tensors.to(device)).cpu()
        for prediction,index in zip(predictions,indices):
            boxes,scores = detections(prediction,dataset.records[index],dataset.size)
            for box,score in zip(boxes.tolist(),scores.tolist()):
                x,y,x2,y2 = box
                results.append({'image_id':index,'category_id':1,'bbox':[x,y,x2-x,y2-y],'score':score})
    if not results:
        return {'AP50_95':0.0,'AP50':0.0,'AP_small':0.0,'AR100':0.0,'detections':0},results
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation = COCOeval(gt,gt.loadRes(results),'bbox')
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return {'AP50_95':float(evaluation.stats[0]),'AP50':float(evaluation.stats[1]),
            'AP_small':float(evaluation.stats[3]),'AR100':float(evaluation.stats[8]),'detections':len(results)},results


def benchmark(model, batch, candidates):
    initial = copy.deepcopy(model.state_dict())
    device=next(model.parameters()).device
    images,targets=batch[0].to(device),batch[1].to(device)
    times = {}
    for threads in candidates:
        torch.set_num_threads(threads)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0003)
        model.train()
        durations = []
        for step in range(3):
            if device.type=='cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model(images,targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite benchmark loss')
            loss.backward()
            optimizer.step()
            if device.type=='cuda':
                torch.cuda.synchronize()
            if step:
                durations.append(time.perf_counter()-start)
        times[threads] = sum(durations)/len(durations)
        print(f'{device.type.upper()} benchmark: host_threads={threads}, seconds/batch={times[threads]:.3f}',flush=True)
    model.load_state_dict(initial)
    return times


def export_and_check(model,dataset,directory,threads):
    import onnx
    import onnxruntime as ort
    # Deliver CPU-portable ONNX regardless of training device.
    model=model.to('cpu').eval()
    sample = dataset[0][0].unsqueeze(0)
    target = directory/'flame.onnx'
    torch.onnx.export(model,sample,str(target),opset_version=13,input_names=['images'],
                      output_names=['predictions'],do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(target)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
    maximum_error = 0.0
    for index in range(min(4,len(dataset))):
        sample = dataset[index][0].unsqueeze(0)
        with torch.inference_mode():
            reference = model(sample).numpy()
        actual = session.run(None,{'images':sample.numpy()})[0]
        maximum_error = max(maximum_error,float(np.max(np.abs(actual-reference))))
        np.testing.assert_allclose(actual,reference,atol=.002,rtol=.001)
    for _ in range(3):
        session.run(None,{'images':sample.numpy()})
    start = time.perf_counter()
    for _ in range(10):
        session.run(None,{'images':sample.numpy()})
    result = {'onnx_valid':True,'parity_inputs':min(4,len(dataset)),'max_absolute_error':maximum_error,
              'parity_atol':.002,'parity_rtol':.001,'cpu_inference_ms':(time.perf_counter()-start)*100,
              'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (directory/'export.json').write_text(json.dumps(result,indent=2))
    (directory/'model_contract.json').write_text(json.dumps({
        'architecture':'TI YOLOX-Nano-ti-lite','classes':['flame'],
        'input_shape':[1,3,dataset.size,dataset.size],'input_dtype':'float32','channel_order':'BGR',
        'normalization':'none; pixel values 0..255','resize':'aspect-preserving, floor dimensions; top-left placement; pad 114',
        'output':'[batch,anchors,6]: decoded centre_x,centre_y,width,height,objectness,class_score in input pixels',
        'postprocessing':'score=objectness*class_score; threshold then NMS; restore boxes using resize ratio',
        'nms_iou':.65,'operating_confidence':'not selected; .001 used for AP evaluation',
        'target_TIDL_compilation':'pending'},indent=2))
    return result


@torch.inference_mode()
def save_examples(model,dataset,directory):
    model.eval()
    chosen = [i for i,r in enumerate(dataset.records) if r['boxes']][:4]
    chosen += [i for i,r in enumerate(dataset.records) if not r['boxes']][:4]
    panels = []
    for i in chosen:
        record = dataset.records[i]
        image = cv2.imread(str(ROOT/record['image']))
        predictions = model(dataset[i][0].unsqueeze(0))[0]
        boxes,scores = detections(predictions,record,dataset.size,threshold=.1)
        for box in record['boxes']:
            cv2.rectangle(image,tuple(map(int,box[:2])),tuple(map(int,box[2:])),(0,255,0),2)
        for box,score in zip(boxes.tolist(),scores.tolist()):
            x,y,x2,y2 = map(int,box)
            cv2.rectangle(image,(x,y),(x2,y2),(0,0,255),2)
            cv2.putText(image,f'{score:.2f}',(x,max(15,y)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        panel = cv2.resize(image,(416,280))
        cv2.putText(panel,f'val {i}: green=label red=prediction >0.1',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        panels.append(panel)
    if len(panels)==8:
        cv2.imwrite(str(directory/'validation_examples.jpg'),np.vstack([np.hstack(panels[:4]),np.hstack(panels[4:])]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True,help='JSON defaults; explicit CLI arguments override these')
    parser.add_argument('--data',default='data/fasdd')
    parser.add_argument('--output',default='runs/full_v1')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--size',type=int,default=416)
    parser.add_argument('--threads',type=int,default=0)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--checkpoint',default='models/pretrained/yolox_nano_ti_lite_coco.pth')
    parser.add_argument('--resume',default=None)
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--ema',action='store_true')
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--warmup-updates',type=int,default=0)
    parser.add_argument('--cosine',action='store_true')
    parser.add_argument('--accumulate',type=int,default=1)
    parser.add_argument('--patience',type=int,default=0)
    parser.add_argument('--max-batches',type=int,default=0,help='Integration check only; 0 means all batches')
    parser.add_argument('--checkpoint-every',type=int,default=250)
    parser.add_argument('--log-every',type=int,default=50)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--stop-after-updates',type=int,default=0,help='Checkpoint/resume integration check only')
    config_args,_=parser.parse_known_args()
    if config_args.config:
        config_path=Path(config_args.config)
        if not config_path.is_absolute():
            config_path=ROOT/config_path
        settings=json.loads(config_path.read_text())
        allowed={action.dest for action in parser._actions}-{'help','config'}
        if set(settings)-allowed:
            raise ValueError(f'Unknown configuration keys: {set(settings)-allowed}')
        parser.set_defaults(**settings)
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    cv2.setNumThreads(0)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: install matching GPU PyTorch and NVIDIA driver, or use --device cpu')
    output = ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    if (output/'last.pth').exists() and not args.resume:
        raise ValueError('Existing checkpoints: use --resume or a new output directory')
    model = FlameDetector()
    loaded = load_pretrained(model,ROOT/args.checkpoint)
    model.to(device)
    print(f'Model parameters={sum(p.numel() for p in model.parameters())}; weights={loaded}',flush=True)
    train = Flames(ROOT/args.data/'train.json',args.size,augment=True)
    val = Flames(ROOT/args.data/'val.json',args.size)
    if args.batch<2 or args.accumulate<1 or args.epochs<1:
        raise ValueError('Require batch>=2, accumulate>=1, epochs>=1')
    generator = torch.Generator().manual_seed(42)
    worker_options={'multiprocessing_context':'spawn'} if args.workers else {}
    loader = DataLoader(train,batch_size=args.batch,shuffle=True,collate_fn=collate,
                        num_workers=args.workers,generator=generator,worker_init_fn=configure_worker,
                        pin_memory=device.type=='cuda',**worker_options)
    first = next(iter(loader))
    candidates = sorted({2,4,min(8,os.cpu_count() or 2),os.cpu_count() or 2})
    timing = benchmark(model,first,([4] if device.type=='cuda' else candidates) if not args.threads else [args.threads])
    threads = min(timing,key=timing.get)
    torch.set_num_threads(threads)
    configuration = vars(args)|{'selected_threads':threads,'benchmark_seconds_per_batch':timing,
        'python':platform.python_version(),'torch':torch.__version__,'available_cpus':os.cpu_count(),
        'parameters':sum(p.numel() for p in model.parameters()),'pretrained':loaded,
        'vendor_revision':'cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6',
        'pretrained_sha256':hashlib.sha256((ROOT/args.checkpoint).read_bytes()).hexdigest(),
        'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
        'train_images':len(train),'validation_images':len(val),
        'dataset_manifest_sha256':{name:hashlib.sha256((ROOT/args.data/name).read_bytes()).hexdigest()
                                  for name in ('train.json','val.json')},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'scripts').glob('*.py')},
        'evaluation_warning':'still-image validation; source independence and production acceptance not established'}
    (output/'config.json').write_text(json.dumps(configuration,indent=2))
    if args.smoke_only:
        print(json.dumps(export_and_check(model,val,output,threads),indent=2),flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0005)
    contract={key:vars(args)[key] for key in ['epochs','batch','size','lr','accumulate','ema','cosine','warmup_updates','max_batches']}
    contract['dataset_manifest_sha256']=configuration['dataset_manifest_sha256']
    ema = None
    if args.ema:
        from importlib import import_module
        ema = import_module('yolox.utils.ema').ModelEMA(model)
    history = []
    first_epoch = 0
    best_ap = -1.0
    start_batch=global_updates=stale_epochs=0
    resume_loss=0.0
    if args.resume:
        state = torch.load(ROOT/args.resume,map_location='cpu',weights_only=True)
        if 'training_contract' in state and state['training_contract']!=contract:
            raise ValueError('Resume data/recipe differs from saved run; use the original manifests/arguments')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        first_epoch,best_ap,history = state['epoch'],state['best_ap'],state['history']
        if best_ap>=0 and not (output/'best.pth').exists():
            raise ValueError('Copy the entire run folder, including best.pth, before resuming on another PC')
        start_batch=state.get('next_batch',0)
        resume_loss=state.get('partial_loss',0.0)
        global_updates=state.get('global_updates',0)
        stale_epochs=state.get('stale_epochs',0)
        if ema and 'ema' in state:
            ema.ema.load_state_dict(state['ema'])
            ema.updates=state['ema_updates']
        elif ema:
            ema.ema.load_state_dict(model.state_dict())
        if 'torch_rng_state' in state:
            torch.set_rng_state(state['torch_rng_state'])
            random.setstate(state['python_rng_state'])
            if device.type=='cuda' and state.get('cuda_rng_state_all'):
                torch.cuda.set_rng_state_all(state['cuda_rng_state_all'])
        else:
            print('Resume note: legacy checkpoint has no RNG state; random sequence restarts from seed 42.',flush=True)
    batches=min(len(loader),args.max_batches) if args.max_batches else len(loader)
    total_updates=args.epochs*math.ceil(batches/args.accumulate)
    def snapshot(epoch,next_batch,partial_loss):
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
               'best_ap':best_ap,'history':history,'next_batch':next_batch,'partial_loss':partial_loss,
               'global_updates':global_updates,'stale_epochs':stale_epochs,
               'torch_rng_state':torch.get_rng_state(),'python_rng_state':random.getstate(),
               'training_contract':contract,
               'cuda_rng_state_all':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        if ema:
            state.update({'ema':ema.ema.state_dict(),'ema_updates':ema.updates})
        return state
    for epoch in range(first_epoch,args.epochs):
        model.train()
        train.epoch=epoch
        # Last two epochs reduce augmentation variation for final refinement.
        train.augment=epoch<max(1,args.epochs-2)
        generator.manual_seed(42+epoch)
        total_loss=resume_loss if epoch==first_epoch else 0.0
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step,(images,targets,_) in enumerate(loader):
            if step<start_batch:
                continue
            if step>=batches:
                break
            group_start=(step//args.accumulate)*args.accumulate
            group_size=min(args.accumulate,batches-group_start)
            images,targets=images.to(device),targets.to(device)
            loss = model(images,targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss epoch={epoch+1} step={step}')
            (loss/group_size).backward()
            update=(step+1)%args.accumulate==0 or step+1==batches
            if update:
                if global_updates<args.warmup_updates:
                    lr=args.lr*(global_updates+1)/args.warmup_updates
                elif args.cosine:
                    fraction=min(1,max(0,(global_updates-args.warmup_updates)/max(1,total_updates-args.warmup_updates)))
                    lr=args.lr*(.1+.9*(1+math.cos(math.pi*fraction))/2)
                else:
                    lr=args.lr
                for group in optimizer.param_groups:
                    group['lr']=lr
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_updates+=1
                if ema:
                    ema.update(model)
            total_loss += loss.item()
            if update and args.checkpoint_every and global_updates%args.checkpoint_every==0:
                atomic_checkpoint(snapshot(epoch,step+1,total_loss),output/'last.pth')
            if update and args.stop_after_updates and global_updates>=args.stop_after_updates:
                atomic_checkpoint(snapshot(epoch,step+1,total_loss),output/'last.pth')
                print(f'Integration stop saved epoch={epoch+1}, next_batch={step+1}, updates={global_updates}',flush=True)
                return
            if step%args.log_every==0:
                elapsed=time.perf_counter()-start
                completed=step+1-start_batch
                eta=elapsed/completed*(batches-step-1)
                print(f'epoch={epoch+1}/{args.epochs} batch={step+1}/{batches} loss={loss.item():.4f} epoch_eta_min={eta/60:.1f}',flush=True)
        train_seconds = time.perf_counter()-start
        validation_model=ema.ema if ema else model
        print(f'epoch={epoch+1}: evaluating {len(val)} fixed validation images',flush=True)
        metrics,predictions = evaluate(validation_model,val,threads,args.batch)
        operating=operating_metrics(val.records,predictions)
        row={'epoch':epoch+1,'train_loss':total_loss/batches,'train_seconds':train_seconds,
             'operating_point':operating['selected'],'global_updates':global_updates}|metrics
        history.append(row)
        improved = metrics['AP50_95'] > best_ap
        best_ap = max(best_ap,metrics['AP50_95'])
        stale_epochs=0 if improved else stale_epochs+1
        atomic_checkpoint(snapshot(epoch+1,0,0.0),output/'last.pth')
        if improved:
            atomic_checkpoint({'model':validation_model.state_dict()},output/'best.pth')
            (output/'validation_predictions.json').write_text(json.dumps(predictions))
            (output/'operating_metrics.json').write_text(json.dumps(operating,indent=2))
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        start_batch=0
        resume_loss=0.0
        if args.patience and stale_epochs>=args.patience:
            print(f'Early stop: no AP improvement for {stale_epochs} epochs',flush=True)
            break
    model.load_state_dict(torch.load(output/'best.pth',map_location='cpu',weights_only=True)['model'])
    exported = export_and_check(model,val,output,threads)
    save_examples(model,val,output)
    print(json.dumps({'best_validation_AP50_95':best_ap,'export':exported}),flush=True)


if __name__=='__main__':
    main()
