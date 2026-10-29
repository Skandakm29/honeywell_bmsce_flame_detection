"""Pre-training checks for YOLOX-S-ti-lite on the Kaggle manifests. Trains nothing that is kept.

Checks: checkpoint keys/shapes, preprocessing and label conversion against the manifest,
loss components, gradients, one optimizer step, save/reload equality, single-class output
shape, decoded detections, and measured seconds/batch and peak GPU memory.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
from s_model import ROOT, SFlameDetector  # noqa: E402
from flame_model import load_pretrained  # noqa: E402
from train import Flames, collate, detections  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',required=True,help='Manifest directory (train.json)')
    parser.add_argument('--checkpoint',required=True,help='TI YOLOX-S-ti-lite COCO checkpoint')
    parser.add_argument('--output',required=True)
    parser.add_argument('--size',type=int,default=640)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    report = {'device':str(device),'torch':torch.__version__,'size':args.size,'batch':args.batch}
    if device.type=='cuda':
        report['device_name'] = torch.cuda.get_device_name(device)

    # 1-2. Architecture and pretrained weights.
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    state = state.get('model',state)
    model = SFlameDetector()
    expected = model.state_dict()
    report['checkpoint'] = {'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'tensors':len(state),
        'unexpected_keys':sorted(set(state)-set(expected)),
        'missing_keys':sorted(set(expected)-set(state)),
        'shape_mismatch':sorted(k for k in set(state)&set(expected) if state[k].shape!=expected[k].shape)}
    loaded = load_pretrained(model,checkpoint)
    report['checkpoint']['loaded_tensors'] = loaded['loaded_tensors']
    report['checkpoint']['new_tensors'] = loaded['new_class_tensors']
    report['parameters'] = sum(p.numel() for p in model.parameters())
    model.to(device)

    # 3-4. A batch from the training manifest; labels must equal manifest boxes scaled by the letterbox ratio.
    plain = Flames(Path(args.data)/'train.json',args.size,augment=False)
    positives = [i for i,r in enumerate(plain.records) if r['boxes']][:args.batch//2]
    negatives = [i for i,r in enumerate(plain.records) if not r['boxes']][:args.batch-len(positives)]
    chosen = positives+negatives
    images,targets,_ = collate([plain[i] for i in chosen])
    worst = 0.0
    for row,index in enumerate(chosen):
        record = plain.records[index]
        ratio = min(args.size/record['height'],args.size/record['width'])
        boxes = np.asarray(record['boxes'],dtype=np.float32).reshape(-1,4)*ratio
        expected_targets = np.zeros((len(boxes),4),np.float32)
        expected_targets[:,:2] = (boxes[:,:2]+boxes[:,2:])/2
        expected_targets[:,2:] = boxes[:,2:]-boxes[:,:2]
        actual = targets[row,:len(boxes),1:].numpy()
        worst = max(worst,float(np.abs(actual-expected_targets).max()) if len(boxes) else 0.0)
        assert torch.all(targets[row,:,0]==0),'class column must be 0 (flame)'
        assert torch.all(targets[row,len(boxes):]==0),'padding rows must be zero'
    assert worst<1e-3,f'label conversion error {worst}'
    report['labels'] = {'images':len(chosen),'flame_images':len(positives),'negative_images':len(negatives),
                        'flame_boxes':int((targets.sum(2)>0).sum()),'max_abs_label_error_px':worst,
                        'input_dtype':str(images.dtype),'input_range':[float(images.min()),float(images.max())],
                        'input_shape':list(images.shape)}
    sheet = []
    for row in range(len(chosen)):
        canvas = np.ascontiguousarray(images[row].numpy().transpose(1,2,0).astype(np.uint8))
        for _,cx,cy,w,h in targets[row].tolist():
            if w>0:
                cv2.rectangle(canvas,(int(cx-w/2),int(cy-h/2)),(int(cx+w/2),int(cy+h/2)),(0,255,0),2)
        sheet.append(cv2.resize(canvas,(320,320)))
    while len(sheet)%4:
        sheet.append(np.zeros_like(sheet[0]))
    cv2.imwrite(str(output/'smoke_labels.jpg'),np.vstack([np.hstack(sheet[i:i+4]) for i in range(0,len(sheet),4)]))

    # 5-6. Loss components, gradients and one optimizer step.
    model.train()
    images,targets = images.to(device),targets.to(device)
    features = model.backbone(images)
    loss,iou_loss,obj_loss,cls_loss,l1_loss,fg_ratio = model.head(features,targets,images)
    assert torch.isfinite(loss),'nonfinite loss'
    loss.backward()
    grads = {name:p.grad for name,p in model.named_parameters()}
    assert all(g is None or torch.isfinite(g).all() for g in grads.values()),'nonfinite gradient'
    cls_grad = sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith('head.cls_preds.'))
    report['loss'] = {'total':float(loss),'iou_x5':float(iou_loss),'objectness':float(obj_loss),
                      'class':float(cls_loss),'l1':float(l1_loss),'foreground_per_gt':float(fg_ratio),
                      'parameters_with_gradient':sum(g is not None and bool(g.abs().sum()>0) for g in grads.values()),
                      'parameters_total':len(grads),'new_class_head_gradient_abs_sum':cls_grad}
    assert cls_grad>0,'flame classification head received no gradient'
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0005)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    report['loss']['after_one_step'] = float(model(images,targets))

    # 7. Save/reload round trip.
    path = output/'smoke_roundtrip.pth'
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},path)
    clone = SFlameDetector()
    clone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
    clone.to(device).eval()
    model.eval()
    with torch.inference_mode():
        a,b = model(images),clone(images)
    report['roundtrip'] = {'max_abs_output_difference':float((a-b).abs().max()),'file_bytes':path.stat().st_size}
    assert torch.equal(a,b),'reloaded model differs'
    path.unlink()

    # 8. Output contract: [batch, anchors, 4 box + objectness + 1 class score].
    strides = (8,16,32)
    anchors = sum((args.size//s)**2 for s in strides)
    assert tuple(a.shape)==(len(chosen),anchors,6),f'unexpected output shape {tuple(a.shape)}'
    boxes,scores = detections(a[0].cpu(),plain.records[chosen[0]],args.size,threshold=.05)
    report['output'] = {'shape':list(a.shape),'anchors_expected':anchors,'classes':1,
                        'sample_detections_ge_0.05':len(scores),
                        'sample_top_scores':[round(s,3) for s in scores[:5].tolist()]}

    # Measured training speed and memory for this batch size.
    model.train()
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for step in range(6):
        if device.type=='cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        model(images,targets).backward()
        optimizer.step()
        if device.type=='cuda':
            torch.cuda.synchronize()
        if step:
            durations.append(time.perf_counter()-start)
    report['speed'] = {'seconds_per_batch_compute_only':sum(durations)/len(durations),
                       'note':'GPU/CPU compute for fwd+bwd+step; excludes data loading'}
    if device.type=='cuda':
        report['speed']['peak_gpu_memory_MiB'] = torch.cuda.max_memory_allocated(device)/2**20
    report['passed'] = True
    (output/'smoke_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
