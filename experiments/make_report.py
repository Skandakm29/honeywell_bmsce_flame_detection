"""Collect a finished run into summary.json, summary.csv and report.md (no new measurements).

Every value carries a label: measured (produced by running code in this run), calculated
(derived from the model or files) or NOT MEASURED. Missing inputs are reported as NOT MEASURED.
"""
import argparse
import csv
import json
from pathlib import Path

MISSING = 'NOT MEASURED'


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def gpu_peak(path):
    path = Path(path)
    if not path.exists():
        return None
    peak = None
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts)>=3 and parts[1].endswith('MiB'):
            used = int(parts[1].split()[0])
            peak = used if peak is None else max(peak,used)
    return peak


def evaluation_rows(name,metrics):
    if not metrics:
        return [(f'{name}.*',MISSING,'NOT MEASURED')]
    m = metrics['measured']
    o = m['operating_point']
    c = m['coco']
    rows = [(f'{name}.images',o['images'],'measured'),(f'{name}.flame_images',o['flame_images'],'measured'),
            (f'{name}.negative_images',o['negative_images'],'measured'),
            (f'{name}.flame_boxes',metrics['dataset']['flame_boxes'],'measured'),
            (f'{name}.AP50',c.get('AP50',MISSING),'measured'),(f'{name}.AP75',c.get('AP75',MISSING),'measured'),
            (f'{name}.AP50_95',c.get('AP50_95',MISSING),'measured'),(f'{name}.AP_small',c.get('AP_small',MISSING),'measured'),
            (f'{name}.AP_medium',c.get('AP_medium',MISSING),'measured'),(f'{name}.AP_large',c.get('AP_large',MISSING),'measured'),
            (f'{name}.AR100',c.get('AR100',MISSING),'measured'),
            (f'{name}.confidence_threshold',o['confidence'],'selected on validation'),
            (f'{name}.precision',o['precision'],'measured'),(f'{name}.recall',o['recall'],'measured'),
            (f'{name}.F1',o['F1'],'measured'),(f'{name}.TP',o['TP'],'measured'),(f'{name}.FP',o['FP'],'measured'),
            (f'{name}.FN',o['FN'],'measured'),(f'{name}.missed_flame_rate',o['missed_flame_rate'],'measured'),
            (f'{name}.mean_matched_iou',o['mean_matched_iou'],'measured'),
            (f'{name}.negative_images_with_false_alarm',o['negative_images_with_false_alarm'],'measured'),
            (f'{name}.image_level_false_positive_rate',o['image_level_false_positive_rate'],'measured'),
            (f'{name}.false_alarms_per_image',o['false_alarms_per_image'],'measured'),
            (f'{name}.false_alarms_per_negative_image',o['false_alarms_per_negative_image'],'measured')]
    for size,row in m['recall_by_coco_area'].items():
        rows.append((f'{name}.recall_{size}',row['recall'],f'measured ({row["boxes"]} boxes)'))
    timing = m['latency_on_evaluation_device']
    device = metrics['device_name']
    for stage in ('decode','preprocess','inference','postprocess','end_to_end'):
        rows.append((f'{name}.latency_{stage}_mean_ms',timing[stage]['mean_ms'],f'measured on {device}'))
    rows.append((f'{name}.fps_end_to_end',timing['fps_from_mean_end_to_end'],f'measured on {device}'))
    if 'peak_gpu_memory_allocated_MiB' in timing:
        rows.append((f'{name}.inference_peak_gpu_memory_MiB',timing['peak_gpu_memory_allocated_MiB'],f'measured on {device} (torch allocator, batch 1)'))
    return rows


def fmt(value):
    if isinstance(value,float):
        return f'{value:.4f}'
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--manifests',required=True)
    parser.add_argument('--smoke-report',required=True)
    args = parser.parse_args()
    run = Path(args.run)
    config = load(run/'config.json') or {}
    history = load(run/'history.json') or []
    export = load(run/'export.json')
    validation = load(run/'eval_validation_full/metrics.json')
    test = load(run/'eval_test/metrics.json')
    prepare = load(Path(args.manifests)/'prepare_report.json')
    smoke = load(args.smoke_report)
    best = max(history,key=lambda r:r['AP50_95']) if history else None
    peak = gpu_peak(run/'gpu_usage.csv')
    rows = [('model','TI YOLOX-S-ti-lite, 1 class (flame)','configuration'),
            ('input_size',config.get('size',MISSING),'configuration'),
            ('parameters',config.get('parameters',MISSING),'calculated'),
            ('best_checkpoint_bytes',(run/'best.pth').stat().st_size if (run/'best.pth').exists() else MISSING,'calculated'),
            ('onnx_bytes',(run/'flame.onnx').stat().st_size if (run/'flame.onnx').exists() else MISSING,'calculated'),
            ('gmacs',validation['calculated']['gmacs'] if validation else MISSING,'calculated (torch flop counter / 2, batch 1)'),
            ('epochs_planned',config.get('epochs',MISSING),'configuration'),
            ('epochs_completed',len(history),'measured'),
            ('selected_epoch',best['epoch'] if best else MISSING,'best monitor-val AP50:95'),
            ('monitor_val_AP50_95_at_selected_epoch',best['AP50_95'] if best else MISSING,'measured (2,000-image val subset)'),
            ('training_seconds',sum(r['train_seconds'] for r in history) if history else MISSING,'measured (excludes per-epoch validation)'),
            ('training_peak_gpu_memory_used_MiB',peak if peak is not None else MISSING,'measured (nvidia-smi, sampled every 60 s, whole device)'),
            ('batch',config.get('batch',MISSING),'configuration'),('accumulate',config.get('accumulate',MISSING),'configuration'),
            ('device',config.get('device_name',MISSING),'measured'),('torch',config.get('torch',MISSING),'measured'),
            ('onnx_parity_passed',export.get('parity_passed',MISSING) if export else MISSING,'measured (4 inputs, CPU onnxruntime, atol 0.002 rtol 0.001)'),
            ('onnx_parity_max_abs_error',export.get('max_absolute_error',MISSING) if export else MISSING,'measured (4 inputs, CPU onnxruntime)'),
            ('AM62A_latency_fps_memory',MISSING,'NOT MEASURED: no board'),('TIDL_compilation',MISSING,'NOT MEASURED: not attempted'),
            ('validation_loss',MISSING,'NOT MEASURED: train.py does not compute it')]
    rows += evaluation_rows('validation',validation)
    rows += evaluation_rows('test',test)
    summary = {'rows':[{'metric':m,'value':v,'label':l} for m,v,l in rows],
               'threshold_selection':validation['threshold_selection'] if validation else MISSING,
               'definitions':validation['definitions'] if validation else MISSING,
               'dataset':prepare['splits'] if prepare else MISSING,'quarantined':prepare['quarantined'] if prepare else MISSING,
               'exact_duplicates':prepare.get('exact_duplicates') if prepare else MISSING,
               'smoke_test':smoke,'history':history,'config':config}
    (run/'summary.json').write_text(json.dumps(summary,indent=2))
    with (run/'summary.csv').open('w',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric','value','label'])
        writer.writerows(rows)
    complete = bool(export and validation and test)
    lines = ['# YOLOX-S-Lite flame experiment report','',
             f'Status: **{"complete" if complete else "INCOMPLETE — see NOT MEASURED rows"}**. '
             f'Epochs completed {len(history)} of {config.get("epochs",MISSING)}'
             + (' (early stopping)' if history and config.get('epochs') and len(history)<config['epochs'] else '')+'.','',
             '## Method','',
             '- Data: Kaggle `yuulind/fasdd-cv-coco`, its own train/val/test splits. Only `fire` boxes are targets (`flame`); '
             'smoke-only and unannotated images are negatives. Quarantined images are listed in summary.json.',
             '- Model: TI YOLOX-S-ti-lite (depth 0.33, width 0.50, ReLU, conv stem, split max-pool) from TI edgeai-yolox '
             '`cbdd9d8`, initialised from TI\'s COCO checkpoint; new one-class head.',
             '- Training: baseline `scripts/train.py` unchanged (AdamW, warmup + cosine, EMA, flip/HSV augmentation), 640×640 letterbox, '
             'seed 42. Checkpoint selection: best AP50:95 on a fixed 2,000-image subset of val.',
             f'- Threshold: {summary["threshold_selection"]}. The test split was evaluated once with that threshold.',
             '- Comparability: these splits differ from the baseline (`full_v1`) splits, so results are not directly comparable with its 62.69 / 32.92.',
             '','## Results','','| Metric | Value | Label |','|---|---|---|']
    lines += [f'| {m} | {fmt(v)} | {l} |' for m,v,l in rows]
    lines += ['','## Definitions','','```',summary['definitions'] if isinstance(summary['definitions'],str) else MISSING,'```','',
              '## Limitations','',
              '- Desktop/Colab GPU timings are not AM62A performance; TIDL compilation and board measurements are not done.',
              '- Validation loss is not computed by the training loop.',
              '- Exact duplicates across Kaggle splits are reported in summary.json and were not removed.',
              '- Still-image metrics; video alarm behaviour is untested.']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:3]))
    print(f'wrote {run/"report.md"}, summary.json, summary.csv')


if __name__=='__main__':
    main()
