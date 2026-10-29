"""Final comparison of the baseline (YOLOX-Nano-ti-lite) and YOLOX-S-ti-lite. Reads results only.

Inputs live under --root (the Drive experiment folder): validation/<model>/metrics.json,
test/<model>/metrics.json, exports/<export>/export.json, checkpoints/<run>/, logs/, configs/,
manifests_v2/prepare_report.json. Writes <reports>/metrics.json, comparison.csv, report.md and
<plots>/comparison_*.png. Models default to baseline vs s_lite; --model KEY EXPORT LABEL (repeated)
compares any evaluated folders. Missing values are N/A with a reason.
"""
import argparse
import csv
import json
from pathlib import Path

NA = 'N/A'
MODELS = {'baseline':'YOLOX-Nano-ti-lite (baseline, 416)','s_lite':'YOLOX-S-ti-lite (640)'}
EXPORTS = {name:name for name in MODELS}
DEFAULT_THRESHOLD_RULE = ("Each model's confidence threshold is the maximum-F1 point on Kaggle **validation**; "
                          'test was evaluated once with it.')
DEFAULT_LIMITATIONS = ['Five epochs only; longer training may change the ranking.',
                       'The training loop disables augmentation for its last two epochs (inherited from the baseline recipe); with 5 epochs that is 40% of training.']


def load(path):
    path = Path(path)
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return None


def mb(value):
    return round(value/2**20,2) if isinstance(value,(int,float)) else NA


def num(value,digits=4):
    return round(value,digits) if isinstance(value,(int,float)) else NA


def model_row(name,root,why_missing):
    test = load(root/'test'/name/'metrics.json')
    export = load(root/'exports'/EXPORTS[name]/'export.json')
    if not test:
        return {'Model':MODELS[name],**{k:NA for k in ('AP50','AP75','mAP50-95','Precision','Recall','F1','False alarms/image',
                'Parameters','Model size (ONNX MB)','FLOPs/MACs (GMACs)','Median latency (GPU-only ms)','FPS (end-to-end)',
                'Median end-to-end latency (ms)','Checkpoint size (MB)','Confidence threshold','Input size','Test images')},
                'Note':why_missing.get(name,'test evaluation not run')}
    m = test['measured']
    o = m['operating_point']
    timing = m['latency_on_evaluation_device']
    return {'Model':MODELS[name],'AP50':num(m['coco'].get('AP50')),'AP75':num(m['coco'].get('AP75')),
            'mAP50-95':num(m['coco'].get('AP50_95')),'Precision':num(o['precision']),'Recall':num(o['recall']),'F1':num(o['F1']),
            'False alarms/image':num(o['false_alarms_per_image']),'Parameters':test['calculated']['parameters'],
            'Model size (ONNX MB)':mb(export.get('onnx_bytes')) if export else NA,
            'FLOPs/MACs (GMACs)':num(test['calculated']['gmacs'],3),
            'Median latency (GPU-only ms)':num(timing['inference']['median_ms'],2),
            'FPS (end-to-end)':num(timing['fps_from_mean_end_to_end'],1),
            'Median end-to-end latency (ms)':num(timing['end_to_end']['median_ms'],2),
            'Checkpoint size (MB)':mb(test['calculated']['checkpoint_file_bytes']),
            'Confidence threshold':o['confidence'],'Input size':test['input_size'],'Test images':o['images'],
            'Note':'' if export else 'ONNX export not found'}


def threshold_rows(name,root,split):
    metrics = load(root/split/name/'metrics.json')
    if not metrics:
        return []
    return [{'model':name,'split':split,'confidence':r['confidence'],'precision':num(r['precision']),'recall':num(r['recall']),
             'F1':num(r['F1']),'TP':r['TP'],'FP':r['FP'],'FN':r['FN'],'false_alarms_per_image':num(r['false_alarms_per_image']),
             'missed_flame_rate':num(r['missed_flame_rate'])} for r in metrics['measured']['fixed_thresholds']]


def charts(rows,root,plots,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    available = [r for r in rows if r['AP50']!=NA]
    if not available:
        return []
    written = []
    keys = ['AP50','AP75','mAP50-95','Precision','Recall','F1']
    figure,axis = plt.subplots(figsize=(10,5))
    width = .8/len(available)
    for i,row in enumerate(available):
        axis.bar([k+i*width for k in range(len(keys))],[row[k] for k in keys],width,label=row['Model'])
    axis.set_xticks([k+width*(len(available)-1)/2 for k in range(len(keys))],keys)
    axis.set(ylim=(0,1),title="Test split: accuracy (P/R/F1 at each model's operating confidence)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots/f'{prefix}_metrics.png',dpi=110)
    plt.close(figure)
    written.append(f'{prefix}_metrics.png')
    figure,axes = plt.subplots(1,2,figsize=(12,5))
    for name in MODELS:
        sweep_path = root/'test'/name/'threshold_sweep.csv'
        if not sweep_path.exists():
            continue
        sweep = list(csv.DictReader(sweep_path.open()))
        axes[0].plot([float(r['recall']) for r in sweep],[float(r['precision']) for r in sweep],label=MODELS[name])
        axes[1].plot([float(r['confidence']) for r in sweep],[float(r['false_alarms_per_image']) for r in sweep],label=MODELS[name])
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Test precision-recall (IoU 0.5, reporting only)',xlim=(0,1),ylim=(0,1))
    axes[1].set(xlabel='Confidence',ylabel='False alarms per image',title='Test false alarms vs confidence',yscale='log')
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(plots/f'{prefix}_pr_false_alarms.png',dpi=110)
    plt.close(figure)
    written.append(f'{prefix}_pr_false_alarms.png')
    return written


def table(rows,columns):
    lines = ['| '+' | '.join(columns)+' |','|'+'---|'*len(columns)]
    lines += ['| '+' | '.join(str(r.get(c,NA)) for c in columns)+' |' for r in rows]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--run',required=True,action='append',help='S-Lite training folder under checkpoints/ (repeatable)')
    parser.add_argument('--model',nargs=3,action='append',metavar=('KEY','EXPORT','LABEL'),
                        help='Evaluated folder under validation/ and test/, folder under exports/, table label (repeatable)')
    parser.add_argument('--reports',default='reports')
    parser.add_argument('--plots',default='plots')
    parser.add_argument('--plot-prefix',default='comparison')
    parser.add_argument('--title',default='Flame detection: YOLOX-Nano-ti-lite baseline vs YOLOX-S-ti-lite')
    parser.add_argument('--threshold-rule',default=DEFAULT_THRESHOLD_RULE)
    parser.add_argument('--limitation',action='append',help='Replaces the default training limitations (repeatable)')
    args = parser.parse_args()
    if args.model:
        MODELS.clear(); EXPORTS.clear()
        for key,export,label in args.model:
            MODELS[key],EXPORTS[key] = label,export
    root = Path(args.root)
    reports,plots = root/args.reports,root/args.plots
    reports.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    baseline_info = load(root/'configs'/'baseline_checkpoint.json')
    why = {name:baseline_info.get('reason',NA) if baseline_info and not baseline_info.get('found') else 'baseline evaluation not run'
           for name in MODELS if name.startswith('baseline')}
    rows = [model_row(name,root,why) for name in MODELS]
    columns = ['Model','AP50','AP75','mAP50-95','Precision','Recall','F1','False alarms/image','Parameters',
               'Model size (ONNX MB)','FLOPs/MACs (GMACs)','Median latency (GPU-only ms)','FPS (end-to-end)']
    extra = ['Model','Median end-to-end latency (ms)','Checkpoint size (MB)','Confidence threshold','Input size','Test images','Note']
    with (reports/'comparison.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=columns+extra[1:])
        writer.writeheader()
        writer.writerows(rows)
    thresholds = []
    for name in MODELS:
        for split in ('validation','test'):
            thresholds += threshold_rows(name,root,split)
    if thresholds:
        with (reports/'fixed_thresholds.csv').open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(thresholds[0]))
            writer.writeheader()
            writer.writerows(thresholds)
    status = load(root/'logs'/'status.json')
    resumes = load(root/'logs'/'resume_events.json') or []
    prepare = load(root/'manifests_v2'/'prepare_report.json')
    runs = {}
    for name in args.run:
        run = root/'checkpoints'/name
        history = load(run/'history.json') or []
        config = load(run/'config.json') or {}
        best = max(history,key=lambda r:r['AP50_95']) if history else None
        runs[name] = {'epochs_planned':config.get('epochs',NA),'epochs_completed':len(history),
                      'best_epoch':best['epoch'] if best else NA,'best_monitor_AP50_95':best['AP50_95'] if best else NA,
                      'monitor_history':[{k:r[k] for k in ('epoch','train_loss','AP50','AP50_95')} for r in history],
                      'seconds_per_epoch':[round(r['train_seconds'],1) for r in history],
                      'total_training_seconds':round(sum(r['train_seconds'] for r in history),1) if history else NA,
                      'initial_weights':config.get('checkpoint',NA),'batch':config.get('batch',NA),
                      'accumulate':config.get('accumulate',NA),'seed':42,
                      'optimizer':f"AdamW lr {config.get('lr',NA)} wd 5e-4, {config.get('warmup_updates',NA)}-update warmup, "
                                  'cosine to 0.1x, EMA, grad clip 10',
                      'patience':config.get('patience',NA),
                      'device':config.get('device_name',NA),'torch':config.get('torch',NA),'python':config.get('python',NA)}
    training = runs[args.run[-1]]|{'last_status':status,'resume_events':len(resumes)}
    written = charts(rows,root,plots,args.plot_prefix)
    complete = all(r['AP50']!=NA for r in rows)
    summary = {'comparison':rows,'fixed_thresholds':thresholds,'training_s_lite':training,'training_runs':runs,'resume_events':resumes,
               'baseline_checkpoint':baseline_info,'dataset':prepare['splits'] if prepare else NA,
               'exact_duplicates':prepare.get('exact_duplicates') if prepare else NA,
               'validation':{n:load(root/'validation'/n/'metrics.json') for n in MODELS},
               'test':{n:load(root/'test'/n/'metrics.json') for n in MODELS},
               'exports':{n:load(root/'exports'/EXPORTS[n]/'export.json') for n in MODELS},'plots':written}
    (reports/'metrics.json').write_text(json.dumps(summary,indent=2))
    s_test = next((load(root/'test'/n/'metrics.json') for n in MODELS if (root/'test'/n/'metrics.json').exists()),None)
    lines = [f'# {args.title}','',
             f'Status: **{"complete" if complete else "INCOMPLETE - see N/A entries"}**. '
             + '; '.join(f'{n}: {r["epochs_completed"]} of {r["epochs_planned"]} epochs' for n,r in runs.items()) + '.','',
             '## Protocol','',
             '- Data: Kaggle `yuulind/fasdd-cv-coco`, its own train/val/test splits; `fire` boxes are the `flame` target, smoke-only and '
             'unannotated images are negatives; quarantined images are listed in metrics.json.',
             '- Both models are evaluated on the same Kaggle validation and test images with the same code and matching rule '
             '(score-ordered one-to-one matching at IoU 0.5; COCO AP via pycocotools, maxDets 100, NMS 0.65).',
             f'- {args.threshold_rule}',
             '- Latency: Colab GPU, batch 1, float32, CUDA-synchronised, 10 warm-up images; GPU-only and end-to-end reported separately. '
             'Not AM62A performance.',
             '- **Fairness caveat:** the baseline was trained on a different (official FASDD) split at 416 px. Its training images may '
             'overlap the Kaggle validation/test images, which would favour the baseline. This could not be checked without its split '
             'files, so the comparison is indicative, not conclusive.','',
             '## Comparison (test split)','']
    lines += table(rows,columns)
    lines += ['',*table(rows,extra),'','## Fixed confidence thresholds','']
    lines += table(thresholds,['model','split','confidence','precision','recall','F1','TP','FP','FN','false_alarms_per_image','missed_flame_rate']) if thresholds else ['N/A']
    lines += ['','## S-Lite training','',f'```\n{json.dumps(runs,indent=2)}\n```',
              f'Resume events: {len(resumes)}.','','## Definitions','','```',
              s_test['definitions'] if s_test else NA,'```','','## Limitations','',
              *[f'- {text}' for text in (args.limitation or DEFAULT_LIMITATIONS)],
              '- No validation loss is computed by the training loop.',
              '- Exact duplicate images across the Kaggle splits are reported, not removed.',
              '- AM62A latency, TIDL compilation and board memory: NOT MEASURED (no board).',
              '- Still-image metrics; video alarm behaviour untested.']
    (reports/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(table(rows,columns)))
    print(f'wrote {reports}/report.md, comparison.csv, metrics.json; plots: {written}')


if __name__=='__main__':
    main()
