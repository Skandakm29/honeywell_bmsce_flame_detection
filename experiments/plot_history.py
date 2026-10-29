"""Training curves from a train.py run folder (history.json + config.json).

Learning rate is not logged by train.py; it is recalculated from the run's own schedule
(warmup then cosine to 0.1x) at each epoch end and labelled as calculated. Validation loss is
not computed by train.py and is therefore absent. Checkpoint selection: best monitor AP50:95.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def learning_rate(config,updates):
    total = config['epochs']*math.ceil((config['max_batches'] or math.ceil(config['train_images']/config['batch']))/config['accumulate'])
    if updates<config['warmup_updates']:
        return config['lr']*(updates+1)/config['warmup_updates']
    if not config['cosine']:
        return config['lr']
    fraction = min(1,max(0,(updates-config['warmup_updates'])/max(1,total-config['warmup_updates'])))
    return config['lr']*(.1+.9*(1+math.cos(math.pi*fraction))/2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--plots',help='Also save the figure and CSV into this folder')
    args = parser.parse_args()
    run = Path(args.run)
    history = json.loads((run/'history.json').read_text())
    config = json.loads((run/'config.json').read_text())
    best = max(history,key=lambda row:row['AP50_95'])
    rows = []
    for row in history:
        point = row['operating_point']
        rows.append({'epoch':row['epoch'],'train_loss':row['train_loss'],'AP50_95':row['AP50_95'],'AP50':row['AP50'],
                     'AP_small':row['AP_small'],'AR100':row['AR100'],'precision':point['precision'],'recall':point['recall'],
                     'F1':point['F1'],'confidence':point['confidence'],'train_seconds':row['train_seconds'],
                     'learning_rate_calculated':learning_rate(config,row['global_updates']-1),
                     'best':row['epoch']==best['epoch']})
    with (run/'history.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in rows]
    figure,axes = plt.subplots(2,2,figsize=(11,8))
    axes[0,0].plot(epochs,[r['train_loss'] for r in rows],marker='o')
    axes[0,0].set(title='Training loss (mean per batch)',xlabel='Epoch')
    for key in ('AP50','AP50_95','AP_small','AR100'):
        axes[0,1].plot(epochs,[r[key] for r in rows],marker='o',label=key)
    axes[0,1].axvline(best['epoch'],color='red',ls='--',label=f'selected epoch {best["epoch"]}')
    axes[0,1].set(title='Monitor validation (COCO)',xlabel='Epoch')
    axes[0,1].legend()
    for key in ('precision','recall','F1'):
        axes[1,0].plot(epochs,[r[key] for r in rows],marker='o',label=key)
    axes[1,0].set(title="At each epoch's max-F1 confidence (monitor val)",xlabel='Epoch')
    axes[1,0].legend()
    axes[1,1].plot(epochs,[r['learning_rate_calculated'] for r in rows],marker='o')
    axes[1,1].set(title='Learning rate at epoch end (calculated)',xlabel='Epoch')
    figure.suptitle(f'{run.name}: selection = best monitor AP50:95; validation loss not computed')
    figure.tight_layout()
    figure.savefig(run/'training_curves.png',dpi=110)
    if args.plots:
        Path(args.plots).mkdir(parents=True,exist_ok=True)
        figure.savefig(Path(args.plots)/'training_curves.png',dpi=110)
        (Path(args.plots)/'history.csv').write_text((run/'history.csv').read_text())
    print(json.dumps({'epochs':len(rows),'selected_epoch':best['epoch'],'selected_AP50_95':best['AP50_95'],
                      'selected_AP50':best['AP50']},indent=2))


if __name__=='__main__':
    main()
