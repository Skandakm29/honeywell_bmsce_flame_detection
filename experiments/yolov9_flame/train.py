"""Resumable GPU training, validation-only LR screening, and evidence collection."""
import argparse
import copy
import json
import math
import os
import platform
import pickle
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Flames, collate, sha256
from metrics import evaluate
from model import Detector, TI_COMMIT, load_pretrained


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def save(state, path):
    path = Path(path)
    temporary = path.with_suffix('.pth.tmp')
    torch.save(state, temporary)
    if path.exists() and path.name == 'last.pth':
        os.replace(path, path.with_name('previous.pth'))
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def worker_init(_):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


@torch.no_grad()
def update_ema(ema, model, decay):
    # Construct the mapping once, rather than traversing the entire model once
    # for each tensor. Keep tensor order and arithmetic identical for reproducibility.
    current_state = model.state_dict()
    for key, value in ema.state_dict().items():
        current = current_state[key].detach()
        if value.is_floating_point():
            value.mul_(decay).add_(current, alpha=1-decay)
        else:
            value.copy_(current)


def train(config, tensorlab, checkpoint, manifests, work, *, train_indices=None, cpu=False):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    seed_everything(config['seed'])
    torch.set_num_threads(2)
    device = torch.device('cpu' if cpu else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime; refusing full CPU training')
    manifests = Path(manifests)
    fingerprints = {name: sha256(manifests / f'{name}.json') for name in ('train', 'val')}
    identity = {'config': config, 'manifests': fingerprints, 'ti_commit': TI_COMMIT,
                'initial_checkpoint_sha256': sha256(checkpoint), 'train_indices': train_indices,
                'source_sha256': {name: sha256(Path(__file__).with_name(name))
                                  for name in ('model.py', 'data.py', 'metrics.py', 'train.py')}}
    identity_path = work / 'run_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError('Run inputs/config changed. Use a new run directory, not resume.')
    atomic_json(identity, identity_path)
    completed = work / 'training_summary.json'
    if completed.exists() and (work / 'best.pth').exists():
        return json.loads(completed.read_text())['best_monitor_AP50_95']
    records = json.loads((manifests / 'train.json').read_text())
    if train_indices is not None:
        records = [records[i] for i in train_indices]
    validation = Flames(json.loads((manifests / 'val.json').read_text()), config['size'])
    model = Detector(tensorlab).to(device)
    pretrained = load_pretrained(model, checkpoint)
    atomic_json(pretrained, work / 'pretrained_loading.json')
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    # Exclude biases and normalization parameters from weight decay.
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or name.endswith('.bias') else decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': config['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0.}], lr=config['lr'])
    scaler = torch.amp.GradScaler('cuda', init_scale=config.get('amp_initial_scale', 128.),
                                enabled=config['amp'] and device.type == 'cuda')
    batch_count = math.ceil(len(records) / config['batch'])
    updates_per_epoch = math.ceil(batch_count / config['accumulate'])
    total_updates = config['epochs'] * updates_per_epoch
    warmup = max(1, config['warmup_epochs'] * updates_per_epoch)
    start_epoch, skip_batches, updates, best, stale, history = 0, 0, 0, -1., 0, []
    resume = None
    for candidate in (work / 'last.pth', work / 'previous.pth'):
        if candidate.exists():
            try:
                # Own trusted run checkpoints include Python/NumPy RNG state.
                resume = torch.load(candidate, map_location='cpu', weights_only=False)
                if resume['identity'] != identity:
                    raise RuntimeError('Resume checkpoint identity differs')
                break
            except (OSError, EOFError, RuntimeError, pickle.UnpicklingError, KeyError) as error:
                print(f'Checkpoint recovery: {candidate}: {error}', flush=True)
                resume = None
    if any((work / name).exists() for name in ('last.pth', 'previous.pth')) and resume is None:
        raise RuntimeError('No readable matching resume checkpoint; refusing to silently restart')
    if resume:
        model.load_state_dict(resume['model'], strict=True)
        ema.load_state_dict(resume['ema'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        start_epoch, skip_batches, updates = resume['epoch'], resume['next_batch'], resume['updates']
        best, stale, history = resume['best'], resume['stale'], resume['history']
        random.setstate(resume['rng_python'])
        np.random.set_state(resume['rng_numpy'])
        torch.set_rng_state(resume['rng_torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(resume['rng_cuda'])
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
                   'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                   'parameters_training_including_aux': sum(p.numel() for p in model.parameters()),
                   'ti_commit': TI_COMMIT, 'resume': bool(resume)}
    atomic_json(environment, work / 'environment.json')
    try:
        (work / 'pip_freeze.txt').write_text(subprocess.check_output(
            [os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    except subprocess.CalledProcessError:
        pass

    def snapshot(epoch, next_batch):
        save(dict(identity=identity, model=model.state_dict(), ema=ema.state_dict(),
                  optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                  next_batch=next_batch, updates=updates, best=best, stale=stale, history=history,
                  running_loss_sums=sums if next_batch else {}, running_loss_batches=seen if next_batch else 0,
                  rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                  rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []), work / 'last.pth')

    for epoch in range(start_epoch, config['epochs']):
        mosaic = config['mosaic_probability'] if epoch < config['epochs']-config['close_mosaic_epochs'] else 0.
        dataset = Flames(records, config['size'], augment=True, seed=config['seed'], mosaic=mosaic)
        dataset.epoch = epoch
        loader = DataLoader(dataset, batch_size=config['batch'], shuffle=True, collate_fn=collate,
                            num_workers=config['workers'], pin_memory=device.type == 'cuda',
                            worker_init_fn=worker_init, persistent_workers=False,
                            generator=torch.Generator().manual_seed(config['seed']+epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = dict(resume.get('running_loss_sums', {})) if resume and epoch == start_epoch and skip_batches else {}
        seen = resume.get('running_loss_batches', 0) if resume and epoch == start_epoch and skip_batches else 0
        started = time.perf_counter()
        for batch_index, (images, targets, _, _) in enumerate(loader):
            if batch_index < skip_batches:
                continue
            if updates < warmup:
                multiplier = .1 + .9 * updates / warmup
            else:
                progress = min(1., (updates-warmup) / max(1, total_updates-warmup-1))
                multiplier = config['min_lr_ratio'] + (1-config['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:
                group['lr'] = config['lr'] * multiplier
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            group_start = (batch_index // config['accumulate']) * config['accumulate']
            group_size = min(config['accumulate'], batch_count-group_start)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled(), dtype=torch.float16):
                components = model(images, targets)
                loss = sum(components.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, batch {batch_index}')
            scaler.scale(loss / group_size).backward()
            for key, value in components.items():
                sums[key] = sums.get(key, 0.) + value.item()
            seen += 1
            if (batch_index+1) % config['accumulate'] == 0 or batch_index+1 == batch_count:
                scaler.unscale_(optimizer)
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                if not torch.isfinite(gradient):
                    # GradScaler may recover FP16 overflow, but never update EMA from skipped weights.
                    if not scaler.is_enabled():
                        raise FloatingPointError('Nonfinite gradients')
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    print('AMP overflow: skipped optimizer update', flush=True)
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                decay_value = config['ema_decay'] * (1-math.exp(-updates/2000))
                update_ema(ema, model, decay_value)
                if updates % config['checkpoint_every_updates'] == 0:
                    snapshot(epoch, batch_index+1)
            if (batch_index+1) % 20 == 0:
                print(json.dumps(dict(epoch=epoch+1, batch=batch_index+1, total_batches=batch_count,
                                      loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                                      elapsed_seconds=time.perf_counter()-started,
                                      processed_batches=seen)), flush=True)
        skip_batches = 0
        metrics, _ = evaluate(ema, validation, config['batch'], config['workers'])
        score = metrics['coco']['AP50_95']
        if score is None:
            raise RuntimeError('Validation has no flame annotations')
        row = dict(epoch=epoch+1, loss={k: v/max(seen, 1) for k, v in sums.items()},
                   monitor_validation=metrics, seconds=time.perf_counter()-started, mosaic=mosaic)
        history.append(row)
        if score > best:
            best, stale = score, 0
            save(dict(model=ema.state_dict(), epoch=epoch+1, metrics=metrics, identity=identity), work / 'best.pth')
        else:
            stale += 1
        atomic_json(history, work / 'history.json')
        snapshot(epoch+1, 0)
        print(json.dumps(row), flush=True)
        if epoch+1 >= config['min_epochs'] and stale >= config['patience']:
            break
    best_state = torch.load(work / 'best.pth', map_location='cpu', weights_only=False)
    atomic_json(dict(completed_epochs=len(history), best_epoch=best_state['epoch'],
                     best_monitor_AP50_95=best_state['metrics']['coco']['AP50_95'],
                     complete=True), work / 'training_summary.json')
    return best_state['metrics']['coco']['AP50_95']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--tensorlab', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifests', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tune', action='store_true', help='Run validation-only LR pilots before final training')
    p.add_argument('--cpu', action='store_true', help='Only for local integration tests')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    if a.tune:
        records = json.loads((Path(a.manifests)/'train.json').read_text())
        indices = random.Random(cfg['seed']).sample(range(len(records)), min(len(records), cfg['pilot_images']))
        trials = []
        for lr in cfg['pilot_lrs']:
            pilot = {**cfg, 'lr': lr, 'epochs': cfg['pilot_epochs'], 'warmup_epochs': 1,
                     'close_mosaic_epochs': 1, 'min_epochs': cfg['pilot_epochs']}
            score = train(pilot, a.tensorlab, a.checkpoint, a.manifests,
                          output / f'pilot_lr_{lr:g}', train_indices=indices, cpu=a.cpu)
            trials.append(dict(lr=lr, monitor_AP50_95=score))
            atomic_json(trials, output/'lr_trials.json')
        cfg['lr'] = max(trials, key=lambda r: r['monitor_AP50_95'])['lr']
        atomic_json(dict(selected_lr=cfg['lr'], trials=trials,
                         scope='Short validation-only screening, not a globally optimal hyperparameter claim'),
                    output/'lr_selection.json')
    train(cfg, a.tensorlab, a.checkpoint, a.manifests, output/'full', cpu=a.cpu)


if __name__ == '__main__':
    main()
