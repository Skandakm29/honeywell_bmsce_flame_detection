"""Real-resolution GPU train/export check, with recorded microbatch selection."""
import argparse
import gc
import json
from pathlib import Path

import torch

from data import Flames, collate
from finish import export
from model import Detector, load_pretrained
from train import atomic_json, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensorlab', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifests', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--max-batch', type=int, default=16)
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    selected = output/'selected_config.json'
    if selected.exists():
        cfg = json.loads(selected.read_text())
    records = json.loads((Path(a.manifests)/'train.json').read_text())
    positives = sorted((r for r in records if r['boxes']), key=lambda r: -len(r['boxes']))
    negatives = [r for r in records if not r['boxes']]
    if not positives or not negatives:
        raise RuntimeError('Preflight requires real positive and negative training examples')
    device = torch.device('cpu' if a.cpu else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Colab GPU runtime is required')
    torch.set_num_threads(2)
    memory = torch.cuda.get_device_properties(0).total_memory if device.type == 'cuda' else 0
    batch = cfg['batch'] if selected.exists() else (16 if memory >= 22*1024**3 else 8)
    batch = min(batch, a.max_batch)
    attempts = []
    while batch >= 1:
        model = optimizer = images = targets = losses = total = scaler = None
        try:
            seed_everything(cfg['seed'])
            model = Detector(a.tensorlab).to(device)
            initialized = load_pretrained(model, a.checkpoint)
            chosen = [((positives if i % 2 == 0 else negatives)[i//2 % len(positives if i % 2 == 0 else negatives)])
                      for i in range(batch)]
            dataset = Flames(chosen, cfg['size'], augment=True, mosaic=cfg['mosaic_probability'])
            images, targets, _, _ = collate([dataset[i] for i in range(batch)])
            images, targets = images.to(device), targets.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
            scaler = torch.amp.GradScaler('cuda', init_scale=cfg.get('amp_initial_scale', 128.),
                                         enabled=cfg['amp'] and device.type == 'cuda')
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled(), dtype=torch.float16):
                losses = model.train()(images, targets)
                total = sum(losses.values())
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite preflight loss')
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite preflight gradient; do not start long training')
            scaler.step(optimizer)
            scaler.update()
            attempts.append(dict(batch=batch, result='passed'))
            break
        except torch.cuda.OutOfMemoryError:
            attempts.append(dict(batch=batch, result='CUDA OOM'))
            if selected.exists():
                raise RuntimeError('Saved recipe no longer fits. Use a new run directory for a different batch size.')
            batch //= 2
        finally:
            model = optimizer = images = targets = losses = total = scaler = None
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if batch < 1:
        raise RuntimeError('YOLOv9 did not fit even batch 1')
    cfg['batch'], cfg['accumulate'] = batch, 32//batch
    atomic_json(cfg, selected)
    model = Detector(a.tensorlab).eval()
    initialized = load_pretrained(model, a.checkpoint)
    val = json.loads((Path(a.manifests)/'val.json').read_text())
    indices = [next(i for i, r in enumerate(val) if r['boxes']),
               next(i for i, r in enumerate(val) if not r['boxes'])]
    validation = Flames([val[i] for i in indices], cfg['size'])
    exported = export(model, validation, output/'export_smoke')
    atomic_json(dict(passed=True, device=str(device), gpu_memory_bytes=memory, batch_attempts=attempts,
                     effective_batch=cfg['batch']*cfg['accumulate'], pretrained=initialized,
                     export=exported, scope='Engineering preflight only; no flame-quality metrics'),
                output/'preflight.json')


if __name__ == '__main__':
    main()
