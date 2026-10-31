"""Short real-data training throughput measurement; never saves trained weights."""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import Flames, collate
from model import Detector, load_pretrained
from train import atomic_json, seed_everything, worker_init, update_ema


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'tensorlab', 'checkpoint', 'manifests', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    seed_everything(cfg['seed'])
    torch.set_num_threads(2)
    records = json.loads((Path(a.manifests)/'train.json').read_text())
    data = Flames(records, cfg['size'], augment=True, seed=cfg['seed'], mosaic=cfg['mosaic_probability'])
    loader = iter(DataLoader(data, batch_size=cfg['batch'], shuffle=True, collate_fn=collate,
                            num_workers=cfg['workers'], pin_memory=True, worker_init_fn=worker_init,
                            generator=torch.Generator().manual_seed(cfg['seed'])))
    model = Detector(a.tensorlab).cuda().train()
    load_pretrained(model, a.checkpoint)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    scaler = torch.amp.GradScaler('cuda', init_scale=cfg.get('amp_initial_scale', 128.), enabled=cfg['amp'])
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for step in range(16):
        torch.cuda.synchronize()
        start = time.perf_counter()
        images, targets, _, _ = next(loader)
        loaded = time.perf_counter()
        images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
            loss = sum(model(images, targets).values())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite benchmark loss')
        scaler.scale(loss/cfg['accumulate']).backward()
        if (step+1) % cfg['accumulate'] == 0:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(norm):
                updates = (step+1)//cfg['accumulate']
                update_ema(ema, model, cfg['ema_decay']*(1-math.exp(-updates/2000)))
            if not torch.isfinite(norm) and not cfg['amp']:
                raise FloatingPointError('Nonfinite benchmark gradient')
        torch.cuda.synchronize()
        row = dict(step=step+1, data_wait_seconds=loaded-start,
                   compute_seconds=time.perf_counter()-loaded, batch_seconds=time.perf_counter()-start,
                   loss=loss.item())
        rows.append(row)
        print(json.dumps(row), flush=True)
    measured = rows[4:]
    seconds = sum(r['batch_seconds'] for r in measured)/len(measured)
    result = dict(gpu=torch.cuda.get_device_name(), batch=cfg['batch'], warmup_batches=4,
                  timed_batches=len(measured), seconds_per_batch=seconds,
                  images_per_second=cfg['batch']/seconds,
                  estimated_full_epoch_hours=seconds*math.ceil(len(records)/cfg['batch'])/3600,
                  scope='Short estimate including EMA; excludes validation, checkpoint writes, worker startup and interruptions; not a completion guarantee',
                  rows=rows)
    atomic_json(result, a.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
