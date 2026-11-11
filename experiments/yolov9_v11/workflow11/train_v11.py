"""Step 3 (pilot) and step 4 (training) for V11.

Training is V10's train_v10.train() unchanged (same optimizer, EMA, AMP, resume logic, per-epoch EMA checkpoints,
per-epoch monitor-val and hn_val), with exactly two differences:
  - initialization: official TI COCO weights (as V9), not V9/V10 weights   [config "initialization": "coco"]
  - the training dataset class is data11.Flames11 (V10 pipeline + zoom-out, small-flame paste, degradations).
The run identity (config + data11.py hash + bank hash + manifest hashes) blocks resuming with changed inputs.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import train_v10
from common import PACKAGE
from data import collate
from data11 import factory
from paths11 import COCO_WEIGHTS, WORK

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(config):
    manifests = WORK / 'manifests_v11'
    bank = manifests / 'flame_bank.npz'
    config = dict(config, v11_identity=dict(data11_sha256=sha(HERE / 'data11.py'), train_v11_sha256=sha(__file__),
                                            flame_bank_sha256=sha(bank)))
    train_v10.Flames = factory(config['augment_v11'], bank)   # training AND validation go through this; val = plain
    return config, manifests


def pilot(config, batches=200):
    """Timing + plumbing on the real data: CPU augmentation throughput and GPU step time. Writes nothing to train_v11."""
    from torch.utils.data import DataLoader
    from model import Detector
    config, manifests = prepare(config)
    records = json.loads((manifests / 'train.json').read_text())
    ds = train_v10.Flames(records, 640, augment=True, seed=config['seed'], mosaic=config['mosaic_probability'])
    loader = DataLoader(ds, batch_size=config['batch'], shuffle=True, collate_fn=collate, num_workers=config['workers'],
                        worker_init_fn=train_v10.worker_init, generator=torch.Generator().manual_seed(0))
    model = Detector(str(PACKAGE / 'tensorlab')).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    data_wait, step, started, n = 0., 0., time.perf_counter(), 0
    t = time.perf_counter()
    for images, targets, _, _ in loader:
        data_wait += time.perf_counter() - t
        s = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            loss = sum(model(images.cuda(non_blocking=True), targets.cuda(non_blocking=True)).values())
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite pilot loss')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step += time.perf_counter() - s
        n += 1
        t = time.perf_counter()
        if n == batches:
            break
    per_batch = (time.perf_counter() - started) / n
    epoch_min = per_batch * (len(records) / config['batch']) / 60
    result = dict(batches=n, seconds_per_batch=per_batch, data_wait_share=data_wait / (data_wait + step),
                  estimated_minutes_per_epoch_training_only=epoch_min,
                  estimated_hours_40_epochs_plus_eval=(epoch_min + 4) * config['epochs'] / 60,
                  last_loss=float(loss), gpu=torch.cuda.get_device_name(0))
    out = WORK / 'pilot_v11'
    out.mkdir(exist_ok=True)
    (out / 'pilot.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)
    if result['estimated_hours_40_epochs_plus_eval'] > config['max_training_seconds'] / 3600:
        print('WARNING: estimate exceeds max_training_seconds; report to the owner, do NOT edit the config.', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['pilot', 'train'])
    a = p.parse_args()
    config = json.loads((HERE / 'config_v11.json').read_text())
    if a.mode == 'pilot':
        return pilot(config)
    if sha(COCO_WEIGHTS) != train_v10.load_pretrained.__globals__['PRETRAINED_SHA256']:
        raise RuntimeError('COCO weights checksum mismatch')
    config, manifests = prepare(config)
    train_v10.train(config, str(PACKAGE / 'tensorlab'), str(COCO_WEIGHTS), str(manifests), str(WORK / 'train_v11'))
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
