"""Compare uninterrupted CPU training with a simulated optimizer-boundary disconnect.

Use only a tiny engineering fixture. This is not model-quality evaluation.
"""
import argparse
import json
from pathlib import Path

import torch

import train as workflow


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensorlab', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifests', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    cfg.update(size=64, epochs=1, batch=2, accumulate=1, workers=0, amp=False,
               warmup_epochs=1, close_mosaic_epochs=0, min_epochs=1, checkpoint_every_updates=1)
    records = json.loads((Path(a.manifests)/'train.json').read_text())
    indices = list(range(min(4, len(records))))
    if len(indices) < 4:
        raise ValueError('Need at least four fixture training records')
    workflow.train(cfg, a.tensorlab, a.checkpoint, a.manifests, output/'continuous', train_indices=indices, cpu=True)
    original_save = workflow.save

    class Disconnect(Exception):
        pass

    def interrupt_after_checkpoint(state, path):
        original_save(state, path)
        if Path(path).name == 'last.pth' and state['epoch'] == 0 and state['next_batch'] == 1:
            raise Disconnect('Simulated Colab disconnect after first optimizer update')

    if not (output/'resumed/last.pth').exists():
        workflow.save = interrupt_after_checkpoint
        try:
            workflow.train(cfg, a.tensorlab, a.checkpoint, a.manifests, output/'resumed', train_indices=indices, cpu=True)
        except Disconnect:
            print('Simulated interruption checkpoint saved', flush=True)
        finally:
            workflow.save = original_save
    workflow.train(cfg, a.tensorlab, a.checkpoint, a.manifests, output/'resumed', train_indices=indices, cpu=True)
    continuous = torch.load(output/'continuous/last.pth', map_location='cpu', weights_only=False)
    resumed = torch.load(output/'resumed/last.pth', map_location='cpu', weights_only=False)
    for group in ('model', 'ema'):
        for name, value in continuous[group].items():
            torch.testing.assert_close(value, resumed[group][name], rtol=0, atol=0)
    assert continuous['updates'] == resumed['updates']
    workflow.atomic_json(dict(passed=True, model_weights_bitwise_equal=True, ema_weights_bitwise_equal=True,
                              updates=continuous['updates'], scope='Same-CPU deterministic fixture only; not cross-GPU identity'),
                         output/'resume_check.json')
    print('Interrupted/resumed model and EMA are bitwise identical to uninterrupted training', flush=True)


if __name__ == '__main__':
    main()
