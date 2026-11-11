"""Step 2: fine-tune flame -> flame + obstruction with V10's own training loop (train_v10.train, unchanged:
AdamW, EMA, AMP, resume, per-epoch EMA checkpoints, monitor validation + hn_val every epoch).

Exactly five names inside train_v10 are swapped, the same mechanism V12 used for its dataset:
  Flames                    -> data_w4.Flames2 (class ids + synthetic obstructions + optional V12 augmentation)
  Detector                  -> Detector(classes=2)
  load_trained              -> expand_w4.load_expanded (1-class checkpoint grown to 2 classes, flame copied exactly)
  evaluate                  -> metrics_w4.evaluate2 (flame on clean images, V10 definitions + obstruction)
  negative_false_positives  -> metrics_w4.negative_false_positives2 (V10 flame counts + obstruction counts)

  python train_w4.py            (paths from common_w4: V10_PACKAGE, V10_WORK, W4_START_CHECKPOINT)
  python train_w4.py --cpu --manifests <dir> --output <dir> --start <ckpt>   local plumbing test only
"""
import argparse
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config, tensorlab, start, manifests, out, cpu=False):
    import train_v10
    from model import Detector
    from data_w4 import factory
    from expand_w4 import load_expanded
    from metrics_w4 import evaluate2, negative_false_positives2

    bank = Path(manifests) / 'flame_bank.npz'
    augment = config.get('augment')
    if augment and augment.get('paste_p', 0) > 0 and not bank.exists():
        raise RuntimeError(f'flame bank missing: {bank}')
    config = dict(config, initial_checkpoint_sha256=sha(start),
                  w4_identity={name: sha(HERE / name) for name in
                               ('data_w4.py', 'occluders.py', 'expand_w4.py', 'metrics_w4.py', 'train_w4.py')})
    train_v10.Flames = factory(config['occluder'], augment, bank if bank.exists() else None)
    train_v10.Detector = lambda tensorlab_path: Detector(tensorlab_path, classes=len(config['classes']))
    train_v10.load_trained = load_expanded
    train_v10.evaluate = evaluate2
    train_v10.negative_false_positives = negative_false_positives2
    train_v10.train(config, str(tensorlab), str(start), str(manifests), str(out), cpu=cpu)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cpu', action='store_true', help='local plumbing test only')
    p.add_argument('--manifests')
    p.add_argument('--output')
    p.add_argument('--start')
    p.add_argument('--config')
    a = p.parse_args()
    from common import read_json
    from common_w4 import CONFIG, MANIFESTS_W4, START, START_SHA256, TENSORLAB, TRAIN_DIR
    config = read_json(a.config) if a.config else CONFIG
    start = Path(a.start) if a.start else START
    if not a.start and sha(start) != START_SHA256:
        raise RuntimeError(f'start checkpoint changed since preflight: {start}')
    output = Path(a.output) if a.output else TRAIN_DIR
    run(dict(config['base'], classes=config['classes'], occluder=config['occluder'], augment=config.get('augment')),
        TENSORLAB, start, a.manifests or MANIFESTS_W4, output, cpu=a.cpu)
    summary = read_json(output / 'training_summary.json')
    if summary.get('stop_reason') != 'epoch_or_patience_limit':
        raise RuntimeError(f'training stopped early ({summary.get("stop_reason")}). Rerun the same step to resume.')
    print('TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
