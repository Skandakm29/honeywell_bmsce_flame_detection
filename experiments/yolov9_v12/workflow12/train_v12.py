"""Steps 2 and 4c: fine-tunes with V10's own training code (train_v10.train unchanged: AdamW, EMA, AMP,
resume, per-epoch EMA checkpoints, monitor val + hn_val every epoch).

  python train_v12.py variant <a_data|b_zoom|c_zoom_paste>   start = frozen V10, data = manifests_v11
  python train_v12.py rec                                     start = step-3 start model, data = manifests_v12

Only the dataset class differs between variants (data11.Flames11 with that variant's augmentation dict;
null = exactly V10's pipeline). Validation always uses the plain pipeline.
"""
import hashlib
import sys
from pathlib import Path

import train_v10
from common import PACKAGE, read_json
from common12 import CONFIG, MANIFESTS11, MANIFESTS12, V10_FROZEN, V10_FROZEN_SHA, WORK
from data11 import factory

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config, augment, start, manifests, out):
    bank = manifests / 'flame_bank.npz'
    config = dict(config, augment_v12=augment,
                  v12_identity=dict(data11_sha256=sha(HERE / 'data11.py'), train_v12_sha256=sha(__file__),
                                    flame_bank_sha256=sha(bank) if bank.exists() else None))
    if augment and augment.get('paste_p', 0) > 0 and not bank.exists():
        raise RuntimeError(f'flame bank missing: {bank}')
    train_v10.Flames = factory(augment, bank)
    train_v10.train(config, str(PACKAGE / 'tensorlab'), str(start), str(manifests), str(out))
    summary = read_json(Path(out) / 'training_summary.json')
    if summary.get('stop_reason') != 'epoch_or_patience_limit':
        raise RuntimeError(f'{out}: training stopped early ({summary.get("stop_reason")}). Rerun the same step to resume.')
    print('TRAINING COMPLETE', out, flush=True)


def main():
    mode = sys.argv[1]
    base = dict(CONFIG['base'])
    if mode == 'variant':
        name = sys.argv[2]
        if sha(V10_FROZEN) != V10_FROZEN_SHA:
            raise RuntimeError('frozen V10 checksum mismatch')
        run(dict(base, variant=name), CONFIG['variants'][name], V10_FROZEN, MANIFESTS11, WORK / 'train_v12' / name)
    elif mode == 'rec':
        start = read_json(WORK / 'select_v12' / 'selection.json')['start_for_recordings']
        rec = CONFIG['recordings']
        config = dict(base, variant='rec_from_' + start['variant'], initial_checkpoint_sha256=sha(start['checkpoint']),
                      lr=rec['lr'], epochs=rec['epochs'], min_epochs=rec['min_epochs'],
                      warmup_updates=rec['warmup_updates'], close_mosaic_epochs=rec['close_mosaic_epochs'])
        augment = CONFIG['variants'].get(start['variant'])
        run(config, augment, start['checkpoint'], MANIFESTS12, WORK / 'train_v12' / 'rec')
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
