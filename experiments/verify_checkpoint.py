"""Check that a checkpoint is readable and matches an architecture before it is used.

--kind train: a resumable train.py snapshot (model, optimizer, epoch, next_batch, updates, RNG).
--kind weights: a best.pth-style {'model': state_dict}.
The weights must load strictly into the named architecture ('nano' baseline or 's'), which also
verifies that a baseline checkpoint really is YOLOX-Nano-ti-lite. Exits non-zero on any problem.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
from s_model import build  # noqa: E402

TRAIN_KEYS = {'model','optimizer','epoch','next_batch','global_updates','best_ap','history','training_contract'}


def verify(path,arch,kind):
    path = Path(path)
    state = torch.load(path,map_location='cpu',weights_only=True)
    missing = TRAIN_KEYS-set(state) if kind=='train' else {'model'}-set(state)
    if missing:
        raise ValueError(f'{path.name}: missing keys {sorted(missing)}')
    model = build(arch)
    model.load_state_dict(state['model'],strict=True)
    result = {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'arch':arch,'kind':kind,'strict_load':True,'parameters':sum(p.numel() for p in model.parameters())}
    if kind=='train':
        result.update({'epoch':state['epoch'],'next_batch':state['next_batch'],'optimizer_updates':state['global_updates'],
                       'best_monitor_AP50_95':state['best_ap'],'has_ema':'ema' in state,'has_rng_state':'torch_rng_state' in state,
                       'training_contract':state['training_contract']})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',required=True)
    parser.add_argument('--arch',choices=['nano','s'],required=True)
    parser.add_argument('--kind',choices=['train','weights'],required=True)
    parser.add_argument('--record',help='Optional JSON file to write the result to')
    args = parser.parse_args()
    result = verify(args.path,args.arch,args.kind)
    if args.record:
        Path(args.record).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='training_contract'},indent=2))


if __name__=='__main__':
    main()
