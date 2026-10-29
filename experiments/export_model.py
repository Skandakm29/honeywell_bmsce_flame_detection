"""Export a checkpoint to ONNX with the baseline's exporter and parity check (train.export_and_check).

Writes <output>/flame.onnx, export.json (validity, parity vs PyTorch on 4 validation inputs,
CPU onnxruntime timing, size, SHA-256) and model_contract.json. A parity failure is recorded
(parity_passed=false), not raised. The source checkpoint is only read.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
from s_model import ARCHITECTURE, build  # noqa: E402
from train import Flames, export_and_check  # noqa: E402
from train_s import tolerant_export  # noqa: E402

NAMES = {'nano':'TI YOLOX-Nano-ti-lite','s':ARCHITECTURE}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',choices=['nano','s'],required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',required=True,help='Manifest directory; val.json supplies parity inputs')
    parser.add_argument('--size',type=int,required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    model = build(args.arch)
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    result = tolerant_export(export_and_check)(model,Flames(Path(args.data)/'val.json',args.size),output,2)
    onnx = output/'flame.onnx'
    result.update({'onnx_bytes':onnx.stat().st_size if onnx.exists() else None,'source_checkpoint':args.checkpoint,
                   'arch':args.arch,'input_size':args.size})
    (output/'export.json').write_text(json.dumps(result,indent=2))
    contract = output/'model_contract.json'
    if contract.exists():
        content = json.loads(contract.read_text())
        content['architecture'] = NAMES[args.arch]
        contract.write_text(json.dumps(content,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='parity_error'},indent=2))


if __name__=='__main__':
    main()
