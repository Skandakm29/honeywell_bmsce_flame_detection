"""Train YOLOX-S-ti-lite with the baseline training loop (scripts/train.py), unmodified.

Only the model class is swapped. Every train.py argument and --config/--resume behaviour is
unchanged; train.py itself refuses to start from pretrained weights when the output folder
already holds last.pth without --resume. Added around it, without editing train.py:
- every checkpoint write (already atomic: temp file + rename) also writes STATUS_DIR/status.json
  with the checkpointed epoch, next batch, optimizer updates and best monitor AP;
- a --resume start is appended to STATUS_DIR/resume_events.json;
- an ONNX parity failure at the end is recorded in export.json (parity_passed=false) instead
  of aborting, so evaluation still runs;
- model_contract.json gets the correct architecture name; sources and provenance are recorded
  (INIT_SOURCE overrides the recorded origin of --checkpoint, e.g. when continuing from a trained run).
The learning-rate schedule is a pure function of the optimizer-update count stored in the
checkpoint, so resuming restores the scheduler position exactly.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
from s_model import ARCHITECTURE, ROOT, SFlameDetector  # noqa: E402
import train  # noqa: E402

VENDOR_REVISION = 'cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6'
CHECKPOINT_SOURCE = ('http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/'
                     'edgeai-yolox/yolox-s-ti-lite_39p1_57p9_checkpoint.pth')


def arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    known,_ = parser.parse_known_args()
    config = Path(known.config)
    config = json.loads((config if config.is_absolute() else ROOT/config).read_text())
    known.output = ROOT/(known.output or config['output'])
    known.checkpoint = ROOT/(known.checkpoint or config['checkpoint'])
    return known


def write_json(path,content):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content,indent=2))
    temporary.replace(path)


def status_checkpoint(original,status):
    """Wrap train.atomic_checkpoint: after each last.pth write, record where training stands."""
    def save(state,path):
        original(state,path)
        if status and path.name=='last.pth':
            write_json(status/'status.json',{
                'time':time.strftime('%Y-%m-%d %H:%M:%S'),'checkpoint':str(path),'checkpoint_bytes':path.stat().st_size,
                # train.py snapshots store the 0-based epoch in progress, or epoch+1 with next_batch 0
                # at an epoch boundary; either way 'epoch' equals the number of completed epochs.
                'completed_epochs':state['epoch'],
                'epoch_in_progress':None if state['next_batch']==0 else state['epoch']+1,
                'next_batch':state['next_batch'],'optimizer_updates':state['global_updates'],
                'best_monitor_AP50_95':state['best_ap'],'epochs_with_validation':len(state['history'])})
    return save


def tolerant_export(original):
    """Record an ONNX parity failure in export.json instead of aborting before evaluation."""
    def export(model,dataset,directory,threads):
        try:
            result = original(model,dataset,directory,threads)
        except AssertionError as error:
            result = {'onnx_written':(directory/'flame.onnx').exists(),'parity_passed':False,
                      'parity_atol':.002,'parity_rtol':.001,'parity_error':str(error)[:3000]}
            print('ONNX parity check FAILED; recorded in export.json, continuing',flush=True)
        else:
            result['parity_passed'] = True
        (directory/'export.json').write_text(json.dumps(result,indent=2))
        return result
    return export


def main():
    args = arguments()
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    status = Path(os.environ['STATUS_DIR']) if os.environ.get('STATUS_DIR') else None
    if status:
        status.mkdir(parents=True,exist_ok=True)
        if args.resume:
            events = status/'resume_events.json'
            history = json.loads(events.read_text()) if events.exists() else []
            previous = json.loads((status/'status.json').read_text()) if (status/'status.json').exists() else None
            history.append({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'resumed_from':args.resume,'state_before':previous})
            write_json(events,history)
    sources = [*EXP.glob('*.py'),*EXP.glob('*.sh'),*EXP.glob('configs/*.json'),ROOT/'scripts/train.py',ROOT/'scripts/flame_model.py']
    write_json(output/'experiment_sources.json',
        {'architecture':ARCHITECTURE,'command':sys.argv,'ti_edgeai_yolox_revision':VENDOR_REVISION,
         'pretrained_checkpoint':{'path':str(args.checkpoint),'source':os.environ.get('INIT_SOURCE',CHECKPOINT_SOURCE),
                                  'sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()},
         'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}})
    train.FlameDetector = SFlameDetector
    train.atomic_checkpoint = status_checkpoint(train.atomic_checkpoint,status)
    train.export_and_check = tolerant_export(train.export_and_check)
    train.main()
    contract = output/'model_contract.json'
    if contract.exists():
        content = json.loads(contract.read_text())
        content['architecture'] = ARCHITECTURE
        contract.write_text(json.dumps(content,indent=2))


if __name__=='__main__':
    main()
