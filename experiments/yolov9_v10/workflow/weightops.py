"""Training-free experiments on frozen V10. Run with V10_WORK pointing at the lab inputs."""
import argparse
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch
from torchvision.ops import box_iou

from common import PACKAGE, detect_records, load_detector, read_json, sha256, write_json
from data import Flames
from finish import export
from metrics import evaluate
from select_v10 import nuisance, small_recall, small_slice

FROZEN_HASH = 'd046dab613faa72ec650d37dc40d0740ff50ac09df4eb97919d10cfd532b721d'
SOUPS = {'A1': range(6, 11), 'A2': range(8, 11), 'A3': range(5, 8)}
ALPHAS = {'B1': .5, 'B2': .7, 'B3': .85}
DAMPING = [(k, scale) for scale in (.5, 0.) for k in (4, 8, 16)]
MIN_FP_ANCHORS = 30          # audit fix: fewer false-alarm samples at a scale -> no edit there (ranking would be noise)
REFERENCE_TOLERANCE = 5e-4   # audit fix: R0/R1 must reproduce select/selection.json (10x below decision margins)


def weights(path):
    return torch.load(path, map_location='cpu', weights_only=False)['model']


def average(paths):
    result = weights(paths[0])
    for path in paths[1:]:
        other = weights(path)
        for key, tensor in result.items():
            if tensor.is_floating_point():
                tensor.add_(other[key])
    for tensor in result.values():
        if tensor.is_floating_point():
            tensor.div_(len(paths))
    return result


def interpolate(first, last, alpha):
    result = weights(first)
    end = weights(last)
    if alpha == 1:
        return end
    for key, tensor in result.items():
        if tensor.is_floating_point():
            tensor.lerp_(end[key], alpha)
    return result


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        a = state[key].contiguous().numpy()
        digest.update(json.dumps([key, str(a.dtype), a.shape]).encode())
        digest.update(memoryview(a).cast('B'))
    return digest.hexdigest()


def coordinate(index, shapes):
    for level, (height, width) in enumerate(shapes):
        if index < height * width:
            return level, *divmod(index, width)
        index -= height * width
    raise IndexError(index)


def sample(records, maximum):
    unique = {r['image']: r for r in records}
    return random.Random(42).sample(list(unique.values()), min(maximum, len(unique)))


@torch.inference_mode()
def measure(model, train, limit):
    maximum = min(limit or 800, 800)
    populations = {
        'fp': sample((r for r in train if not r['boxes'] and r.get('hard')
                      and r['category'] != 'synthetic_screen'), maximum),
        'tp': sample((r for r in train if r['boxes'] and r.get('source', 'fasdd') == 'fasdd'), maximum),
    }
    device = next(model.parameters()).device
    layers = [h.class_conv[2] for h in model.bbox_head.heads]
    features = {}
    def hook(level):
        def capture(module, args):
            features[level] = args[0]
        return capture
    hooks = [layer.register_forward_pre_hook(hook(i)) for i, layer in enumerate(layers)]
    summaries = {}
    try:
        for label, records in populations.items():
            stats = [[0, torch.zeros(layer.in_channels, dtype=torch.float64),
                      torch.zeros(layer.in_channels, dtype=torch.float64)] for layer in layers]
            for tensor, target, _, _ in Flames(records, 640):
                output = model(tensor.unsqueeze(0).to(device))[0]
                scores = output[:, 4]
                if label == 'fp':
                    indices = scores.argmax().reshape(1)
                    indices = indices[scores[indices] >= .2]
                else:
                    indices = torch.where(scores >= .4)[0]
                    gt = target[target[:, 0] >= 0, 1:].to(device)
                    indices = indices[box_iou(output[indices, :4], gt).amax(1) >= .5]
                offset = 0
                for i, layer in enumerate(layers):
                    feature = features[i][0]
                    channels, height, width = feature.shape
                    assert channels == layer.in_channels
                    local = indices[(indices >= offset) & (indices < offset + height * width)] - offset
                    offset += height * width
                    if local.numel():
                        values = (feature.flatten(1)[:, local] * layer.weight.reshape(-1, 1)).cpu().double()
                        stats[i][0] += len(local)
                        stats[i][1] += values.sum(1)
                        stats[i][2] += values.square().sum(1)
                assert offset == len(output)
            summaries[label] = stats
    finally:
        for h in hooks:
            h.remove()
    levels = []
    for fp, tp in zip(summaries['fp'], summaries['tp']):
        ranked, skipped = [], None
        if fp[0] < MIN_FP_ANCHORS:
            skipped = f'only {fp[0]} false-alarm anchors (< {MIN_FP_ANCHORS}); no edit at this scale'
        elif tp[0] and fp[0] + tp[0] > 2:
            f, t = fp[1] / fp[0], tp[1] / tp[0]
            variance = (fp[2] - fp[1]*f + tp[2] - tp[1]*t) / (fp[0] + tp[0] - 2)
            # audit fix: only channels that push false alarms up AND favour false alarms over real flames
            valid = (f > 0) & (f > t) & (variance > 0)
            for k in torch.where(valid)[0].tolist():
                ranked.append(dict(channel=k, score=float((f[k]-t[k])/variance[k].sqrt())))
            ranked.sort(key=lambda item: -item['score'])
        levels.append(dict(fp_anchors=fp[0], tp_anchors=tp[0], ranked=ranked, skipped=skipped))
    return levels


def damp(state, levels, k, factor):
    for i, level in enumerate(levels):
        key = f'bbox_head.heads.{i}.class_conv.2.weight'
        indices = [item['channel'] for item in level['ranked'][:k]]
        state[key][:, indices] *= factor
    return state


def judge(row, v9, v10):
    score = row['nuisance']
    first = all(row[k] >= v9[k] - .01 for k in ('AP50', 'recall', 'small_recall'))
    first = first and v9['nuisance'] > 0 and score <= .90 * v9['nuisance']
    second = row['small_recall'] >= v10['small_recall'] - .005 and (
        (row['recall'] >= v10['recall'] + .005 and score <= v10['nuisance']) or
        (score <= .90 * v10['nuisance'] and row['recall'] >= v10['recall'] - .0025))
    return first, second


def assess(path, validation, small, negatives, cpu):
    model = load_detector(path)
    batch, workers = (2, 0) if cpu else (16, 4)
    metrics, preds = evaluate(model, validation, batch, workers)
    sr, tp, boxes = small_recall(validation.records, preds, small)
    nu, scenes = nuisance(negatives, detect_records(model, negatives, batch=batch, workers=workers, floor=.2))
    scr = nu['synthetic_screens']
    return dict(AP50=metrics['coco']['AP50'], AP50_95=metrics['coco']['AP50_95'],
                precision=metrics['at_confidence_0.40']['precision'], recall=metrics['at_confidence_0.40']['recall'],
                small_recall=sr, small_tp=tp, small_boxes=boxes, nuisance=nu['score'], nuisance_counts=nu,
                synthetic_rate=scr['flag_040']/scr['images'] if scr['images'] else None, scenes=scenes)


def run(out, work, limit, cpu):
    run_dir = work / 'train_v10'
    initial = run_dir / 'initial.pth'
    def epoch(i):
        return run_dir / f'epoch_{i:02d}.pth'
    frozen = work / 'FROZEN_V10' / 'selected.pth'
    if sha256(epoch(6)) != FROZEN_HASH or sha256(frozen) != FROZEN_HASH:
        raise ValueError('epoch_06 / frozen checkpoint SHA256 mismatch')
    for path in [initial] + [epoch(i) for i in range(5, 11)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = dict(torch=str(torch.__version__), python=platform.python_version(), device=device,
               gpu=torch.cuda.get_device_name() if device == 'cuda' else None, seed=42)
    print(env, flush=True)
    manifests = work / 'manifests_v10'
    val = read_json(manifests / 'validation_full.json')[:limit]
    negatives = read_json(manifests / 'hn_val.json')[:limit]
    validation = Flames(val, 640)
    small = small_slice(val)
    if not limit and (len(small) != 564 or not any(r['category'] != 'synthetic_screen' for r in negatives)):
        raise ValueError('Expected the frozen small-flame and real-nuisance evaluation sets')
    files = [Path(__file__), PACKAGE/'refs'/'segments_exact.csv.gz']
    files += [Path(__file__).with_name(n+'.py') for n in ('common','metrics','data','model','select_v10')]
    context = dict(env=env, limit=limit, files=[sha256(p) for p in files],
                   eval_hash=hashlib.sha256(json.dumps([val, negatives], sort_keys=True).encode()).hexdigest())
    cache = out/'cache'/hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    folder = out/'candidates'
    folder.mkdir(parents=True, exist_ok=True)
    levels = measure(load_detector(epoch(6)), read_json(manifests/'train.json'), limit)
    selections = {f'C{i}': [[x['channel'] for x in level['ranked'][:k]] for level in levels]
                  for i,(k,_) in enumerate(DAMPING,1)}
    write_json(dict(levels=levels, selected=selections), out/'channels.json')

    def recipes():
        yield 'R0', 'V9 EMA', weights(initial)
        yield 'R1', 'Frozen V10 epoch 6', weights(epoch(6))
        for name, epochs in SOUPS.items():
            yield name, f'Mean of epochs {list(epochs)}', average([epoch(i) for i in epochs])
        for name, alpha in ALPHAS.items():
            yield name, f'Interpolate V9 to V10 at {alpha}', interpolate(initial, epoch(6), alpha)
        for i,(k, scale) in enumerate(DAMPING, 1):
            yield f'C{i}', f'Top {k} channels per scale times {scale}', damp(weights(epoch(6)), levels, k, scale)

    rows = []
    for name, note, state in recipes():
        path = folder/f'{name}.pth'
        tmp = path.with_suffix('.tmp')
        torch.save(dict(model=state, epoch=name, note=note), tmp)
        tmp.replace(path)
        digest = state_hash(state)
        cached = cache/f'{digest}.json'
        if not cached.exists():
            write_json(assess(path, validation, small, negatives, cpu), cached)
        row = read_json(cached)
        scenes = row.pop('scenes')
        if name == 'R0':
            baseline_scenes = scenes
        row.update(id=name, note=note, state_sha256=digest,
                   scenes_better=sum(v < baseline_scenes[g] for g,v in scenes.items()),
                   scenes_worse=sum(v > baseline_scenes[g] for g,v in scenes.items()))
        rows.append(row)
        print(name, row['recall'], row['nuisance'], flush=True)
        if len(rows) == 2 and not limit:   # audit fix: references must reproduce the frozen selection exactly
            frozen_rows = {r['checkpoint']: r for r in read_json(work / 'select' / 'selection.json')['candidates']}
            for reference, key in ((rows[0], 'initial.pth'), (rows[1], 'epoch_06.pth')):
                ref = frozen_rows[key]
                pairs = dict(AP50=ref['AP50'], recall=ref['recall'], small_recall=ref['small_recall'],
                             nuisance=ref['nuisance']['score'])
                off = {k: (reference[k], v) for k, v in pairs.items() if abs(reference[k] - v) > REFERENCE_TOLERANCE}
                if off:
                    raise RuntimeError(f'{reference["id"]} does not reproduce select/selection.json: {off}')
    for row in rows:
        row['pass1'], row['pass2'] = judge(row, rows[0], rows[1])
    passing = [r for r in rows[2:] if r['pass1'] and r['pass2']]
    winner = min(passing, key=lambda r:(r['nuisance'], -r['recall'])) if passing else rows[1]
    name = winner['id']
    result = dict(environment=env, limit=limit, frozen_sha256=FROZEN_HASH, candidates_tested=12, rows=rows,
                  selected=name, decision_scope='limited smoke test' if limit else 'full evaluation')
    if passing:
        target = out/f'export_{name}'
        result['export'] = export(load_detector(folder/f'{name}.pth'), validation, target)
    msg = f"{result['decision_scope']}: " + (f'Select {name}.' if passing else 'Retain frozen V10 epoch 6.')
    lines = ['# Weight operations', '', '| ID | AP50 | AP50:95 | P | R | Small R | Nuisance | Screens flagged | Scenes better / worse | Pass 1 | Pass 2 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        vals = ['n/a' if r[k] is None else format(r[k], '.3%') for k in
                ('AP50','AP50_95','precision','recall','small_recall','nuisance','synthetic_rate')]
        lines.append('| ' + ' | '.join([r['id'], *vals, f"{r['scenes_better']} / {r['scenes_worse']}",
                                         str(r['pass1']), str(r['pass2'])]) + ' |')
    lines += ['', msg, '', 'Testing 12 candidates gives a chance of a lucky pass.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(result, out/'results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    work = Path(os.environ['V10_WORK']).resolve()
    out = args.out.resolve()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if out == work or any(out.is_relative_to(work/n) for n in ('select','FROZEN_V10','manifests_v10','train_v10')):
        parser.error('Output must be separate from protected inputs')
    lab = Path('D:/Honeywell-V10-work').resolve()
    if work == lab and not out.is_relative_to(lab/'weightops'):
        parser.error('Lab outputs must be inside V10_WORK/weightops')
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(4)
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    run(out, work, args.limit, args.cpu or not torch.cuda.is_available())
