"""Build manifests_v10/: train (FASDD cleaned + additions with repeats), hn_val, and the unchanged FASDD val/test.

Split by WHOLE near-duplicate groups first (audit fix): a group holding a flame image (D-Fire box or own boxed
frame) or an own train-clip frame goes to train; a group holding own 'hnval' frames goes to hn_val and its other
members are quarantined; remaining groups go to hn_val by hash (~10%).
Repeats: real hard negatives x3, own clip frames (negative and boxed flame) x3, synthetic screens x1, rest x1.
Dropped: mining 'drop', whole-image CLIP check failures (negcheck.json), and synthetic screens whose Open Images
background was dropped during mining. Easy negatives are subsampled so flame images stay >= --min-positive.
The build stops if train and hn_val share any group or identical image, or hn_val shares an image with FASDD train.
"""
import argparse
import hashlib
import os
import random
import shutil
import zipfile
from pathlib import Path

from common import PACKAGE, make_record, read_json, work_dir, write_json

KEEP = ('image', 'width', 'height', 'boxes', 'source_image_id', 'sha256', 'source', 'category', 'group')


def own_flame_records(videos):
    """Boxed small-flame frames from TRAIN clips, shipped with the package (exact reviewed pixels)."""
    folder = PACKAGE / 'own_flame'
    if not (folder / 'own_boxes.json').exists() or os.environ.get('V10_TEST_SKIP_OWN_FLAME') == '1':
        return []
    split = {v['name']: v['split'] for v in videos if v['source'] == 'own'}
    category = {v['name']: v['category'] for v in videos if v['source'] == 'own'}
    records = []
    for e in read_json(folder / 'own_boxes.json')['frames']:
        if split.get(e['clip']) != 'train':
            raise RuntimeError(f'Boxed frame from a non-train clip: {e["clip"]} ({split.get(e["clip"])})')
        records.append(make_record(folder / e['image'], e['boxes'], source='own', category='own_flame:' + category[e['clip']],
                                   group='own:' + e['clip'], split='train'))
    return records


def slim(record, **extra):
    return {**{k: record[k] for k in KEEP if k in record}, **extra}


def fraction(group):
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hn-val-fraction', type=float, default=.10)
    p.add_argument('--hn-val-max', type=int, default=6000)
    p.add_argument('--min-positive', type=float, default=.30)
    p.add_argument('--hard-repeat', type=int, default=3)
    p.add_argument('--screen-repeat', type=int, default=1)
    p.add_argument('--own-repeat', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    work = work_dir()
    rng = random.Random(a.seed)
    fasdd = work / 'manifests_fasdd'
    decisions_file = read_json(work / 'mine' / 'decisions.json')
    decisions, exclusions = decisions_file['decisions'], set(decisions_file['fasdd_train_exclusions'])
    dedup = read_json(work / 'dedup' / 'dedup.json')
    leaked, groups = set(dedup['leaked']), dedup['groups']

    train_fasdd = read_json(fasdd / 'train.json')
    kept_fasdd = [r for r in train_fasdd if Path(r['image']).name not in exclusions]
    removed = sorted(Path(r['image']).name for r in train_fasdd if Path(r['image']).name in exclusions)

    sources = work / 'sources'
    new = []
    for name in ('dfire.json', 'openimages.json', 'roboflow.json', 'screens.json', 'own_frames.json'):
        if (sources / name).exists():
            new += read_json(sources / name)['records']
    own_flame = own_flame_records(read_json(sources / 'videos.json')['videos'])
    negcheck = set(read_json(work / 'mine' / 'negcheck.json')['drop']) if (work / 'mine' / 'negcheck.json').exists() else set()
    oi_dropped = {r['group'] for r in new if r['source'] == 'openimages' and decisions.get(r['image']) == 'drop'}
    group_of = lambda r: groups.get(r['image'], r['group'])

    # 1) decide every group's split before routing any image (audit fix)
    forced = {}
    for r in new + own_flame:
        g = group_of(r)
        if r['source'] == 'own' and r.get('split') == 'hnval':
            forced[g] = 'hnval'
        elif (r['boxes'] or (r['source'] == 'own' and r.get('split') == 'train')) and forced.get(g) != 'hnval':
            forced[g] = 'train'
    split_of = lambda g: forced.get(g) or ('hnval' if fraction(g) < a.hn_val_fraction else 'train')

    positives, own_pos, hard, screens_hard, own, easy, hn_val, dropped = [], [], [], [], [], [], [], {}
    count = lambda key: dropped.__setitem__(key, dropped.get(key, 0) + 1)
    for r in new + own_flame:
        g, decision = group_of(r), decisions.get(r['image'], 'negative')
        if r['image'] in leaked:
            count('leak'); continue
        if decision == 'drop':
            count('mining'); continue
        if r['image'] in negcheck:
            count('whole_image_check'); continue
        if r['source'] == 'screens' and r['group'] in oi_dropped:
            count('screen_background_dropped'); continue
        if r['source'] == 'own' and r.get('split') not in ('train', 'hnval'):
            continue                                               # eval clips never enter manifests
        target = split_of(g)
        if r['source'] == 'own' and r.get('split') == 'train' and target == 'hnval':
            count('quarantined_own_train'); continue
        if target == 'hnval':
            if r['boxes']:
                count('quarantined_positive'); continue
            hn_val.append(slim(r, group=g, hard=decision == 'hard_negative'))
        elif r['boxes']:
            (own_pos if r['source'] == 'own' else positives).append(slim(r, group=g))
        elif r['source'] == 'own':
            own.append(slim(r, group=g, hard=decision == 'hard_negative'))
        elif decision == 'hard_negative':
            (screens_hard if r['source'] == 'screens' else hard).append(slim(r, group=g, hard=True))
        else:
            easy.append(slim(r, group=g, hard=False))
    if len(hn_val) > a.hn_val_max:
        own_hn = [r for r in hn_val if r['source'] == 'own']
        rest = [r for r in hn_val if r['source'] != 'own']
        hard_first = sorted(rest, key=lambda r: (not r['hard'], fraction(r['image'])))
        hn_val = own_hn + hard_first[:max(0, a.hn_val_max - len(own_hn))]

    fixed = (kept_fasdd + positives + (own_pos + own) * a.own_repeat + hard * a.hard_repeat
             + screens_hard * a.screen_repeat)
    n_pos = sum(bool(r['boxes']) for r in fixed)
    budget = int(n_pos / a.min_positive) - len(fixed)       # how many easy negatives fit under the ratio
    rng.shuffle(easy)
    easy_used = easy[:max(0, budget)]
    train = fixed + easy_used
    rng.shuffle(train)
    if sum(bool(r['boxes']) for r in train) / len(train) < a.min_positive - .02:
        print(f'WARNING: flame fraction below {a.min_positive} even without easy negatives; '
              'report this at STOP 2 (repeats may need lowering)', flush=True)

    # 2) hard overlap checks (stop the build rather than train on a leaky split)
    train_sha = {r['sha256'] for r in train} | {r['sha256'] for r in train_fasdd}
    before = len(hn_val)                     # e.g. a public image byte-identical to a FASDD train image
    hn_val = [r for r in hn_val if r['sha256'] not in train_sha]
    if len(hn_val) < before:
        dropped['hn_val_identical_to_a_train_image'] = before - len(hn_val)
    shared = {r['group'] for r in train if 'group' in r} & {r['group'] for r in hn_val}
    if shared:
        raise RuntimeError(f'train and hn_val share {len(shared)} groups, e.g. {sorted(shared)[:5]}')
    if train_sha & {r['sha256'] for r in hn_val}:
        raise RuntimeError('train and hn_val share identical images')

    out = work / 'manifests_v10'
    out.mkdir(parents=True, exist_ok=True)
    write_json(train, out / 'train.json')
    write_json(hn_val, out / 'hn_val.json')
    for name in ('val', 'validation_full', 'test'):
        shutil.copyfile(fasdd / f'{name}.json', out / f'{name}.json')

    def by(records, key='category'):
        counts = {}
        for r in records:
            counts[r[key]] = counts.get(r[key], 0) + 1
        return dict(sorted(counts.items()))

    report = dict(
        train_samples=len(train), train_unique_images=len({r['image'] for r in train}),
        train_positive_fraction=round(sum(bool(r['boxes']) for r in train) / len(train), 4),
        fasdd_train_kept=len(kept_fasdd), fasdd_train_removed=removed,
        added=dict(dfire_flame_images=len(positives), own_flame_frames=len(own_pos), real_hard_negatives=len(hard),
                   hard_repeat=a.hard_repeat, synthetic_screen_hard=len(screens_hard), screen_repeat=a.screen_repeat,
                   own_frames=len(own), own_repeat=a.own_repeat, easy_negatives_available=len(easy),
                   easy_negatives_used=len(easy_used)),
        checks=dict(shared_groups_train_hn_val=0, shared_images_train_hn_val=0),
        hard_by_category=by(hard), own_by_category=by(own + own_pos), easy_used_by_source=by(easy_used, 'source'),
        hn_val_images=len(hn_val), hn_val_hard=sum(r['hard'] for r in hn_val), hn_val_by_category=by(hn_val),
        dropped=dropped, settings=vars(a))
    write_json(report, out / 'manifest_report.json')
    archive = work / 'outbox' / 'step3_manifest_report.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out / 'manifest_report.json', 'manifest_report.json')
        z.write(work / 'dedup' / 'dedup.json', 'dedup.json')
        z.write(work / 'mine' / 'decisions.json', 'decisions.json')
    print({k: v for k, v in report.items() if not isinstance(v, dict) or k in ('added', 'dropped')})
    print(f'\nSTOP 2: send {archive}. Continue with the next step only after the go-ahead.', flush=True)


if __name__ == '__main__':
    main()
