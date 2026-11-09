"""Steps 4a / 4b: owner recordings (far flames, new places) -> reviewed boxes -> manifests_v12.

Input: D:\\Honeywell-V10-work\\recordings_v12\\ (any sub-folders). File names (case-insensitive, .mp4/.mov):
   <place>_<candle|diya|lighter>_<N>m[_locked].mp4    flame clip, e.g. corridor_candle_10m.mp4
   <place>_noflame.mp4                               same view, no flame (negatives)
   place = any lowercase name (letters, digits, underscores). Names that do not match are listed and NOT used.
Split by PLACE (fixed in config_v12.json): test_places (corridor) -> distance TEST only; every other place -> TRAINING.

4a propose (frame every 0.5 s; fixed camera assumed):
   anchor = where the flame is: frozen V10 median box if V10 scores >= .25 in >= 20% of sampled frames, otherwise the
            bright warm pixel region that is present most often AND flickers most (flame), found over the whole clip;
   per frame box = bright warm core nearest the anchor (dense.py colour rule); a V10 box >= .10 overlapping it wins.
   No warm core near the anchor -> no proposal for that frame. Every proposal (train AND test places) is put on
   contact sheets: outbox/v12_step4a_recordings.zip.
4b finalize: inbox/rec_review.csv (id,decision ok|bad for EVERY id). ok frames of train places + noflame frames of
   train places go to manifests_v12/train.json (x5) on top of manifests_v11/train.json; eval manifests are copied
   byte-identical; corridor ok frames + its noflame frames -> rec_v12/distance_test.json (never trained on).
"""
import csv
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from common import contact_sheet, detect_frames, load_detector, read_json, write_json
from common12 import CONFIG, EVAL, INBOX, MANIFESTS11, MANIFESTS12, OUTBOX, V10_FROZEN, V10_FROZEN_SHA, WORK

SRC = WORK / 'recordings_v12'
OUT = WORK / 'rec_v12'
NAME = re.compile(r'^([a-z0-9]+(?:_[a-z0-9]+)*?)_(?:(candle|diya|lighter)_(\d+(?:\.\d+)?)m(_locked)?|(noflame))$', re.I)
REC = CONFIG['recordings']
_SEEDS_FILE = Path(__file__).resolve().parent.parent / 'recordings_seeds.json'
SEEDS = read_json(_SEEDS_FILE)['clips'] if _SEEDS_FILE.exists() else {}


def hot(frame):
    f = frame.astype(np.int16)
    return (f.max(2) >= 240) & (f.min(2) >= 170) & (f[:, :, 2] >= f[:, :, 0])   # near-white flame core (not orange glow)


def seed_box(frame, seed, t):
    """Box for one frame of a seeded clip: seeds interpolated in time, then snapped to the hot flame core(s)
    inside a window of 1x the seed size around it (cluster: union of all cores). Falls back to the seed box."""
    pts = sorted(seed['seeds'])
    if t <= pts[0][0]:
        s = pts[0][1:]
    elif t >= pts[-1][0]:
        s = pts[-1][1:]
    else:
        for a, b in zip(pts, pts[1:]):
            if a[0] <= t <= b[0]:
                u = (t - a[0]) / max(1e-6, b[0] - a[0])
                s = [a[i] + u * (b[i] - a[i]) for i in range(1, 5)]
                break
    x1, y1, x2, y2 = s
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    if seed.get('snap') is False:                                     # marked by eye, too faint to snap reliably
        return [float(v) for v in s]
    A, B = int(max(0, x1 - w)), int(max(0, y1 - h))
    C, D = int(min(W, x2 + w)), int(min(H, y2 + h))
    n, _, st, ce = cv2.connectedComponentsWithStats(hot(frame[B:D, A:C]).astype(np.uint8), 8)
    comps = [(st[i], ce[i]) for i in range(1, n) if st[i][4] >= 2]
    if not comps:
        return [float(v) for v in s]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if seed.get('cluster'):
        inside = [c for c in comps if x1 - w * .25 <= c[1][0] + A <= x2 + w * .25 and y1 - h * .5 <= c[1][1] + B <= y2 + h * .25]
        if not inside:
            return [float(v) for v in s]
        bx1 = min(c[0][0] for c in inside) + A; by1 = min(c[0][1] for c in inside) + B
        bx2 = max(c[0][0] + c[0][2] for c in inside) + A; by2 = max(c[0][1] + c[0][3] for c in inside) + B
        pad = max(3., .15 * max(bx2 - bx1, by2 - by1))
        return [float(max(0, bx1 - pad)), float(max(0, by1 - 2 * pad)), float(min(W, bx2 + pad)), float(min(H, by2 + pad))]
    st_k, _ = min(comps, key=lambda c: np.hypot(c[0][0] + c[0][2] / 2 + A - cx, c[0][1] + c[0][3] / 2 + B - cy))
    mx, my = st_k[0] + st_k[2] / 2 + A, st_k[1] + st_k[3] / 2 + B   # centre of the core's bounding box
    if abs(mx - cx) > .75 * w or abs(my - cy) > .75 * h:              # core too far from the seed: keep the seed
        return [float(v) for v in s]
    # keep the marked flame size, centred on the core
    return [float(max(0, mx - w / 2)), float(max(0, my - h / 2)), float(min(W, mx + w / 2)), float(min(H, my + h / 2))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou1(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def warm(frame):
    f = frame.astype(np.int16)
    return (f.max(2) >= 200) & (f[:, :, 2] >= f[:, :, 0])


def colour_box(frame, anchor):
    """Same rule as V11 dense.py: bright warm core nearest the anchor centre, in a 4x anchor window."""
    x1, y1, x2, y2 = anchor
    w, h = max(4, x2 - x1), max(4, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = frame.shape[:2]
    a, b = int(max(0, cx - 2 * w)), int(max(0, cy - 2 * h))
    c, d = int(min(W, cx + 2 * w)), int(min(H, cy + 2 * h))
    mask = warm(frame[b:d, a:c]).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmin([np.hypot(cents[i][0] + a - cx, cents[i][1] + b - cy) for i in range(1, n)]))
    bx, by, bw, bh = stats[k][:4]
    bw2, bh2 = max(bw, .6 * w), max(bh, .6 * h)
    mx, my = bx + a + bw / 2, by + b + bh / 2
    return [float(max(0., mx - bw2 / 2)), float(max(0., my - bh2 / 2)), float(min(W, mx + bw2 / 2)), float(min(H, my + bh2 / 2))]


def frames(path, every_s):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step, index = max(1, round(fps * every_s)), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            yield index, frame
        index += 1


def clips():
    found, ignored = [], []
    for p in sorted(SRC.rglob('*')):
        if p.suffix.lower() not in ('.mp4', '.mov') or not p.is_file():
            continue
        m = NAME.match(p.stem)
        if not m:
            ignored.append(str(p))
            continue
        place = m.group(1).lower()
        split = 'test' if place in REC['test_places'] else 'train'
        found.append(dict(path=str(p), clip=p.stem.lower(), place=place, split=split,
                          flame=(m.group(2) or 'none').lower(), distance_m=float(m.group(3)) if m.group(3) else None,
                          locked=bool(m.group(4))))
    return found, ignored


def anchor_for(model, path):
    sampled = list(frames(path, REC['sample_every_s']))
    if not sampled:
        return None, 'unreadable', sampled, []
    dets = []
    for s in range(0, len(sampled), 16):
        dets += detect_frames(model, [f for _, f in sampled[s:s + 16]], floor=.10)
    strong = [[d for d in ds if d[4] >= .25] for ds in dets]
    hits = [max(d, key=lambda x: x[4])[:4] for d in strong if d]
    if len(hits) >= .2 * len(sampled):
        return [float(v) for v in np.median(np.array(hits), 0)], 'v10_median', sampled, dets
    H, W = sampled[0][1].shape[:2]
    count = np.zeros((H, W), np.float32)
    s1 = np.zeros((H, W), np.float32)
    s2 = np.zeros((H, W), np.float32)
    for _, f in sampled:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
        count += warm(f)
        s1 += g
        s2 += g * g
    n = len(sampled)
    freq = count / n
    flicker = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))
    cand = (freq >= .3).astype(np.uint8)
    k, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if k <= 1:
        return None, 'no_warm_region', sampled, dets
    score = freq * (flicker + 1e-3)
    sums = np.bincount(labels.ravel(), weights=score.ravel(), minlength=k)
    best = 1 + int(np.argmax(sums[1:]))
    x, y, w, h = stats[best][:4]
    w2, h2 = max(w, 6), max(h, 8)
    return [x + w / 2 - w2 / 2, y + h / 2 - h2 / 2, x + w / 2 + w2 / 2, y + h / 2 + h2 / 2], 'warm_flicker', sampled, dets


def propose():
    if not SRC.exists():
        raise RuntimeError(f'{SRC} not found: copy the owner recordings there first')
    found, ignored = clips()
    if not any(c['flame'] != 'none' for c in found):
        raise RuntimeError(f'no correctly named flame clips under {SRC}; ignored names: {ignored[:10]}')
    model = (load_detector(V10_FROZEN, V10_FROZEN_SHA)            # only needed for clips without marked seeds
             if any(c['flame'] != 'none' and c['clip'] not in SEEDS for c in found) else None)
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    rows, negatives, per_clip = [], [], {}
    for c in found:
        if c['flame'] == 'none':
            n = 0
            for index, frame in frames(c['path'], REC['negative_every_s']):
                name = f'{c["clip"]}_f{index:05d}.png'
                cv2.imwrite(str(OUT / 'frames' / name), frame)
                negatives.append(dict(image=str(OUT / 'frames' / name), width=frame.shape[1], height=frame.shape[0], boxes=[],
                                      source_image_id=Path(name).stem, location='rec:' + c['place'], group='rec:' + c['clip'],
                                      split=c['split'], category='own_v12_negative', place=c['place'], clip=c['clip']))
                n += 1
            per_clip[c['clip']] = dict(kind='noflame', frames=n, split=c['split'])
            continue
        made = 0
        seed = SEEDS.get(c['clip'])
        if seed:                                   # owner clips marked by eye: follow the seeds, refine on the hot core
            how = 'seeds_cluster' if seed.get('cluster') else 'seeds'
            sampled = list(frames(c['path'], REC['sample_every_s']))
            total = max(1, int(cv2.VideoCapture(c['path']).get(cv2.CAP_PROP_FRAME_COUNT)))
            for index, frame in sampled:
                box = seed_box(frame, seed, index / total)
                name = f'{c["clip"]}_f{index:05d}.png'
                cv2.imwrite(str(OUT / 'frames' / name), frame)        # lossless
                rows.append(dict(id=len(rows) + 1, clip=c['clip'], place=c['place'], split=c['split'], flame=c['flame'],
                                 distance_m=c['distance_m'], locked=c['locked'], frame=index, image=str(OUT / 'frames' / name),
                                 width=frame.shape[1], height=frame.shape[0], box=[round(v, 1) for v in box], kind=how,
                                 cluster=bool(seed.get('cluster'))))
                made += 1
            per_clip[c['clip']] = dict(kind=c['flame'], distance_m=c['distance_m'], split=c['split'], anchor=how,
                                       sampled=len(sampled), proposals=made)
            print(c['clip'], per_clip[c['clip']], flush=True)
            continue
        anchor, how, sampled, all_dets = anchor_for(model, c['path'])
        if anchor:
            for (index, frame), frame_dets in zip(sampled, all_dets):
                box = colour_box(frame, anchor)
                if not box:
                    continue
                dets = [d for d in frame_dets if iou1(d[:4], box) >= .1]
                kind = 'colour'
                if dets:
                    box, kind = [float(v) for v in max(dets, key=lambda d: d[4])[:4]], 'v10'
                name = f'{c["clip"]}_f{index:05d}.png'
                cv2.imwrite(str(OUT / 'frames' / name), frame)        # lossless
                rows.append(dict(id=len(rows) + 1, clip=c['clip'], place=c['place'], split=c['split'], flame=c['flame'],
                                 distance_m=c['distance_m'], locked=c['locked'], frame=index, image=str(OUT / 'frames' / name),
                                 width=frame.shape[1], height=frame.shape[0], box=[round(v, 1) for v in box], kind=kind))
                made += 1
        per_clip[c['clip']] = dict(kind=c['flame'], distance_m=c['distance_m'], split=c['split'], anchor=how,
                                   sampled=len(sampled), proposals=made)
        print(c['clip'], per_clip[c['clip']], flush=True)
    write_json(rows, OUT / 'proposals.json')
    write_json(negatives, OUT / 'negatives.json')
    write_json(dict(clips=found, ignored_names=ignored, per_clip=per_clip, proposals=len(rows), negative_frames=len(negatives)),
               OUT / 'propose_summary.json')
    sheets = []
    for s in range(0, len(rows), 36):
        path = OUT / f'review_{s // 36 + 1:02d}.jpg'
        contact_sheet([(r['image'], r['box'], f'id {r["id"]} {r["clip"][:22]}') for r in rows[s:s + 36]], path)
        sheets.append(path)
    OUTBOX.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTBOX / 'v12_step4a_recordings.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sheets + [OUT / 'proposals.json', OUT / 'propose_summary.json']:
            z.write(p, p.name)
    if ignored:
        print('WARNING: badly named files NOT used:', ignored, flush=True)
    print(f'{len(rows)} proposals on {len(sheets)} sheets, {len(negatives)} negative frames. Review EVERY box -> inbox/rec_review.csv')


def finalize():
    rows = read_json(OUT / 'proposals.json')
    path = INBOX / 'rec_review.csv'
    if not path.exists():
        raise RuntimeError('inbox/rec_review.csv missing (id,decision ok|bad for every proposal)')
    decisions = {int(r['id']): r['decision'].strip().lower() for r in csv.DictReader(path.open())}
    missing = [r['id'] for r in rows if r['id'] not in decisions]
    invalid = set(decisions.values()) - {'ok', 'bad'}
    if missing or invalid:
        raise RuntimeError(f'review incomplete: missing ids {missing[:10]}, invalid values {invalid}')
    ok = [r for r in rows if decisions[r['id']] == 'ok']
    negatives = read_json(OUT / 'negatives.json')
    record = lambda r: dict(image=r['image'], width=r['width'], height=r['height'], boxes=[r['box']],
                            source_image_id=Path(r['image']).stem, location='rec:' + r['place'], group='rec:' + r['clip'],
                            split=r['split'], category='own_v12_flame', source='own_v12')
    train_extra = [record(r) for r in ok if r['split'] == 'train'] + [dict(n, source='own_v12') for n in negatives if n['split'] == 'train']
    test = [dict(record(r), clip=r['clip'], flame=r['flame'], distance_m=r['distance_m'], place=r['place'])
            for r in ok if r['split'] == 'test']
    test += [dict(n, flame='none', distance_m=None) for n in negatives if n['split'] == 'test']
    MANIFESTS12.mkdir(parents=True, exist_ok=True)
    for name in EVAL:
        if (MANIFESTS11 / name).exists():
            shutil.copyfile(MANIFESTS11 / name, MANIFESTS12 / name)
            assert sha(MANIFESTS11 / name) == sha(MANIFESTS12 / name)
    if (MANIFESTS11 / 'flame_bank.npz').exists():
        shutil.copyfile(MANIFESTS11 / 'flame_bank.npz', MANIFESTS12 / 'flame_bank.npz')
    train = read_json(MANIFESTS11 / 'train.json') + train_extra * REC['repeat']
    banned = {r['image'] for r in test}
    for name in EVAL:
        if name != 'val.json' and (MANIFESTS12 / name).exists():
            banned |= {r['image'] for r in read_json(MANIFESTS12 / name)}
    leak = banned & {r['image'] for r in train}
    if leak:
        raise RuntimeError(f'{len(leak)} test/evaluation images in V12 training, e.g. {sorted(leak)[:3]}')
    if any(r['location'].endswith(tuple(REC['test_places'])) for r in train_extra):
        raise RuntimeError('a test-place recording reached training')
    write_json(train, MANIFESTS12 / 'train.json')
    write_json(test, OUT / 'distance_test.json')
    summary = dict(proposals=len(rows), ok=len(ok), bad=len(rows) - len(ok),
                   train_flame_frames=sum(r['split'] == 'train' for r in ok),
                   train_negative_frames=sum(n['split'] == 'train' for n in negatives), repeat=REC['repeat'],
                   test_flame_frames=sum(r['split'] == 'test' for r in ok),
                   test_negative_frames=sum(n['split'] == 'test' for n in negatives),
                   train_records=len(train), train_sha256=sha(MANIFESTS12 / 'train.json'),
                   ok_per_clip={c: sum(r['clip'] == c for r in ok) for c in sorted({r['clip'] for r in rows})})
    write_json(summary, OUT / 'finalize_summary.json')
    print(summary, flush=True)


if __name__ == '__main__':
    {'propose': propose, 'finalize': finalize}[sys.argv[1]]()
