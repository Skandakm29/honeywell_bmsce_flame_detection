"""Steps 1b / 1c: box many more frames of the 4 TRAIN small-flame clips (adds volume, not new scenes).

Clips (train split, by location; never eval/hn_val): ut_diya_01, ut_farflame_01, ut_farflame_02, ut_flame_occlusion.
1b propose: one frame every 0.33 s. Proposal per frame, fixed before results:
   anchor = the reviewed box (own_flame/own_boxes.json) of the nearest reviewed frame in time;
   colour box = the bright warm core (max channel >= 200, R >= B) component nearest the anchor centre, inside a
                window of 4x the anchor size, padded to the anchor's width/height ratio (at least anchor size x .6);
   V10 box    = a frozen V10 detection >= .10 that overlaps the colour box (IoU >= .1) is used instead, if any.
   No warm component in the window -> no proposal (the frame is not used). Reviewed V10 frames are skipped.
   Contact sheets of every proposal -> outbox/v11_step1b_dense.zip.
WhatsApp clips (package/whatsapp, owner decision 6 Nov: ALL used for training, no longer a test set):
   flame clips use the same proposal, anchored on frozen-V10 seed boxes (>= .30) and then on the previous
   proposal in time (tracking, so the walk-away candle is followed as it shrinks); every box is reviewed.
   Non-fire clips (screen showing fire, lantern + orange objects, lights switching): 1 frame/s as negatives.
1c finalize: needs inbox/dense_review.csv (columns id,decision with decision ok|bad, EVERY id). Only 'ok' frames go
   to dense_v11/extra.json (lossless PNG, split=train) and are used x3 in V11 training by step 2.
"""
import csv
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from common import PACKAGE, contact_sheet, detect_frames, load_detector, read_json, write_json
from paths11 import INBOX, OUTBOX, V10_FROZEN, V10_FROZEN_SHA, WORK, iou

CLIPS = ['ut_diya_01.mp4', 'ut_farflame_01.mp4', 'ut_farflame_02.mp4', 'ut_flame_occlusion.mp4']
STEP_S = 1 / 3
OUT = WORK / 'dense_v11'


def find_clip(name):
    for root in (WORK / 'clips', WORK / 'data' / 'clips', WORK / 'sources'):
        hits = list(root.rglob(name)) if root.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f'{name} not found under {WORK}\\clips (set up as in V10 step 1)')


def colour_box(frame, anchor):
    x1, y1, x2, y2 = anchor
    w, h = max(4, x2 - x1), max(4, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = frame.shape[:2]
    a, b = int(max(0, cx - 2 * w)), int(max(0, cy - 2 * h))
    c, d = int(min(W, cx + 2 * w)), int(min(H, cy + 2 * h))
    win = frame[b:d, a:c].astype(np.int16)
    mask = ((win.max(2) >= 200) & (win[:, :, 2] >= win[:, :, 0])).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmin([np.hypot(cents[i][0] + a - cx, cents[i][1] + b - cy) for i in range(1, n)]))
    bx, by, bw, bh = stats[k][:4]
    bw2, bh2 = max(bw, .6 * w), max(bh, .6 * h)
    mx, my = bx + a + bw / 2, by + b + bh / 2
    return [max(0., mx - bw2 / 2), max(0., my - bh2 / 2), min(float(W), mx + bw2 / 2), min(float(H), my + bh2 / 2)]


WA = Path(__file__).resolve().parent.parent / 'whatsapp'


def jobs():
    own = read_json(PACKAGE / 'own_flame' / 'own_boxes.json')['frames']
    for clip in CLIPS:
        yield clip, find_clip(clip), sorted((e['frame'], e['boxes'][0]) for e in own if e['clip'] == clip), False
    for clip, e in read_json(WA / 'whatsapp.json')['clips'].items():
        if e['kind'] == 'flame':
            yield clip, WA / clip, [tuple(x) for x in e['seeds']], True


def negatives():
    rows = []
    for clip, e in read_json(WA / 'whatsapp.json')['clips'].items():
        if e['kind'] != 'none':
            continue
        cap = cv2.VideoCapture(str(WA / clip))
        step, index = max(1, round(cap.get(cv2.CAP_PROP_FPS) or 30)), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                name = f'{Path(clip).stem}_f{index:05d}.png'
                cv2.imwrite(str(OUT / 'frames' / name), frame)
                rows.append(dict(image=str(OUT / 'frames' / name), width=frame.shape[1], height=frame.shape[0], boxes=[],
                                 source_image_id=Path(name).stem, location='whatsapp_room', group='wa:' + clip,
                                 split='train', category='own_v11_negative'))
            index += 1
    return rows


def propose():
    model = load_detector(V10_FROZEN, V10_FROZEN_SHA)
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    rows = []
    for clip, path, anchors, track in jobs():
        if not anchors:
            print('no anchor for', clip, '- skipped'); continue
        reviewed = set() if track else {f for f, _ in anchors}
        last = None
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(fps * STEP_S))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0 and index not in reviewed:
                nearest = min(anchors, key=lambda fa: abs(fa[0] - index))
                anchor = (last[1] if track and last and index - last[0] <= 2 * fps and abs(nearest[0] - index) > fps
                          else nearest[1])
                box = colour_box(frame, anchor)
                if box:
                    dets = [d for d in detect_frames(model, [frame], floor=.10)[0] if iou([d[:4]], [box])[0, 0] >= .1]
                    kind = 'colour'
                    if dets:
                        box, kind = [float(v) for v in max(dets, key=lambda d: d[4])[:4]], 'v10'
                    last = (index, box)
                    name = f'{Path(clip).stem}_f{index:05d}.png'
                    cv2.imwrite(str(OUT / 'frames' / name), frame)          # lossless
                    rows.append(dict(id=len(rows) + 1, clip=clip, frame=index, image=str(OUT / 'frames' / name),
                                     width=frame.shape[1], height=frame.shape[0], box=[round(v, 1) for v in box], kind=kind))
            index += 1
        print(clip, 'proposals so far', len(rows), flush=True)
    write_json(rows, OUT / 'proposals.json')
    write_json(negatives(), OUT / 'negatives.json')
    sheets = []
    for s in range(0, len(rows), 36):
        path = OUT / f'review_{s // 36 + 1:02d}.jpg'
        contact_sheet([(r['image'], r['box'], f'id {r["id"]} {r["kind"]}') for r in rows[s:s + 36]], path)
        sheets.append(path)
    OUTBOX.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTBOX / 'v11_step1b_dense.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sheets + [OUT / 'proposals.json']:
            z.write(p, p.name)
    print(f'{len(rows)} proposals on {len(sheets)} sheets in {OUT}. Review EVERY box -> inbox/dense_review.csv')


def finalize():
    rows = read_json(OUT / 'proposals.json')
    path = INBOX / 'dense_review.csv'
    if not path.exists():
        raise RuntimeError('inbox/dense_review.csv missing (id,decision ok|bad for every proposal)')
    decisions = {int(r['id']): r['decision'].strip().lower() for r in csv.DictReader(path.open())}
    missing = [r['id'] for r in rows if r['id'] not in decisions]
    bad = {v for v in decisions.values()} - {'ok', 'bad'}
    if missing or bad:
        raise RuntimeError(f'review incomplete: missing ids {missing[:10]}..., invalid values {bad}')
    keep = [dict(image=r['image'], width=r['width'], height=r['height'], boxes=[r['box']], source_image_id=Path(r['image']).stem,
                 location=r['clip'], group='own:' + r['clip'], split='train') for r in rows if decisions[r['id']] == 'ok']
    negs = read_json(OUT / 'negatives.json')
    keep += negs
    write_json(keep, OUT / 'extra.json')
    ok = len(keep) - len(negs)
    write_json(dict(proposals=len(rows), ok=ok, bad=len(rows) - ok, whatsapp_negative_frames=len(negs),
                    per_clip={c: sum(k['location'] == c for k in keep) for c in {r['clip'] for r in rows}}), OUT / 'summary.json')
    print('dense frames accepted', ok, 'of', len(rows), '+', len(negs), 'WhatsApp negative frames')


if __name__ == '__main__':
    {'propose': propose, 'finalize': finalize}[sys.argv[1]]()
