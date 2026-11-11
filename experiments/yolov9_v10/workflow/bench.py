"""Benchmarks for one detector checkpoint (run identically for V9 and V10).

E1 FASDD full validation: COCO AP, P/R@.40, recall per flame size, false-positive image rate per negative group
E2 hn_val: false-positive image rate @.20/.40 per category (selection set)
E3 BoWFire (if ingested): image-level hit rate on fire / false-positive rate on non-fire
Videos (own clips, MIVIA): sampled at 10 fps, detections >= .25 with in-box statistics -> videos/<name>.jsonl
FASDD test only with --test (step 7, after approval).
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from common import PACKAGE, detect_frames, detect_records, load_detector, read_json, work_dir, write_json
from temporal import FPS


def subset_metrics(records, predictions, keep):
    from metrics import operating_metrics
    index = {i: k for k, i in enumerate(i for i, r in enumerate(records) if keep(r))}
    chosen = [records[i] for i in index]
    preds = [dict(p, image_id=index[p['image_id']]) for p in predictions if p['image_id'] in index]
    result = operating_metrics(chosen, preds)
    flagged = len({p['image_id'] for p in preds if p['score'] >= .4})
    result.update(images=len(chosen), images_with_detection_040=flagged,
                  image_detection_rate_040=flagged / max(1, len(chosen)))
    return result


def fasdd(model, name, out):
    from data import Flames
    from metrics import evaluate
    records = read_json(work_dir() / 'manifests_v10' / f'{name}.json')
    metrics, predictions = evaluate(model, Flames(records, 640), 16, 4)
    split = 'val' if name == 'validation_full' else 'test'
    segments = {}
    with gzip.open(PACKAGE / 'refs' / 'segments_exact.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            if row['split'] == split:
                segments[Path(row['name']).name] = row
    seg = lambda r: segments.get(Path(r['image']).name, {})
    metrics['segments'] = {f'flame_size_{size}': subset_metrics(records, predictions, lambda r, s=size: seg(r).get('flame_size') == s)
                           for size in ('small', 'medium', 'large')}
    for group in ('smoke', 'none'):
        metrics['segments'][f'negatives_{group}'] = subset_metrics(records, predictions, lambda r, g=group: seg(r).get('group') == g)
    write_json(metrics, out / f'{name}_metrics.json')
    write_json(predictions, out / f'{name}_predictions.json')
    return metrics


def negatives(model, records, out, name):
    detections = detect_records(model, records, floor=.2)
    rows = {}
    for r, d in zip(records, detections):
        row = rows.setdefault(r['category'], dict(images=0, fp_020=0, fp_040=0))
        row['images'] += 1
        row['fp_020'] += any(x[4] >= .2 for x in d)
        row['fp_040'] += any(x[4] >= .4 for x in d)
    total = dict(images=len(records), fp_020=sum(r['fp_020'] for r in rows.values()), fp_040=sum(r['fp_040'] for r in rows.values()))
    for row in list(rows.values()) + [total]:
        row['fp_rate_020'], row['fp_rate_040'] = row['fp_020'] / max(1, row['images']), row['fp_040'] / max(1, row['images'])
    write_json({r['image']: d for r, d in zip(records, detections)}, out / f'{name}_detections.json')
    return dict(total=total, by_category=dict(sorted(rows.items())))


def bowfire(model, out):
    path = work_dir() / 'sources' / 'bowfire.json'
    if not path.exists():
        return None
    records = read_json(path)['records']
    detections = detect_records(model, records, floor=.2)
    fire = [any(x[4] >= .4 for x in d) for r, d in zip(records, detections) if r['fire']]
    other = [any(x[4] >= .4 for x in d) for r, d in zip(records, detections) if not r['fire']]
    write_json({r['image']: d for r, d in zip(records, detections)}, out / 'bowfire_detections.json')
    return dict(fire_images=len(fire), fire_hit_rate_040=float(np.mean(fire)) if fire else None,
                nonfire_images=len(other), nonfire_fp_rate_040=float(np.mean(other)) if other else None)


def box_stats(frame, hsv, gray, previous, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, min(frame.shape[1], x2)), max(y1 + 1, min(frame.shape[0], y2))
    region = hsv[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    coloured = region[region[:, 1] > 60, 0]
    spread = float(np.std(coloured) / 180) if len(coloured) > 8 else 0.
    diff = float(np.abs(gray[y1:y2, x1:x2].astype(np.float32) - previous[y1:y2, x1:x2]).mean() / 255) if previous is not None else 0.
    return [float(region[:, 2].mean() / 255), float(region[:, 1].mean() / 255), spread, diff]


def video(model, entry, out, batch=16):
    capture = cv2.VideoCapture(entry['path'])
    fps = entry['fps']
    lines, pending, previous, index, next_t = [], [], None, 0, 0.
    frame_rows = []

    def flush():
        nonlocal previous
        detections = detect_frames(model, [f for _, f in pending], floor=.25)
        for (t, frame), dets in zip(pending, detections):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            rows = [d[:5] + box_stats(frame, hsv, gray, previous, d) for d in dets]
            lines.append(json.dumps(dict(t=round(t, 3), dets=[[round(v, 4) for v in r] for r in rows])))
            frame_rows.append((t, any(d[4] >= .4 for d in dets)))
            previous = gray
        pending.clear()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t = index / fps
        if t + 1e-6 >= next_t:
            pending.append((t, frame))
            next_t += 1 / FPS
            if len(pending) == batch:
                flush()
        index += 1
    if pending:
        flush()
    capture.release()
    (out / 'videos').mkdir(parents=True, exist_ok=True)
    (out / 'videos' / f'{entry["name"]}.jsonl').write_text('\n'.join(lines))
    inside = [hit for t, hit in frame_rows if any(a <= t <= b for a, b in entry['fire_intervals'])]
    outside = [hit for t, hit in frame_rows if not any(a <= t <= b for a, b in entry['fire_intervals'])]
    return dict(frames=len(frame_rows), fire_frame_hit_rate=float(np.mean(inside)) if inside else None,
                nonfire_frame_fp_rate=float(np.mean(outside)) if outside else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='v9 or v10')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test', action='store_true', help='FASDD test once (step 7 only)')
    p.add_argument('--skip-fasdd', action='store_true', help='plumbing tests only')
    a = p.parse_args()
    work = work_dir()
    out = work / 'bench' / a.name
    out.mkdir(parents=True, exist_ok=True)
    model = load_detector(a.checkpoint)
    if a.test:
        write_json(fasdd(model, 'test', out), out / 'test_summary.json')
        return
    summary = dict(checkpoint=a.checkpoint)
    if not a.skip_fasdd:
        m = fasdd(model, 'validation_full', out)
        summary['E1_fasdd_val'] = dict(coco=m['coco'], at_040=m['at_confidence_0.40'], segments=m['segments'])
    summary['E2_hn_val'] = negatives(model, read_json(work / 'manifests_v10' / 'hn_val.json'), out, 'hn_val')
    summary['E3_bowfire'] = bowfire(model, out)
    videos = read_json(work / 'sources' / 'videos.json')['videos'] if (work / 'sources' / 'videos.json').exists() else []
    summary['videos_frame_level'] = {}
    for entry in videos:
        summary['videos_frame_level'][entry['name']] = dict(split=entry['split'], source=entry['source'],
                                                            category=entry['category'], **video(model, entry, out))
        print('video', entry['name'], summary['videos_frame_level'][entry['name']], flush=True)
    write_json(summary, out / 'bench.json')


if __name__ == '__main__':
    main()
