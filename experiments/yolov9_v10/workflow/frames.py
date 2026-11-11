"""Own recorded clips (+ optional MIVIA / ONFIRE test videos): split by clip, extract negative training frames.

clips.csv columns: clip,category,location,daynight,fire_start_s,fire_end_s[,split]
  fire_start_s/fire_end_s empty = no flame anywhere in the clip. split empty = assigned here
  (per category: ~30% 'eval', one 'hnval' clip when a category has >=3 remaining, rest 'train').
Fire clips are never used for detector training (no boxes); they feed the temporal layer and event evaluation.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

from common import make_record, work_dir, write_json

VIDEO = ('.mp4', '.mov', '.avi', '.mkv', '.m4v', '.3gp')


def _rank(name):
    return hashlib.sha256(name.encode()).hexdigest()


FLAME = {'candle', 'match', 'binfire', 'farflame', 'mixed', 'diya', 'occluded', 'flame'}
NONFIRE = {'screen', 'empty', 'sunlight', 'heater', 'torch', 'lamp', 'led', 'vehicle', 'vest', 'reflection',
           'sunset', 'orange', 'sparks', 'other'}


def table_from_names(folder):
    """No annotation needed: <initials>_<category>_<nn>.mp4. Flame clips must show the flame for the whole clip."""
    rows, bad = [], []
    for path in sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in VIDEO):
        parts = path.stem.lower().split('_')
        category = parts[1] if len(parts) >= 3 else ''
        if category not in FLAME | NONFIRE:
            bad.append(path.name)
            continue
        fire = category in FLAME
        rows.append(dict(clip=path.name, category=('mixed_nuisance' if category == 'mixed' else category),
                         location=parts[0], daynight='', fire_start_s='0' if fire else '', fire_end_s='', split=''))
    if bad:
        raise RuntimeError(f'File names not in <initials>_<category>_<nn> form with a known category: {bad}')
    table = Path(folder) / 'clips.csv'
    with open(table, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'clips.csv written from {len(rows)} file names')
    return table


def read_table(folder, table, source):
    folder = Path(folder)
    if not Path(table).exists():
        table = table_from_names(folder)
    rows = list(csv.DictReader(open(table, newline='', encoding='utf-8-sig')))
    videos, problems = [], []
    for row in rows:
        name = row['clip'].strip()
        if not name:
            continue
        candidates = [folder / name] + [folder / (name + e) for e in VIDEO]
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            problems.append(f'missing video file for clip "{name}"')
            continue
        capture = cv2.VideoCapture(str(path))
        fps, count = capture.get(cv2.CAP_PROP_FPS) or 0, capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()
        if fps <= 0:
            problems.append(f'cannot read {path}')
            continue
        start, end = (row.get('fire_start_s') or '').strip(), (row.get('fire_end_s') or '').strip()
        if not start and (row.get('fire_start_frame') or '').strip():      # ONFIRE-style frame annotation
            start = str(float(row['fire_start_frame']) / fps)
        duration = count / fps
        intervals = [[float(start), float(end) if end else duration]] if start else []
        videos.append(dict(name=name.replace('/', '_').replace('\\', '_'), path=str(path.resolve()), source=source, category=row.get('category', '').strip() or source,
                           location=row.get('location', '').strip(), daynight=row.get('daynight', '').strip(),
                           fire=bool(intervals), fire_intervals=intervals, fps=fps, duration_s=duration,
                           split=(row.get('split') or '').strip().lower()))
    if problems:
        raise RuntimeError('clips table problems:\n' + '\n'.join(problems))
    return videos


def assign_splits(videos):
    by_category = {}
    for v in videos:
        if not v['split']:
            by_category.setdefault(v['category'], []).append(v)
    for group in by_category.values():
        group.sort(key=lambda v: _rank(v['name']))
        n_eval = max(1, round(.3 * len(group))) if len(group) >= 2 else 0
        for i, v in enumerate(group):
            v['split'] = 'eval' if i < n_eval else 'train'
        rest = [v for v in group if v['split'] == 'train']
        if len(rest) >= 3 and not rest[0]['fire']:
            rest[0]['split'] = 'hnval'
    for v in videos:
        if v['split'] not in ('train', 'eval', 'hnval'):
            raise ValueError(f'bad split {v["split"]} for {v["name"]}')


def extract(video, every_s=1.0, still=3.0):
    """Frames every `every_s` seconds, skipping near-identical consecutive frames (mean abs diff < still)."""
    out = work_dir() / 'data' / 'own_frames' / video['name']
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video['path'])
    step = max(1, round(video['fps'] * every_s))
    records, previous, index = [], None, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
            if previous is None or np.abs(small - previous).mean() >= still:
                path = out / f'f{index:06d}.jpg'
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                records.append(make_record(path, [], source='own', category='own:' + video['category'],
                                           group='own:' + video['name'], split=video['split'], time_s=index / video['fps']))
                previous = small
        index += 1
    capture.release()
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clips', help='folder with recorded clips and clips.csv')
    p.add_argument('--mivia', help='optional folder with MIVIA videos and mivia.csv (same columns; all evaluation)')
    p.add_argument('--onfire', help='optional folder with ONFIRE videos and onfire.csv '
                                     '(clip + fire_start_s or fire_start_frame; empty = no fire; all evaluation)')
    a = p.parse_args()
    work = work_dir()
    videos = []
    if a.clips:
        own = read_table(a.clips, Path(a.clips) / 'clips.csv', 'own')
        assign_splits(own)
        videos += own
    for source, folder in (('mivia', a.mivia), ('onfire', a.onfire)):
        if folder:
            table = Path(folder) / f'{source}.csv'
            if not table.exists():
                raise RuntimeError(f'{table} missing: one row per video, columns clip,fire_start_s (or fire_start_frame)')
            extra = read_table(folder, table, source)
            for v in extra:
                v['split'] = 'eval'                                 # test only, never trained or tuned on
            videos += extra
    records = []
    for v in videos:
        if v['source'] == 'own' and not v['fire'] and v['split'] in ('train', 'hnval'):
            records += extract(v)
            print(f'{v["name"]}: {sum(r["group"] == "own:" + v["name"] for r in records)} frames', flush=True)
    summary = {}
    for v in videos:
        key = f'{v["source"]}/{v["split"]}/{"fire" if v["fire"] else "nonfire"}'
        summary[key] = summary.get(key, 0) + 1
    print(summary)
    write_json(dict(videos=videos, summary=summary), work / 'sources' / 'videos.json')
    write_json(dict(records=records), work / 'sources' / 'own_frames.json')


if __name__ == '__main__':
    main()
