"""Leakage removal and grouping for every new image.

Drop a new image when it is a near-duplicate of any evaluation image (FASDD val/test, BoWFire, MIVIA/own eval
video frames): CLIP cosine >= .95 or 64-bit pHash Hamming distance <= 6.
Group new images: union of their base group (clip / source image) and CLIP near-duplicate links (cosine >= .92),
so hn_val never shares a scene with training.
"""
import argparse

import cv2
import numpy as np
import torch

from common import Clip, device, read_json, work_dir, write_json


def phash(picture):
    gray = cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(gray)[:8, :8].ravel()
    return low > np.median(low[1:])


def describe(clip, pictures):
    return clip.images(pictures), np.stack([phash(p) for p in pictures])


def embed(clip, paths_or_frames, label):
    embeddings, hashes = [], []
    for start in range(0, len(paths_or_frames), 256):
        chunk = [p if isinstance(p, np.ndarray) else cv2.imread(p) for p in paths_or_frames[start:start+256]]
        chunk = [c if c is not None else np.zeros((64, 64, 3), np.uint8) for c in chunk]
        e, h = describe(clip, chunk)
        embeddings.append(e)
        hashes.append(h)
        if start % 5120 == 0:
            print(f'{label}: {start}/{len(paths_or_frames)}', flush=True)
    return np.concatenate(embeddings), np.concatenate(hashes)


def video_frames(videos, every_s=2.0):
    frames = []
    for v in videos:
        capture = cv2.VideoCapture(v['path'])
        step, index = max(1, round(v['fps'] * every_s)), 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                frames.append(cv2.resize(frame, (448, int(448 * frame.shape[0] / frame.shape[1]))))
            index += 1
        capture.release()
    return frames


def max_similarity(a, b, chunk=4096):
    """Max cosine and min Hamming of each row of a against all rows of b (GPU when available)."""
    dev = device()
    ea, eb = torch.from_numpy(a[0]).to(dev).half(), torch.from_numpy(b[0]).to(dev).half()
    ha, hb = torch.from_numpy(a[1]).to(dev).half(), torch.from_numpy(b[1]).to(dev).half()
    cos, ham = [], []
    for s in range(0, len(ea), chunk):
        sim = ea[s:s+chunk] @ eb.T
        x = ha[s:s+chunk]
        distance = x @ (1 - hb).T + (1 - x) @ hb.T
        cos.append(sim.max(1).values.float().cpu())
        ham.append(distance.min(1).values.float().cpu())
    return torch.cat(cos).numpy(), torch.cat(ham).numpy()


class Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--leak-cos', type=float, default=.95)
    p.add_argument('--leak-hamming', type=int, default=6)
    p.add_argument('--group-cos', type=float, default=.92)
    a = p.parse_args()
    work = work_dir()
    sources = work / 'sources'
    new = []
    for name in ('dfire.json', 'openimages.json', 'roboflow.json', 'screens.json', 'own_frames.json'):
        if (sources / name).exists():
            new += read_json(sources / name)['records']
    evaluation = [r['image'] for name in ('validation_full.json', 'test.json')
                  for r in read_json(work / 'manifests_fasdd' / name)]
    if (sources / 'bowfire.json').exists():
        evaluation += [r['image'] for r in read_json(sources / 'bowfire.json')['records']]
    videos = read_json(sources / 'videos.json')['videos'] if (sources / 'videos.json').exists() else []
    clip = Clip()
    eval_desc = embed(clip, evaluation + video_frames([v for v in videos if v['split'] == 'eval']), 'evaluation')
    new_desc = embed(clip, [r['image'] for r in new], 'new')
    cos, ham = max_similarity(new_desc, eval_desc)
    leaked = (cos >= a.leak_cos) | (ham <= a.leak_hamming)
    # own frames come from clips already split by clip; only cross-source leaks matter for them
    union = Union(len(new))
    base = {}
    for i, r in enumerate(new):
        if r['group'] in base:
            union.join(i, base[r['group']])
        base.setdefault(r['group'], i)
    dev = device()
    e = torch.from_numpy(new_desc[0]).to(dev).half()
    links = 0
    for s in range(0, len(e), 4096):
        sim = e[s:s+4096] @ e.T
        rows, cols = torch.nonzero(sim >= a.group_cos, as_tuple=True)
        for i, j in zip((rows + s).tolist(), cols.tolist()):
            if i < j:
                union.join(i, j)
                links += 1
    groups = {r['image']: f'g{union.find(i)}' for i, r in enumerate(new)}
    result = dict(leaked=[r['image'] for r, bad in zip(new, leaked) if bad], groups=groups,
                  thresholds=vars(a), near_duplicate_links=links,
                  leaked_by_source={s: int(sum(bad for r, bad in zip(new, leaked) if r['source'] == s))
                                    for s in {r['source'] for r in new}})
    write_json(result, work / 'dedup' / 'dedup.json')
    print('leaked (dropped):', result['leaked_by_source'], '| groups', len(set(groups.values())), '| links', links)


if __name__ == '__main__':
    main()
