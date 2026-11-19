"""Zoom on candidate regions: a second look at weak detections.

1. Run the model on the full frame as usual. Boxes at or above `full_confidence` (0.40) are kept.
2. Weak full-frame boxes (score in [candidate_min, full_confidence)) are candidates: the model saw something
   flame-like but was not sure. Up to `max_crops` of them, strongest first.
3. Around each candidate a `crop` x `crop` square is cut from the ORIGINAL frame (centre snapped to a `grid` px
   grid so the window does not jitter frame to frame, then shifted inside the image), resized to the model input,
   and run through the model again. Crop boxes at or above `zoom_confidence` (0.20) are mapped back to
   full-frame pixels.
4. Crop boxes that overlap a kept full-frame box (IoU >= `merge_iou`) are dropped as duplicates.

The full frame is never replaced, so big fires and fires away from the candidates are still seen. Frames with no
weak candidate cost one model run, as before. Measured on the W3 1.5 m test clips (see
docs/W3_SPATIAL_CALIBRATION.md): alarm coverage 93-100% vs 62-100% for full frame alone, no false alarm on the
no-flame clips, about 1.6-2x the CPU time on frames that have candidates.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZoomConfig:
    full_confidence: float = 0.40
    zoom_confidence: float = 0.20
    candidate_min: float = 0.10
    max_crops: int = 3
    crop: int = 320
    grid: int = 32
    merge_iou: float = 0.30

    def __post_init__(self):
        if not 0 < self.candidate_min < self.full_confidence < 1:
            raise ValueError('need 0 < candidate_min < full_confidence < 1')
        if not 0 < self.zoom_confidence < 1:
            raise ValueError('zoom_confidence must be in (0, 1)')
        if self.crop < 32 or self.max_crops < 0 or self.grid < 1:
            raise ValueError('crop >= 32, max_crops >= 0, grid >= 1')


def _as_rows(boxes):
    rows = np.asarray(boxes, dtype=float).reshape(-1, 5) if len(boxes) else np.zeros((0, 5))
    return rows


def select_candidates(boxes, cfg):
    """Weak boxes worth a closer look, strongest first (at most cfg.max_crops)."""
    rows = _as_rows(boxes)
    weak = rows[(rows[:, 4] >= cfg.candidate_min) & (rows[:, 4] < cfg.full_confidence)]
    return weak[np.argsort(-weak[:, 4], kind='stable')][:cfg.max_crops]


def crop_windows(candidates, width, height, cfg):
    """Top-left corners (x0, y0) and size of the crop around each candidate; duplicates removed.
    If the frame is smaller than the crop, the crop is the whole frame (no zoom gain, still valid)."""
    size_w, size_h = min(cfg.crop, width), min(cfg.crop, height)
    out = []
    for x1, y1, x2, y2, _ in _as_rows(candidates):
        cx = int(round((x1 + x2) / 2 / cfg.grid) * cfg.grid)
        cy = int(round((y1 + y2) / 2 / cfg.grid) * cfg.grid)
        x0 = min(max(0, cx - size_w // 2), width - size_w)
        y0 = min(max(0, cy - size_h // 2), height - size_h)
        if (x0, y0, size_w, size_h) not in out:
            out.append((x0, y0, size_w, size_h))
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_object(a, b, merge_iou):
    """Overlap by IoU, or one box's centre inside the other (a crop box is often a tighter/looser box of the same
    flame, so IoU alone misses the duplicate)."""
    if _iou(a, b) >= merge_iou:
        return True
    for p, q in ((a, b), (b, a)):
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        if q[0] <= cx <= q[2] and q[1] <= cy <= q[3]:
            return True
    return False


def merge_boxes(full, zoom, merge_iou=0.30):
    """Full-frame boxes first; a zoom box is added only if it is not the same object as a box already kept.
    Returns (rows [N,5], from_zoom [N] bool)."""
    kept = [list(b) for b in _as_rows(full)]
    src = [False] * len(kept)
    for z in sorted(_as_rows(zoom).tolist(), key=lambda b: -b[4]):
        if not any(_same_object(z, k, merge_iou) for k in kept):
            kept.append(z)
            src.append(True)
    return _as_rows(kept), np.array(src, dtype=bool)


def detect_with_zoom(frame, infer, cfg=ZoomConfig()):
    """frame: HxWx3 image. infer(image) -> rows [N,5] (x1,y1,x2,y2,score) in that image's pixels, decoded at a
    score floor <= cfg.candidate_min. Returns (rows, from_zoom, windows)."""
    height, width = frame.shape[:2]
    full = _as_rows(infer(frame))
    windows = crop_windows(select_candidates(full, cfg), width, height, cfg) if cfg.max_crops else []
    zoom = []
    for x0, y0, w, h in windows:
        for x1, y1, x2, y2, s in _as_rows(infer(np.ascontiguousarray(frame[y0:y0 + h, x0:x0 + w]))):
            if s >= cfg.zoom_confidence:
                zoom.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0, s])
    rows, from_zoom = merge_boxes(full[full[:, 4] >= cfg.full_confidence], zoom, cfg.merge_iou)
    return rows, from_zoom, windows
