"""Deep-feature scene-change detection (DINOv2 ViT-S/14 patch features, PatchCore/PaDiM style). No training.

Same interface as ReferenceDetector, so SceneMonitor, zones, timers, alarms and fire-hidden tracking are unchanged.
It wraps a ReferenceDetector, which still decides when the scene is still enough to learn, the flame-at-start-up
unknown area and the analysis size; this class replaces only "which pixels changed".

- Features: the frame (RGB, ImageNet normalised) is resized to the model input (default 252 x 448 -> 18 x 32
  patches of 14 px) and DINOv2 gives one 384-d feature per patch. They describe what is there (texture, parts,
  objects), so a global or local brightness change moves them much less than raw grey levels.
- Memory bank: while the pixel detector learns the still scene, a feature map is stored every sample_s seconds.
- Score per patch: 1 - max cosine similarity to the bank at the same patch or its 8 neighbours (tolerates a
  small shift). A patch is changed if its score > threshold.
- Threshold from data, no labels: leave-one-out scores of the learning samples against the rest of the bank give
  the normal spread; threshold = max(min_threshold, factor x their 99.5th percentile).
- Slow adaptation: every bank_update_s, patches that are unchanged, not frozen (blocked cells) and not flame
  replace their oldest bank entry, so a blocker is never learned.
- method 'deep': changed patches are the mask. 'fused': pixel foreground AND (changed patch, grown by one patch):
  pixel outlines, deep decision.
- Cost: about 100 ms per feature map on a laptop CPU (12 threads). every_s > 0 reuses the last map between
  updates; background=True runs it in a worker thread so the caller does not wait for it. The thread still shares
  the CPU with flame detection: measured with V12 on the laptop, flame time went from 133 ms to 397 ms per frame
  with DINOv2 5x/s on all cores, and to 169 ms with 1 thread once per second (the live default in live_flame.py).
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import ReferenceDetector

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / 'models/dinov2/dinov2_vits14_252x448.onnx'


class OnnxPatchFeatures:
    """DINOv2 patch features from an ONNX file with input image [1, 3, H, W] and output [1, H/14 * W/14, D]."""
    def __init__(self, path=DEFAULT_MODEL, threads=0):
        import onnxruntime as ort
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'{path} not found: export DINOv2 to ONNX first (see docs/WEEK4_OBSTRUCTION.md)')
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        _, _, self.height, self.width = self.session.get_inputs()[0].shape
        self.rows, self.cols = self.height // 14, self.width // 14

    def __call__(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        tensor = ((image - MEAN) / STD).transpose(2, 0, 1)[None]
        features = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][0]
        features = features.reshape(self.rows, self.cols, -1)
        return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-6)


def neighbour_similarity(features, bank):
    """Max cosine similarity of each patch [R, C, D] to bank [S, R, C, D] at the same or a neighbouring patch."""
    rows, cols, _ = features.shape
    padded = np.pad(bank, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='edge')
    best = np.full((rows, cols), -1., np.float32)
    for dy in range(3):
        for dx in range(3):
            shifted = padded[:, dy:dy + rows, dx:dx + cols]
            best = np.maximum(best, np.einsum('srcd,rcd->src', shifted, features).max(0))
    return best


class _Worker:
    """Runs the feature extractor on the newest submitted frame in a background thread."""
    def __init__(self, extract):
        self.extract, self.result, self.error = extract, None, None
        self._job, self._busy, self._stop = None, False, False
        self._lock, self._wake = threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=5.):
        """Let a running feature map finish, then end the thread (clean exit; no half-run inference)."""
        self._stop = True
        self._wake.set()
        self._thread.join(timeout)

    def submit(self, frame, t):
        with self._lock:
            if self._busy:
                return False
            self._job, self._busy = (frame.copy(), t), True
        self._wake.set()
        return True

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            frame, t = self._job
            try:
                started = time.perf_counter()
                features = self.extract(frame)
                self.result = (features, t, (time.perf_counter() - started) * 1000)
            except Exception as error:          # reported by the caller; never silently "clear"
                self.error = error
            with self._lock:
                self._busy = False


class DeepChangeDetector:
    def __init__(self, extractor=None, method='deep', pixel=None, sample_s=.25, factor=1.5, min_threshold=.1,
                 every_s=0., background=False, bank_update_s=30., max_bank=24):
        if method not in ('deep', 'fused'):
            raise ValueError("method must be 'deep' or 'fused'")
        self.extract = extractor or OnnxPatchFeatures()
        self.method, self.pixel = method, pixel or ReferenceDetector()
        self.sample_s, self.factor, self.min_threshold = sample_s, factor, min_threshold
        self.every_s, self.bank_update_s, self.max_bank = every_s, bank_update_s, max_bank
        self.worker = _Worker(self.extract) if background else None
        self.relearn()

    # ReferenceDetector interface used by SceneMonitor and live_flame.py
    def __getattr__(self, name):
        if name in ('width', 'learn_s', 'adapt_s', 'grey_threshold', 'edge_threshold', 'fill_s', 'unknown',
                    'set_size'):
            return getattr(self.pixel, name)
        raise AttributeError(name)

    @property
    def size(self):
        return self.pixel.size

    @property
    def ready(self):
        return self.pixel.ready and self.bank is not None

    def relearn(self):
        self.pixel.relearn()
        self.bank = self.threshold = None
        self._samples, self._learn_start, self._last_sample = [], None, None
        self._changed = None                    # last changed-patch map [R, C]
        self._score = None
        self._last_deep_t = self._last_bank_update = None
        self._slot = 0
        self.deep_ms = None

    def _features_now(self, frame, t):
        """Feature map for this frame, or None when the cadence/worker says to reuse the last result."""
        if self.worker is None:
            if self._last_deep_t is not None and t - self._last_deep_t < self.every_s:
                return None
            started = time.perf_counter()
            features = self.extract(frame)
            self.deep_ms, self._last_deep_t = (time.perf_counter() - started) * 1000, t
            return features
        if self.worker.error is not None:
            raise RuntimeError(f'deep feature worker failed: {self.worker.error}')
        if self._last_deep_t is None or t - self._last_deep_t >= self.every_s:
            if self.worker.submit(frame, t):
                self._last_deep_t = t
        result, self.worker.result = self.worker.result, None
        if result is None:
            return None
        self.deep_ms = result[2]
        return result[0]

    def close(self):
        """Stop the background worker, if any (call at the end of a live session)."""
        if self.worker is not None:
            self.worker.stop()

    def _calibrate(self):
        bank = np.stack(self._samples)                                   # [S, R, C, D]
        scores = []
        for s in range(len(bank)):
            rest = np.delete(bank, s, axis=0)
            scores.append(1 - neighbour_similarity(bank[s], rest))
        normal = np.concatenate([x.ravel() for x in scores])
        self.normal_q995 = float(np.quantile(normal, .995))
        self.threshold = max(self.min_threshold, self.factor * self.normal_q995)
        self.bank = bank[-self.max_bank:]

    def update(self, frame, t, freeze=None, exclude=None):
        result = self.pixel.update(frame, t, freeze=freeze, exclude=exclude)
        if result['state'] == 'learning' or self.bank is None:
            # Sample features while the pixel detector learns a still scene; restart with it.
            if self._learn_start != self.pixel._learn_start:
                # Keep the sampling cadence across restarts: a scene that never stops moving must not cost a
                # feature map on every frame.
                self._samples, self._learn_start = [], self.pixel._learn_start
            if result['state'] == 'learning' or not self._samples:
                if self._last_sample is None or t - self._last_sample >= self.sample_s:
                    started = time.perf_counter()
                    self._samples.append(self.extract(frame))
                    self.deep_ms, self._last_sample = (time.perf_counter() - started) * 1000, t
            if result['state'] == 'ready':
                if len(self._samples) < 2:          # a learn_s too short for two samples: take one more
                    self._samples.append(self.extract(frame))
                self._calibrate()
                self._last_bank_update = t
            result = dict(result, state='learning', mask=np.zeros_like(result['mask']), blobs=[])
            return dict(result, deep_ms=self.deep_ms, deep_threshold=self.threshold, deep_max=None)
        features = self._features_now(frame, t)
        if features is not None:
            self._score = 1 - neighbour_similarity(features, self.bank)
            self._changed = self._score > self.threshold
            self._adapt(features, t, freeze, exclude)
        if self._changed is None:
            mask = np.zeros_like(result['mask'])
        else:
            h, w = result['mask'].shape
            changed = self._changed
            if self.method == 'fused':
                changed = cv2.dilate(changed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            patch_mask = cv2.resize(changed.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask = patch_mask & result['mask'] if self.method == 'fused' else patch_mask
            if exclude is not None:
                mask &= ~exclude
            if result.get('unknown') is not None:   # flame-at-start-up area the pixel detector cannot judge now
                mask &= ~result['unknown']
        return dict(result, mask=mask, pixel_mask=result['mask'], foreground_pct=float(100 * mask.mean()),
                    deep_ms=self.deep_ms, deep_threshold=self.threshold,
                    deep_max=None if self._score is None else float(self._score.max()))

    def _adapt(self, features, t, freeze, exclude):
        if self.bank_update_s <= 0 or t - self._last_bank_update < self.bank_update_s:
            return
        rows, cols = self._changed.shape
        blocked = self._changed.copy()
        for extra in (freeze, exclude, self.pixel.unknown):
            if extra is not None:
                blocked |= cv2.resize(extra.astype(np.uint8), (cols, rows), interpolation=cv2.INTER_AREA) > 0
        keep = ~blocked
        self.bank[self._slot][keep] = features[keep]
        self._slot = (self._slot + 1) % len(self.bank)
        self._last_bank_update = t
