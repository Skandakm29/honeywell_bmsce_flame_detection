"""Class-agnostic scene-change detection against a long-term reference of the unobstructed scene.

No model and no training: anything that differs persistently from the reference (a person, a vehicle, a box, a
sheet) is foreground. Works on a small greyscale copy of the frame (default 320 px wide).

- Learning: learn_s seconds of a STILL scene are averaged into the reference; nothing is reported as blocked
  meanwhile. Any motion during learning (someone walking out of view) restarts it, so a moving person is not
  learned as background. A person standing perfectly still is: keep the zone clear while learning.
- Lighting: a global gain (median ratio) is removed before comparing, so a whole-room brightness change is not
  foreground. Local shadows and auto-exposure in parts of the image can still be (validate on real clips).
- Flame at start-up: pixels inside flame boxes (exclude) for >= unknown_share of the learning frames are UNKNOWN:
  a fire burning while learning never becomes "normal". Unknown pixels are compared with the scene as learned
  (with the fire) but count as foreground only while the known ring around them is blocked too: a forklift in
  front of the fire is seen, the fire simply going out is not. They are filled in once no flame has been there for
  fill_s seconds, the area is still and the ring is background (an object parked in front is never learned).
- Calibration check: if no still period long enough to learn has been found within learn_timeout_s, the result
  says calibration_failed (the scene is unsuitable: keep the zone clear and re-learn) instead of waiting silently.
- Ghost correction: something that was in view while learning and then LEFT leaves a "ghost": its old outline is
  in the reference, not in the live image, and the live image inside it looks like the background around it. A
  blob for which both hold for ghost_s seconds is re-learned from the live image (logged as a ghost event). A real
  blocker has its outline in the live image, and a plain object or steam over a textured background does not look
  like its surroundings, so neither is re-learned.
- Reference update: only pixels that are background AND outside `freeze` (cells the caller considers blocked)
  adapt, with time constant adapt_s seconds. A stationary obstruction is therefore never learned into the
  reference, unlike plain MOG2 background subtraction.
"""
import math

import cv2
import numpy as np


class ReferenceDetector:
    def __init__(self, width=320, learn_s=3., adapt_s=600., grey_threshold=25., edge_threshold=40.,
                 min_blob_fraction=.002, still_threshold=.02, unknown_share=.2, fill_s=5., fill_ring=7,
                 learn_timeout_s=30., ghost_s=2., ghost_ratio=1.5, ghost_texture=.5, ghost_min_edge=8.):
        if width < 16 or learn_s < 0 or adapt_s <= 0:
            raise ValueError('width >= 16, learn_s >= 0 and adapt_s > 0 are required')
        self.width, self.learn_s, self.adapt_s = width, learn_s, adapt_s
        self.grey_threshold, self.edge_threshold = grey_threshold, edge_threshold
        self.min_blob_fraction = min_blob_fraction
        self.still_threshold = still_threshold   # share of pixels allowed to change between learning frames
        self.unknown_share, self.fill_s, self.fill_ring = unknown_share, fill_s, fill_ring
        self.ring_blocked_share = .05            # share of the ring that must be foreground to call it blocked
        self.learn_timeout_s = learn_timeout_s
        self.ghost_s, self.ghost_ratio, self.ghost_texture = ghost_s, ghost_ratio, ghost_texture
        self.ghost_min_edge = ghost_min_edge     # mean reference edge strength on the outline to judge at all
        self.size = None                         # (width, height) of the analysis image
        self.relearn()

    def relearn(self):
        """Forget the reference and learn it again from the next frames (after a deliberate camera move)."""
        self.reference = self.reference_edges = None
        self._sum = self._edge_sum = self._seen = None
        self._count, self._learn_start, self._last_t, self._previous = 0, None, None, None
        self.unknown = None                      # bool [h, w]: never seen without a flame while learning
        self._fill_since = self._fill_previous = None
        self._first_learn_t = None
        self._ghost_since = None
        self.ghost_events = []                   # (time, share of the image re-learned)

    @property
    def ready(self):
        return self.reference is not None

    def set_size(self, frame_shape):
        """Fix the analysis size (width x proportional height) from the first frame's shape."""
        if self.size is None:
            h, w = frame_shape[:2]
            self.size = (self.width, max(1, round(h * self.width / w)))
        return self.size

    def _prepare(self, frame):
        self.set_size(frame.shape)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        grey = cv2.GaussianBlur(cv2.resize(grey, self.size, interpolation=cv2.INTER_AREA), (5, 5), 0)
        grey = grey.astype(np.float32)
        return grey, self._edges(grey)

    @staticmethod
    def _edges(grey):
        return cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))

    def update(self, frame, t, freeze=None, exclude=None):
        """frame: BGR or grey image; t: seconds; freeze: optional bool mask at analysis size, never adapted;
        exclude: optional bool mask at analysis size that is never foreground and never adapted (flame boxes).

        Returns dict(state='learning'|'ready', mask=bool [h, w] foreground, blobs=[(x1, y1, x2, y2, pixels)],
        gain, foreground_pct)."""
        grey, edges = self._prepare(frame)
        h, w = grey.shape
        empty = np.zeros((h, w), bool)
        if not self.ready:
            changed = np.abs(grey - self._previous) > self.grey_threshold if self._previous is not None else None
            if changed is not None and exclude is not None:
                changed &= ~exclude                  # a flickering flame is not "scene moving"
            moving = changed is not None and changed.mean() > self.still_threshold
            self._previous = grey
            if self._learn_start is None or moving:
                self._count = 0
                self._learn_start, self._sum, self._edge_sum = t, np.zeros_like(grey), np.zeros_like(grey)
                self._seen, self._all, self._edge_all = np.zeros_like(grey), np.zeros_like(grey), np.zeros_like(grey)
            seen = np.ones((h, w), bool) if exclude is None else ~exclude   # flame pixels are not learned
            self._sum[seen] += grey[seen]
            self._edge_sum[seen] += edges[seen]
            self._seen += seen
            self._all += grey
            self._edge_all += edges
            self._count += 1
            if t - self._learn_start >= self.learn_s:
                n = np.maximum(self._seen, 1)
                self.reference, self.reference_edges = self._sum / n, self._edge_sum / n
                self.unknown = self._seen < (1 - self.unknown_share) * self._count
                if self.unknown.any():          # the scene as learned, flame included
                    self.reference[self.unknown] = self._all[self.unknown] / self._count
                    self.reference_edges[self.unknown] = self._edge_all[self.unknown] / self._count
                else:
                    self.unknown = None
                self._sum = self._edge_sum = self._seen = self._previous = self._all = self._edge_all = None
            self._last_t = t
            if self._first_learn_t is None:
                self._first_learn_t = t
            remaining = 0. if self.ready else max(0., self.learn_s - (t - self._learn_start))
            return dict(state='learning', mask=empty, blobs=[], gain=1., foreground_pct=0., moving=moving,
                        learn_remaining_s=remaining, learning_for_s=t - self._first_learn_t,
                        calibration_failed=not self.ready and t - self._first_learn_t >= self.learn_timeout_s)
        gain = float(np.median(self.reference) / max(1., float(np.median(grey))))
        grey_n, edges_n = grey * gain, edges * gain
        foreground = ((np.abs(grey_n - self.reference) > self.grey_threshold)
                      | (np.abs(edges_n - self.reference_edges) > self.edge_threshold)).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        keep = np.zeros(count, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_blob_fraction * h * w
        mask = keep[labels]
        if exclude is not None:
            mask &= ~exclude
        unjudged = None
        if self.unknown is not None:
            ring = self._ring()
            if not (ring.any() and (mask & ring).sum() / ring.sum() > self.ring_blocked_share):
                mask &= ~self.unknown           # ring clear: a change here is the flame itself, not a blocker
                unjudged = self.unknown
        healed = self._ghosts(mask, labels, count, keep, grey_n, edges_n, t)
        if healed is not None:
            mask &= ~healed
        blobs = [(int(x), int(y), int(x + bw), int(y + bh), int(a))
                 for (x, y, bw, bh, a), k in zip(stats, keep) if k]
        # Adapt only background pixels outside frozen cells; time-based so it does not depend on frame rate.
        dt = max(0., t - self._last_t) if self._last_t is not None else 0.
        alpha = 1 - math.exp(-dt / self.adapt_s)
        adapt = ~mask if freeze is None else ~mask & ~freeze
        if exclude is not None:
            adapt &= ~exclude
        if self.unknown is not None:
            adapt &= ~self.unknown
        if alpha > 0 and adapt.any():
            self.reference[adapt] += alpha * (grey_n[adapt] - self.reference[adapt])
            self.reference_edges[adapt] += alpha * (edges_n[adapt] - self.reference_edges[adapt])
        self._last_t = t
        unknown = self.unknown
        self._fill_unknown(grey_n, edges_n, mask, exclude, t)
        return dict(state='ready', mask=mask, blobs=blobs, gain=gain, foreground_pct=float(100 * mask.mean()),
                    unknown=unjudged, unknown_pct=0. if unknown is None else float(100 * unknown.mean()),
                    healed=healed, healed_pct=0. if healed is None else float(100 * healed.mean()))

    def _ghosts(self, mask, labels, count, keep, grey_n, edges_n, t):
        """Re-learn blobs that are ghosts of something that left (see the module notes). Returns the bool mask of
        pixels re-learned now, or None."""
        h, w = mask.shape
        if self._ghost_since is None:
            self._ghost_since = np.full((h, w), np.nan)
        ghost_now = np.zeros((h, w), bool)
        kernel = np.ones((3, 3), np.uint8)
        for i in np.flatnonzero(keep):
            blob = (labels == i) & mask
            if blob.sum() < 20:
                continue
            b8 = blob.astype(np.uint8)
            outline = cv2.dilate(b8, kernel, iterations=2).astype(bool) & ~cv2.erode(b8, kernel, iterations=2).astype(bool)
            outside = cv2.dilate(b8, kernel, iterations=6).astype(bool) & ~cv2.dilate(b8, kernel, iterations=2).astype(bool)
            inside = cv2.erode(b8, kernel, iterations=2).astype(bool)
            if not outline.any() or not outside.any() or not inside.any():
                continue
            e_ref, e_live = float(self.reference_edges[outline].mean()), float(edges_n[outline].mean())
            outline_left = e_ref >= self.ghost_min_edge and e_ref > self.ghost_ratio * e_live
            # The live image inside looks like the live background around it (texture energy).
            texture_in, texture_out = float(edges_n[inside].mean()), float(edges_n[outside].mean())
            looks_like_background = (min(texture_in, texture_out) >= self.ghost_texture * max(texture_in, texture_out)
                                     or max(texture_in, texture_out) < 2.)
            if outline_left and looks_like_background:
                ghost_now |= blob
        self._ghost_since[~ghost_now] = np.nan
        self._ghost_since[ghost_now & np.isnan(self._ghost_since)] = t
        heal = ghost_now & (t - self._ghost_since >= self.ghost_s - 1e-9)
        if not heal.any():
            return None
        self.reference[heal] = grey_n[heal]
        self.reference_edges[heal] = edges_n[heal]
        self._ghost_since[heal] = np.nan
        self.ghost_events.append((t, float(100 * heal.mean())))
        return heal

    def _ring(self):
        """Known pixels within fill_ring of the unknown area (cached until the unknown area changes)."""
        if getattr(self, '_ring_for', None) is not self.unknown:
            kernel = np.ones((2 * self.fill_ring + 1,) * 2, np.uint8)
            self._ring_mask = cv2.dilate(self.unknown.astype(np.uint8), kernel).astype(bool) & ~self.unknown
            self._ring_for = self.unknown
        return self._ring_mask

    def _fill_unknown(self, grey_n, edges_n, mask, exclude, t):
        """Learn the unknown (start-up flame) area once it has been flame-free, still and surrounded by background
        for fill_s seconds. Any condition failing restarts the wait."""
        if self.unknown is None:
            return
        ring = self._ring()
        flame = exclude is not None and (exclude & (ring | self.unknown)).any()
        blocked_ring = ring.any() and (mask & ring).sum() / ring.sum() > self.ring_blocked_share
        moving = (self._fill_previous is not None and
                  (np.abs(grey_n[self.unknown] - self._fill_previous) > self.grey_threshold).mean() > .05)
        self._fill_previous = grey_n[self.unknown].copy()
        if flame or blocked_ring or moving:
            self._fill_since = None
            return
        if self._fill_since is None:
            self._fill_since = t
        if t - self._fill_since >= self.fill_s:
            self.reference[self.unknown] = grey_n[self.unknown]
            self.reference_edges[self.unknown] = edges_n[self.unknown]
            self.unknown = self._fill_since = self._fill_previous = None
