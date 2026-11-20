"""Session blind-zone map: how long each grid cell was blocked while the monitor was running.

Accumulated in seconds over 'ready' frames only (learning time does not count). A gap between frames is
credited for at most max_step_s, so a stall or pause cannot inflate the map. Image-space only.

  blocked_s     seconds the cell was blocked (any duration: includes people walking through)
  persistent_s  seconds the cell was blocked at least as long as its zone's persist_s (obstruction)
  longest_s     longest single persistent spell seen in the cell
"""
import json

import cv2
import numpy as np


class BlindMap:
    def __init__(self, grid, persist_s, max_step_s=1.):
        self.grid, self.persist_s, self.max_step_s = grid, persist_s, max_step_s
        shape = (grid.rows, grid.cols)
        self.blocked_s, self.persistent_s, self.longest_s = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.monitored_s, self._last_t = 0., None

    def update(self, cells, dwell, t):
        step = 0. if self._last_t is None else min(max(0., t - self._last_t), self.max_step_s)
        self._last_t = t
        persistent = cells & (dwell >= self.persist_s)
        self.blocked_s += cells * step
        self.persistent_s += persistent * step
        self.longest_s = np.maximum(self.longest_s, np.where(persistent, dwell, 0.))
        self.monitored_s += step
        return persistent

    def pause(self):
        """Stop crediting time until the next update (learning, re-learning)."""
        self._last_t = None

    def summary(self):
        """Per zone and for the whole frame: time-weighted blocked shares and the share ever obstructed."""
        def stats(weight):
            total = weight.sum()
            if total == 0 or self.monitored_s == 0:
                return dict(mean_blocked_pct=None, mean_obstructed_pct=None, ever_obstructed_pct=None,
                            longest_obstruction_s=None)
            inside = weight > 0
            return dict(
                mean_blocked_pct=round(float(100 * (weight * self.blocked_s).sum() / total / self.monitored_s), 3),
                mean_obstructed_pct=round(float(100 * (weight * self.persistent_s).sum() / total
                                                / self.monitored_s), 3),
                ever_obstructed_pct=round(float(100 * (weight * (self.persistent_s > 0)).sum() / total), 3),
                longest_obstruction_s=round(float(self.longest_s[inside].max()), 3))
        out = dict(monitored_s=round(self.monitored_s, 3), grid=[self.grid.cols, self.grid.rows],
                   zones={name: stats(w) for name, w in self.grid.weights.items()},
                   frame=stats(np.ones((self.grid.rows, self.grid.cols))))
        return out

    def save(self, folder, frame=None):
        """blind_map.json (summary + per-cell seconds) and, with a frame, blind_map.png drawn over it."""
        data = dict(self.summary(), units='seconds; image-space only (no floor m2 without Week 3 calibration)',
                    cells=dict(blocked_s=np.round(self.blocked_s, 2).tolist(),
                               persistent_s=np.round(self.persistent_s, 2).tolist(),
                               longest_s=np.round(self.longest_s, 2).tolist()))
        (folder / 'blind_map.json').write_text(json.dumps(data, indent=1), encoding='utf-8')
        if frame is not None:
            cv2.imwrite(str(folder / 'blind_map.png'), self.render(frame))

    def render(self, frame):
        """Heat map of the share of monitored time each cell was obstructed (persistent), over the frame."""
        h, w = frame.shape[:2]
        share = self.persistent_s / self.monitored_s if self.monitored_s else np.zeros_like(self.persistent_s)
        heat = cv2.applyColorMap(np.uint8(255 * share.clip(0, 1)), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
        shown = cv2.resize((share > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        view = frame.copy()
        view[shown] = cv2.addWeighted(heat, .55, frame, .45, 0)[shown]
        for z in self.grid.config.zones:
            points = np.round(np.asarray(z.polygon) * [w, h]).astype(np.int32)
            cv2.polylines(view, [points], True, (255, 255, 0), 2)
            s = self.summary()['zones'][z.name]
            label = (f'{z.name}: obstructed {s["mean_obstructed_pct"]:.0f}% of time, '
                     f'{s["ever_obstructed_pct"]:.0f}% of zone ever' if s['mean_obstructed_pct'] is not None
                     else z.name)
            cv2.putText(view, label, tuple(points[0] + [4, 18]), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1,
                        cv2.LINE_AA)
        cv2.putText(view, f'blind-zone map | {self.monitored_s:.0f} s monitored | colour = share of time obstructed',
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        return view
