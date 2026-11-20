"""Zone obstruction monitor: detector -> blocked cells -> per-cell dwell -> zone percentages -> zone alarms.

Independent of the flame detector: it reads the undrawn frame and a timestamp in seconds and never touches flame
boxes or the flame alarm. Image-space percentages only; floor area comes from a Week 3 projector when available.
"""
import cv2
import numpy as np

from .blindmap import BlindMap
from .confirm import ZoneAlarm
from .detector import ReferenceDetector
from .grid import CellTimers, ZoneGrid
from .hidden import FireVisibility
from .motion import CameraMotion
from .ground import NullProjector


class SceneMonitor:
    def __init__(self, config, detector=None, gap_s=1.5, min_cell_fraction=.5, projector=None, exclude_margin=.5,
                 visibility=None, view_changed_pct=60., require_confirmation=False, motion=None):
        self.config = config
        self.detector = detector or ReferenceDetector()
        self.gap_s, self.min_cell_fraction = gap_s, min_cell_fraction
        self.projector = projector or NullProjector()
        self.grid = self.timers = self.blind_map = None
        self.alarms = {z.name: ZoneAlarm(z.name, z.min_blocked_pct, z.clear_s) for z in config.zones}
        self.exclude_margin = exclude_margin    # grow each excluded box by this share of its longer side
        self._freeze = None
        # Previously detected fire - visibility compromised: follows the fires the Week 2 alarm confirmed; it has
        # no flame tracking of its own and never changes the flame alarm.
        self.visibility = FireVisibility() if visibility is None else visibility or None
        self.view_changed_pct = view_changed_pct   # this share of the frame changed at once: view changed
        # Operator confirmation: after learning, nothing is armed until confirm() (the operator has checked the
        # learned empty scene, saved by the caller via reference_image()).
        self.require_confirmation, self.confirmed = require_confirmation, False
        # Camera movement (shift / rotation / zoom) against the learned scene; False switches it off.
        self.motion = CameraMotion() if motion is None else motion or None

    def exclusion(self, boxes, frame_shape):
        """Bool mask at analysis size covering the given frame-pixel boxes (flames) plus a margin for glow."""
        if boxes is None or not len(boxes):
            return None
        width, height = self.detector.size
        sx, sy = width / frame_shape[1], height / frame_shape[0]
        mask = np.zeros((height, width), bool)
        for x1, y1, x2, y2 in np.asarray(boxes, float)[:, :4]:
            grow = self.exclude_margin * max(x2 - x1, y2 - y1)
            a = (max(0, int((x1 - grow) * sx)), max(0, int((y1 - grow) * sy)))
            b = (min(width, int(np.ceil((x2 + grow) * sx))), min(height, int(np.ceil((y2 + grow) * sy))))
            mask[a[1]:b[1], a[0]:b[0]] = True
        return mask

    def update(self, frame, t, exclude_boxes=None, flame=None):
        """exclude_boxes: [M, >=4] xyxy in frame pixels that must never count as obstruction (flame detections).
        flame: the Week 2 alarm's output for this frame (src/flame_alarm.py), for the fire visibility check.

        Returns dict(state, zones={name: dict(blocked_pct, persistent_pct, active, area_m2)}, frame_pct,
        events=[finished events], cells, dwell, blobs, foreground_pct, gain, excluded_pct)."""
        self.detector.set_size(frame.shape)      # the exclusion mask needs the analysis size
        exclude = self.exclusion(exclude_boxes, frame.shape)
        result = self.detector.update(frame, t, freeze=self._freeze, exclude=exclude)
        result['excluded_pct'] = 0. if exclude is None else float(100 * exclude.mean())
        ready = result['state'] == 'ready'
        moved = self._camera_motion(frame, t) if ready and self.motion else None
        result['camera_motion'] = moved
        result['view_changed'] = ready and (result['foreground_pct'] >= self.view_changed_pct
                                            or bool(moved and moved['moved']))
        result['visibility'] = (self.visibility.update(flame, t, result['mask'] if ready else None,
                                                       result.get('unknown') if ready else None, frame.shape,
                                                       result['view_changed'])
                                if self.visibility else dict(visibility_compromised=False, compromised_unknown=False,
                                                             fire_area_covered_pct=0., fires=[], events=[]))
        if self.grid is None:
            width, height = self.detector.size
            self.grid = ZoneGrid(self.config, width, height)
            self.timers = CellTimers((self.grid.rows, self.grid.cols), self.gap_s)
            # Per cell, the persistence of the strictest zone containing it (cells in no zone: strictest overall).
            persist = np.full((self.grid.rows, self.grid.cols), min(z.persist_s for z in self.config.zones))
            for z in sorted(self.config.zones, key=lambda z: -z.persist_s):
                persist[self.grid.weights[z.name] > 0] = z.persist_s
            self.blind_map = BlindMap(self.grid, np.maximum(persist, 1e-9))
        if result['state'] == 'ready' and self.require_confirmation and not self.confirmed:
            result['state'] = 'awaiting_confirmation'
        if result['state'] == 'ready' and result['view_changed']:
            # The learned scene no longer matches the view: no obstruction is judged (nothing new can alarm, open
            # alarms are closed), the blind map pauses and cell timers restart once the view is back.
            events = [e for e in (a.close(t) for a in self.alarms.values()) if e]
            self.timers.reset()
            self._freeze = None
            self.blind_map.pause()
            zones = {z.name: dict(blocked_pct=0., persistent_pct=0., active=False, area_m2=None)
                     for z in self.config.zones}
            return dict(result, state='view_changed', zones=zones, frame_pct=0., events=events, cells=None,
                        dwell=None)
        if result['state'] != 'ready':
            self.blind_map.pause()
            zones = {z.name: dict(blocked_pct=0., persistent_pct=0., active=False, area_m2=None)
                     for z in self.config.zones}
            return dict(result, zones=zones, frame_pct=0., events=[], cells=None, dwell=None)
        cells = self.grid.blocked_cells(result['mask'], self.min_cell_fraction)
        dwell = self.timers.update(cells, t)
        self.blind_map.update(cells, dwell, t)
        # Cells blocked now or held through a short gap are frozen: the reference never learns an obstruction.
        held = dwell > 0
        self._freeze = cv2.resize(held.astype(np.uint8), self.detector.size,
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        now = self.grid.percent(cells)
        zones, events = {}, []
        for z in self.config.zones:
            # Held cells keep their timer through a gap but only count while actually blocked, so an event ends
            # when the object leaves; the alarm's clear_s rides over brief detection drop-outs.
            persistent = cells & (dwell >= max(z.persist_s, 1e-9))
            pct = self.grid.percent(persistent)[z.name]
            active, finished = self.alarms[z.name].update(pct, t)
            if finished:
                events.append(finished)
            zones[z.name] = dict(blocked_pct=now[z.name], persistent_pct=pct, active=active,
                                 area_m2=self.projector.blocked_area_m2(z.name, persistent, self.grid))
        return dict(result, zones=zones, frame_pct=now['_frame'], events=events, cells=cells, dwell=dwell)

    def _camera_motion(self, frame, t):
        """Measured camera movement against the learned scene, with the shift in frame pixels."""
        if self.motion.reference is None:
            self.motion.set_reference(self.reference_image())
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        grey = cv2.GaussianBlur(cv2.resize(grey, self.detector.size, interpolation=cv2.INTER_AREA), (5, 5), 0)
        out = dict(self.motion.update(grey, t))
        if out['shift_px'] is not None:
            out['shift_px'] *= frame.shape[1] / self.detector.size[0]
        return out

    def confirm(self):
        """The operator has checked the learned empty scene: arm the monitor."""
        self.confirmed = True

    def reference_image(self):
        """The learned empty scene (grey, analysis size, uint8) for the operator to check, or None."""
        detector = getattr(self.detector, 'pixel', self.detector)
        return None if detector.reference is None else np.clip(detector.reference, 0, 255).astype(np.uint8)

    def relearn(self, t):
        """Re-learn the empty scene (after a deliberate camera move); open alarms are closed first."""
        events = self.close(t)
        self.confirmed = False
        if self.motion:
            self.motion.reset()
        if self.visibility:
            self.visibility.reset()
        self.detector.relearn()
        if self.timers is not None:
            self.timers.reset()
        self._freeze = None
        if self.blind_map is not None:
            self.blind_map.pause()
        return events

    def close(self, t):
        """Finish any open zone alarm events (end of stream)."""
        return [e for e in (a.close(t) for a in self.alarms.values()) if e]
