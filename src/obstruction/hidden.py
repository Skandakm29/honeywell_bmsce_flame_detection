"""Previously detected fire - visibility compromised.

Builds on the Week 2 flame alarm (src/flame_alarm.py): it does no flame tracking or confirmation of its own. It
follows the fires that Week 2 CONFIRMED (a track that met the alarm rule) and checks whether the area where each
was last seen is now covered by something, using the zone monitor's foreground mask. It never changes flame
detection, flame boxes or the flame alarm.

This does NOT establish that the fire is still burning: it only says that the area of a previously detected fire
can no longer be seen. The fire may have gone out, or the detector may simply have missed it.

Per remembered fire (keyed by its Week 2 track id; remembered after Week 2 drops the track):
- visible:      Week 2 still reports the track this frame.
- lost:         not seen for lost_s; area not (yet) covered. Grace period of clear_s for a blocker to arrive.
- compromised:  not seen for lost_s AND >= cover_share of the known pixels in its area (last box grown by its own
                size on every side) are foreground. Stays on as long as that holds, with no time limit.
- compromised_unknown: not seen and too little of its area can be judged (flame there while learning).
- forgotten:    the area stayed clear for clear_s (fire out or moved), a flame is seen again in or right next to
                its area (it is visible again, possibly under a new Week 2 track), or the view changed.
fire_area_covered_pct: per frame, the largest covered share of any remembered fire's area. Unlike the zone %, it
measures the area of the fire itself, and it is reported while the fire is still visible too.
"""
import numpy as np


class FireVisibility:
    def __init__(self, lost_s=.5, cover_share=.3, clear_s=2., grow=1., min_known_share=.2):
        self.lost_s, self.cover_share, self.clear_s = lost_s, cover_share, clear_s
        self.grow, self.min_known_share = grow, min_known_share
        self.fires = {}

    def _near(self, box, other):
        """Is other in or right next to the area checked around box (box grown by 1.5 x grow x its longer side)?"""
        g = 1.5 * self.grow * max(box[2] - box[0], box[3] - box[1])
        return other[0] < box[2] + g and other[2] > box[0] - g and other[1] < box[3] + g and other[3] > box[1] - g

    def _cover(self, box, mask, unknown, frame_shape):
        """Share of the known pixels in the fire's area that are foreground (None if it cannot be judged)."""
        if mask is None:
            return None
        h, w = mask.shape
        sx, sy = w / frame_shape[1], h / frame_shape[0]
        x1, y1, x2, y2 = box
        g = self.grow * max(x2 - x1, y2 - y1)
        a = (max(0, int((x1 - g) * sx)), max(0, int((y1 - g) * sy)))
        b = (min(w, int(np.ceil((x2 + g) * sx))), min(h, int(np.ceil((y2 + g) * sy))))
        region = mask[a[1]:b[1], a[0]:b[0]]
        if region.size == 0:
            return None
        known = np.ones_like(region) if unknown is None else ~unknown[a[1]:b[1], a[0]:b[0]]
        if known.mean() < self.min_known_share:
            return None
        return float((region & known).sum() / known.sum())

    def update(self, flame, t, mask, unknown, frame_shape, view_changed=False):
        """flame: FlameAlarm.update() output (Week 2 tracks), or None. mask/unknown: bool at analysis size (None
        while learning). Returns dict(visibility_compromised, compromised_unknown, fire_area_covered_pct,
        fires=[...], events=[finished compromised spells])."""
        tracks = flame['tracks'] if flame else []
        seen_now = [k['box'] for k in tracks if k['last_seen'] == t]
        events = []
        for k in tracks:                                   # remember / refresh Week 2 confirmed fires
            if not k['confirmed']:
                continue
            fire = self.fires.setdefault(k['id'], dict(id=k['id'], state='visible', since=None, clear_since=None,
                                                       cover=None))
            if k['last_seen'] == t and fire['state'] == 'compromised':
                events.append(self._finish(fire, t, 'fire visible again'))   # before last_seen is refreshed
            fire.update(box=tuple(k['box']), last_seen=k['last_seen'])
            if k['last_seen'] == t:
                fire.update(state='visible', clear_since=None)
        for fire in self.fires.values():
            fire['cover'] = self._cover(fire['box'], mask, unknown, frame_shape)
            if fire['state'] == 'forgotten' or fire['last_seen'] == t:
                continue
            if view_changed:
                if fire['state'] == 'compromised':
                    events.append(self._finish(fire, t, 'view changed'))
                fire['state'] = 'forgotten'
                continue
            if any(self._near(fire['box'], b) for b in seen_now):
                # A flame is seen in or right next to this fire's area: it is visible (maybe as a new track).
                if fire['state'] == 'compromised':
                    events.append(self._finish(fire, t, 'fire visible nearby'))
                fire['state'] = 'forgotten'
                continue
            if t - fire['last_seen'] < self.lost_s:
                continue
            if fire['cover'] is None:
                if fire['state'] != 'compromised' and mask is not None:
                    fire['state'] = 'compromised_unknown'
                continue
            if fire['cover'] >= self.cover_share:
                fire['clear_since'] = None
                if fire['state'] != 'compromised':
                    fire.update(state='compromised', since=t)
                continue
            fire['clear_since'] = fire['clear_since'] if fire['clear_since'] is not None else t
            if t - fire['clear_since'] >= self.clear_s:
                if fire['state'] == 'compromised':
                    events.append(self._finish(fire, t, 'area clear'))
                fire['state'] = 'forgotten'          # fire out, or moved away
            elif fire['state'] != 'compromised':
                fire['state'] = 'lost'
        self.fires = {k: f for k, f in self.fires.items() if f['state'] != 'forgotten'}
        covers = [f['cover'] for f in self.fires.values() if f['cover'] is not None]
        return dict(visibility_compromised=any(f['state'] == 'compromised' for f in self.fires.values()),
                    compromised_unknown=any(f['state'] == 'compromised_unknown' for f in self.fires.values()),
                    fire_area_covered_pct=100 * max(covers) if covers else 0.,
                    fires=[dict(id=f['id'], state=f['state'], box=f['box'], cover=f['cover'],
                                last_seen=f['last_seen']) for f in self.fires.values()],
                    events=events)

    def _finish(self, fire, t, reason):
        event = dict(track_id=fire['id'], start_s=fire['since'], end_s=t, reason=reason,
                     last_seen_s=fire['last_seen'], box=[round(v, 1) for v in fire['box']])
        fire['since'] = None
        return event

    def close(self, t):
        """Finish compromised spells still open at the end of the stream."""
        return [self._finish(f, t, 'end of stream') for f in self.fires.values() if f['state'] == 'compromised']

    def reset(self):
        self.fires = {}
