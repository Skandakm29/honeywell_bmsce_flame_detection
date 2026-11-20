"""Seconds-based zone obstruction alarm with hysteresis, independent of frame rate.

ON  as soon as the persistently blocked share reaches min_blocked_pct (persistence is already timed per cell).
OFF once the share has stayed below min_blocked_pct for clear_s seconds.
Each finished or open alarm is one event with its start, end, peak and mean blocked share.
"""


class ZoneAlarm:
    def __init__(self, name, min_blocked_pct, clear_s):
        self.name, self.min_blocked_pct, self.clear_s = name, min_blocked_pct, clear_s
        self.active = False
        self.event = None
        self._below_since = None

    def update(self, persistent_pct, t):
        """Returns (active, finished_event or None)."""
        above = persistent_pct >= self.min_blocked_pct
        finished = None
        if not self.active and above:
            self.active, self._below_since = True, None
            self.event = dict(zone=self.name, start_s=t, end_s=None, peak_pct=persistent_pct,
                              sum_pct=0., samples=0, _pending=[0., 0])
        elif self.active:
            if above:
                self._below_since = None
                pending = self.event['_pending']         # a short dip belongs to the event
                self.event['sum_pct'] += pending[0]
                self.event['samples'] += pending[1]
                self.event['_pending'] = [0., 0]
            elif self._below_since is None:
                self._below_since = t
            if self._below_since is not None and t - self._below_since >= self.clear_s:
                finished = self.close(t)
        if self.active:
            self.event['peak_pct'] = max(self.event['peak_pct'], persistent_pct)
            if above:
                self.event['sum_pct'] += persistent_pct
                self.event['samples'] += 1
            else:
                self.event['_pending'][0] += persistent_pct
                self.event['_pending'][1] += 1
        return self.active, finished

    def close(self, t):
        """End the open event (also when the stream stops or the scene is re-learned).

        The end is when the share first fell below the threshold if it is still below, otherwise t. Samples from
        that trailing below-threshold spell are left out of mean_pct."""
        if not self.active:
            return None
        event = dict(self.event, end_s=self._below_since if self._below_since is not None else t)
        event.pop('_pending')
        event['mean_pct'] = event.pop('sum_pct') / max(1, event.pop('samples'))
        self.active, self.event, self._below_since = False, None, None
        return event
