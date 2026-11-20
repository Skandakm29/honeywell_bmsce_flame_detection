"""Drawing for the zone monitor, shared by scripts/test_zone_monitor.py and scripts/live_flame.py --zones.

Always draws on a copy; the monitor and the flame detector only ever see undrawn frames.
"""
import cv2
import numpy as np


def draw_zones(frame, monitor, out):
    """Zone outlines, temporarily blocked cells (yellow) and persistent cells (red), on a copy of frame."""
    view = frame.copy()
    h, w = view.shape[:2]
    grid = monitor.grid
    if out['cells'] is not None:
        persist = np.full(out['cells'].shape, np.inf)
        for z in monitor.config.zones:          # a cell is persistent for the strictest zone that contains it
            inside = grid.weights[z.name] > 0
            persist[inside] = np.minimum(persist[inside], z.persist_s)
        overlay = view.copy()
        ys = np.linspace(0, h, grid.rows + 1).round().astype(int)
        xs = np.linspace(0, w, grid.cols + 1).round().astype(int)
        for r, c in zip(*np.nonzero(out['cells'])):
            colour = (0, 0, 255) if out['dwell'][r, c] >= persist[r, c] else (0, 220, 255)
            cv2.rectangle(overlay, (xs[c], ys[r]), (xs[c + 1] - 1, ys[r + 1] - 1), colour, -1)
        view = cv2.addWeighted(overlay, .35, view, .65, 0)
    for z in monitor.config.zones:
        points = np.round(np.asarray(z.polygon) * [w, h]).astype(np.int32)
        colour = (0, 0, 255) if out['zones'][z.name]['active'] else (255, 255, 0)
        cv2.polylines(view, [points], True, colour, 2)
        cv2.putText(view, z.name, tuple(points[0] + [4, 18]), cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2)
    for polygon in monitor.config.ignore:
        cv2.polylines(view, [np.round(np.asarray(polygon) * [w, h]).astype(np.int32)], True, (128, 128, 128), 1)
    return view


def draw_visibility(view, visibility):
    """Last box of each remembered fire not seen now: magenta = visibility compromised, grey = lost / unknown."""
    if not visibility or not visibility['fires']:
        return view
    view = view.copy()
    label = {'compromised': 'PREVIOUSLY DETECTED FIRE - VISIBILITY COMPROMISED',
             'compromised_unknown': 'fire not seen - view unknown', 'lost': 'fire not seen'}
    for f in visibility['fires']:
        if f['state'] not in label:
            continue
        colour = (255, 0, 255) if f['state'] == 'compromised' else (180, 180, 180)
        x1, y1, x2, y2 = map(int, f['box'])
        for i in range(x1, x2, 12):                 # dashed: the fire is not seen, this is where it was
            cv2.line(view, (i, y1), (min(i + 6, x2), y1), colour, 2)
            cv2.line(view, (i, y2), (min(i + 6, x2), y2), colour, 2)
        for i in range(y1, y2, 12):
            cv2.line(view, (x1, i), (x1, min(i + 6, y2)), colour, 2)
            cv2.line(view, (x2, i), (x2, min(i + 6, y2)), colour, 2)
        cv2.putText(view, label[f['state']], (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2)
    return view


def zone_lines(monitor, out, ready_s=None, t=None):
    """Status lines [(text, BGR colour)] for the zone monitor."""
    lines = []
    visibility = out.get('visibility') or {}
    if out.get('view_changed'):
        lines.append(('VIEW CHANGED: the camera view moved or changed - obstruction alarms paused; check camera, '
                      're-learn', (0, 165, 255)))
    if visibility.get('visibility_compromised'):
        last = min(f['last_seen'] for f in visibility['fires'] if f['state'] == 'compromised')
        lines.append((f'PREVIOUSLY DETECTED FIRE - VISIBILITY COMPROMISED: its area is covered (last seen at '
                      f'{last:.1f} s; the fire may or may not still be burning)', (255, 0, 255)))
    elif visibility.get('compromised_unknown'):
        lines.append(('Previously detected fire not seen: its area cannot be judged yet (fire there while learning)',
                      (180, 180, 180)))
    if visibility.get('fires'):
        lines.append((f'fire area covered: {visibility["fire_area_covered_pct"]:.0f}% '
                      '(share of the last fire position covered; separate from the zone %)', (255, 180, 255)))
    if out['state'] == 'learning':
        if out.get('calibration_failed'):
            lines.append((f'CALIBRATION FAILED: the scene has not been still for {out.get("learning_for_s", 0):.0f} s '
                          '- clear the zone, stop movement, then press R to re-learn (still trying)', (0, 0, 255)))
        lines.append(('LEARNING EMPTY SCENE: scene moving, waiting for it to be still' if out.get('moving') else
                      f'LEARNING EMPTY SCENE: keep the zone clear ({out.get("learn_remaining_s", 0):.1f} s)',
                      (0, 255, 255)))
    if out['state'] == 'awaiting_confirmation':
        lines.append(('NOT ARMED: check the learned empty scene (second window). C = confirm, R = re-learn',
                      (0, 255, 255)))
    if out.get('healed_pct'):
        lines.append((f'ghost corrected: {out["healed_pct"]:.1f}% of the image re-learned (something present while '
                      'learning has left)', (0, 200, 255)))
    if out['state'] == 'ready' and ready_s is not None and t is not None and t - ready_s < 10 \
            and out['frame_pct'] > 2:
        lines.append(('CHECK: changes right after learning. If the view is EMPTY, press R to re-learn',
                      (0, 165, 255)))
    for z in monitor.config.zones:
        s = out['zones'][z.name]
        status = 'OBSTRUCTED - flame detection blind here' if s['active'] else 'clear'
        lines.append((f'{z.name}: now {s["blocked_pct"]:.0f}% | persistent {s["persistent_pct"]:.0f}% '
                      f'(>= {z.min_blocked_pct:.0f}% for alarm, {z.persist_s:.0f}s) | {status}',
                      (80, 80, 255) if s['active'] else (0, 255, 0)))
    return lines


def draw_lines(view, lines, top):
    """Text lines on a dark strip starting at y=top (an outline at another thickness misaligns in OpenCV 5)."""
    if not lines:
        return view
    w = view.shape[1]
    strip = view.copy()
    cv2.rectangle(strip, (0, top), (w, top + 10 + 22 * len(lines)), (0, 0, 0), -1)
    view = cv2.addWeighted(strip, .6, view, .4, 0)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(view, text, (10, top + 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 1, cv2.LINE_AA)
    return view
