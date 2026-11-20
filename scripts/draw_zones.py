"""Draw monitored zones (and ignore areas) on a camera frame, video frame or image, and save the zone JSON.

  python scripts/draw_zones.py --source 0 --output configs/zones/my_site.json
  python scripts/draw_zones.py --source test_clips/clip.mp4 --output configs/zones/clip.json

Mouse: left click adds a point; right click removes the last point.
Keys:  Enter or N  close the current polygon (at least 3 points) as a zone
       I           close the current polygon as an IGNORE area (doors, screens, windows)
       U           undo the last closed polygon
       S           save and quit          Q / Esc  quit without saving
Zones are named zone1, zone2, ... and get the thresholds given on the command line; edit the JSON to rename
them or to change thresholds per zone. Coordinates are saved normalised (0..1), so any resolution works.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from obstruction import parse_zones  # noqa: E402


def build_config(zones, ignores, width, height, min_blocked_pct, persist_s, clear_s, grid=(32, 24)):
    """Pixel polygons -> validated zone config dict with normalised coordinates."""
    norm = lambda polygon: [[round(x / width, 4), round(y / height, 4)] for x, y in polygon]
    data = {'grid': {'cols': grid[0], 'rows': grid[1]},
            'zones': [{'name': f'zone{i + 1}', 'polygon': norm(p), 'min_blocked_pct': min_blocked_pct,
                       'persist_s': persist_s, 'clear_s': clear_s} for i, p in enumerate(zones)],
            'ignore': [norm(p) for p in ignores]}
    parse_zones(data)                            # same validation the monitor uses
    return data


def grab(source):
    if source.isdecimal() or Path(source).suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
        capture = cv2.VideoCapture(int(source) if source.isdecimal() else source)
        frame = None
        for _ in range(15 if source.isdecimal() else 1):   # let a webcam settle its exposure
            ok, frame = capture.read()
            if not ok:
                break
        capture.release()
    else:
        frame = cv2.imread(source)
    if frame is None:
        raise SystemExit(f'ERROR: no frame from {source}')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='0', help='Camera index, video file or image')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-blocked-pct', type=float, default=30.)
    parser.add_argument('--persist-s', type=float, default=10.)
    parser.add_argument('--clear-s', type=float, default=5.)
    parser.add_argument('--force', action='store_true', help='Overwrite an existing output file')
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f'ERROR: {args.output} exists; use --force or another name')
    frame = grab(args.source)
    h, w = frame.shape[:2]
    zones, ignores, current, closed = [], [], [], []   # closed: order of ('zone'|'ignore') for undo

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    window = 'Draw zones - click points, Enter=zone, I=ignore, U=undo, S=save, Q=quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        view = frame.copy()
        for i, p in enumerate(zones):
            cv2.polylines(view, [np.int32(p)], True, (255, 255, 0), 2)
            cv2.putText(view, f'zone{i + 1}', (p[0][0] + 4, p[0][1] + 18), cv2.FONT_HERSHEY_SIMPLEX, .55,
                        (255, 255, 0), 1, cv2.LINE_AA)
        for p in ignores:
            cv2.polylines(view, [np.int32(p)], True, (160, 160, 160), 2)
        if current:
            cv2.polylines(view, [np.int32(current)], False, (0, 255, 255), 1)
            for point in current:
                cv2.circle(view, point, 3, (0, 255, 255), -1)
        cv2.putText(view, f'zones {len(zones)} | ignore {len(ignores)} | points {len(current)} | '
                    'Enter=zone I=ignore U=undo S=save Q=quit', (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, view)
        key = cv2.waitKey(30) & 0xff
        if key in (13, 10, ord('n'), ord('i')) and len(current) >= 3:
            (ignores if key == ord('i') else zones).append(list(current))
            closed.append('ignore' if key == ord('i') else 'zone')
            current.clear()
        elif key == ord('u') and closed:
            (ignores if closed.pop() == 'ignore' else zones).pop()
        elif key == ord('s'):
            if not zones:
                print('Draw at least one zone before saving.')
                continue
            data = build_config(zones, ignores, w, h, args.min_blocked_pct, args.persist_s, args.clear_s)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, indent=2), encoding='utf-8')
            print(f'Saved {args.output} ({len(zones)} zones, {len(ignores)} ignore areas)')
            break
        elif key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            print('Quit without saving.')
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
