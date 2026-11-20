"""Real-camera test runner for the Week 4 zone-obstruction monitor (src/obstruction). No YOLO, no ONNX.

Feeds undrawn BGR frames and timestamps (seconds) to SceneMonitor and shows zones, blocked cells and alarms.
Image-space percentages only: no floor area in m2 is computed (Week 3 calibration does not exist yet).

Keys: Q/Esc quit | S screenshot | R re-learn the empty scene (after moving the camera)
      A..G write a test-case marker into frames.csv (A empty, B walking, C standing, D object placed,
      E object removed, F lighting, G two objects) | M generic marker
Logs go to results/obstruction_live/<timestamp>/ (frames.csv, events.csv, session.json, blind_map.json/.png);
never overwritten.
A camera or monitor failure stops the run and is recorded in session.json; it is never logged as a clear scene.
"""
import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from obstruction import ReferenceDetector, SceneMonitor, load_zones  # noqa: E402
from obstruction.overlay import draw_lines, draw_zones, zone_lines  # noqa: E402

MARKERS = {ord(k): name for k, name in zip('abcdefg', ['A_empty', 'B_walking', 'C_standing', 'D_object_placed',
                                                       'E_object_removed', 'F_lighting', 'G_two_objects'])}
MARKERS[ord('m')] = 'marker'
EVENT_COLUMNS = ['zone', 'start_s', 'end_s', 'duration_s', 'peak_pct', 'mean_pct', 'start_local', 'end_local']


def draw(frame, monitor, out, loop_fps, monitor_ms):
    """Overlay on a copy: zones and blocked cells (shared overlay) plus a status strip at the top."""
    view = draw_zones(frame, monitor, out)
    lines = [(f'{loop_fps:.1f} FPS | monitor {monitor_ms:.1f} ms | {out["state"]} | '
              f'foreground {out["foreground_pct"]:.1f}% | gain {out["gain"]:.2f}', (0, 255, 255))]
    lines += zone_lines(monitor, out, out.get('ready_s'), out.get('t'))
    lines.append(('image-space % only; no floor m2 | Q quit S shot R relearn A-G markers', (200, 200, 200)))
    return draw_lines(view, lines, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='0', help='Camera index (default 0) or a video file')
    parser.add_argument('--zones', type=Path, default=ROOT / 'configs/zones/webcam_test.json')
    parser.add_argument('--width', type=int, default=640, help='Requested camera width')
    parser.add_argument('--height', type=int, default=480, help='Requested camera height')
    parser.add_argument('--learn-s', type=float, default=3., help='Seconds of empty scene learned at start')
    parser.add_argument('--adapt-s', type=float, default=600., help='Reference adaptation time constant, s')
    parser.add_argument('--grey-threshold', type=float, default=25.)
    parser.add_argument('--edge-threshold', type=float, default=40.)
    parser.add_argument('--gap-s', type=float, default=1.5, help='Detection gap a blocked cell survives, s')
    parser.add_argument('--min-cell-fraction', type=float, default=.5)
    parser.add_argument('--duration', type=float, help='Stop after this many seconds')
    parser.add_argument('--headless', action='store_true', help='No window (for automated checks)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    config = load_zones(args.zones)
    detector = ReferenceDetector(learn_s=args.learn_s, adapt_s=args.adapt_s, grey_threshold=args.grey_threshold,
                                 edge_threshold=args.edge_threshold)
    monitor = SceneMonitor(config, detector, gap_s=args.gap_s, min_cell_fraction=args.min_cell_fraction)
    is_video = not args.source.isdecimal()
    capture = cv2.VideoCapture(args.source if is_video else int(args.source))
    if not capture.isOpened():
        raise SystemExit(f'ERROR: cannot open source {args.source}. Close other camera apps or try --source 1.')
    if not is_video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    video_fps = capture.get(cv2.CAP_PROP_FPS)

    output = args.output or ROOT / 'results/obstruction_live' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'ERROR: {output} already has files; choose a new --output')
    output.mkdir(parents=True, exist_ok=True)
    zones = [z.name for z in config.zones]
    session = dict(
        purpose='Week 4 zone-obstruction monitor, real-camera test (no YOLO, no flame pipeline)',
        source=args.source, source_kind='video' if is_video else 'camera',
        requested_resolution=None if is_video else [args.width, args.height], reported_fps=video_fps,
        zones_file=str(args.zones.resolve()), grid=dict(cols=config.grid[0], rows=config.grid[1]),
        zones=[dict(name=z.name, polygon=z.polygon, min_blocked_pct=z.min_blocked_pct, persist_s=z.persist_s,
                    clear_s=z.clear_s) for z in config.zones],
        ignore=config.ignore,
        detector=dict(type='ReferenceDetector', analysis_width=detector.width, learn_s=detector.learn_s,
                      adapt_s=detector.adapt_s, grey_threshold=detector.grey_threshold,
                      edge_threshold=detector.edge_threshold, min_blob_fraction=detector.min_blob_fraction),
        monitor=dict(gap_s=args.gap_s, min_cell_fraction=args.min_cell_fraction),
        python=sys.version.split()[0], opencv=cv2.__version__, numpy=np.__version__, platform=platform.platform(),
        started_local=datetime.now().isoformat(), status='running',
        note='Image-space percentages only. No floor area (m2): Week 3 calibration is not available. '
             'Thresholds are untuned starting values; synthetic tests do not establish real-world accuracy.')
    write_session = lambda: (output / 'session.json').write_text(json.dumps(session, indent=2), encoding='utf-8')
    write_session()

    window = 'Week 4 zone obstruction test - Q quits'
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    frame_columns = ['time_local', 't_s', 'frame', 'state', 'monitor_ms', 'loop_fps', 'foreground_pct', 'gain',
                     'blobs', 'frame_blocked_pct', 'marker']
    for name in zones:
        frame_columns += [f'{name}_blocked_pct', f'{name}_persistent_pct', f'{name}_alarm']
    latencies, frames, marker, error, ready_s, last_frame = [], 0, '', None, None, None
    started, previous, loop_fps, t = time.monotonic(), None, 0., 0.
    start_wall = datetime.now()
    local = lambda seconds: (start_wall + (datetime.fromtimestamp(seconds) - datetime.fromtimestamp(0))).isoformat()
    with (output / 'frames.csv').open('w', newline='', encoding='utf-8') as frames_file, \
            (output / 'events.csv').open('w', newline='', encoding='utf-8') as events_file:
        frame_writer, event_writer = csv.writer(frames_file), csv.writer(events_file)
        frame_writer.writerow(frame_columns)
        event_writer.writerow(EVENT_COLUMNS)

        def log_events(events):
            for e in events:
                event_writer.writerow([e['zone'], round(e['start_s'], 3), round(e['end_s'], 3),
                                       round(e['end_s'] - e['start_s'], 3), round(e['peak_pct'], 2),
                                       round(e['mean_pct'], 2), local(e['start_s']), local(e['end_s'])])
            events_file.flush()

        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    if is_video:
                        break                    # end of file is a normal end, not a camera fault
                    raise RuntimeError('camera stopped returning frames')
                now = time.monotonic()
                t = frames / video_fps if is_video and video_fps > 0 else now - started
                if previous is not None and now > previous:
                    loop_fps = .9 * loop_fps + .1 / (now - previous) if loop_fps else 1 / (now - previous)
                previous = now
                if frames == 0:
                    session['actual_resolution'] = [frame.shape[1], frame.shape[0]]
                    write_session()
                checked = time.perf_counter()
                out = monitor.update(frame, t)
                last_frame = frame
                monitor_ms = (time.perf_counter() - checked) * 1000
                if out['state'] == 'ready' and ready_s is None:
                    ready_s = t
                    print(f'learned the empty scene at t={t:.1f}s; zone should read about 0% if the view is empty')
                elif out['state'] != 'ready':
                    ready_s = None
                out.update(t=t, ready_s=ready_s)
                latencies.append(monitor_ms)
                log_events(out['events'])
                row = [datetime.now().isoformat(), round(t, 3), frames, out['state'], round(monitor_ms, 3),
                       round(loop_fps, 2), round(out['foreground_pct'], 3), round(out['gain'], 4),
                       len(out['blobs']), round(out['frame_pct'], 3), marker]
                for name in zones:
                    s = out['zones'][name]
                    row += [round(s['blocked_pct'], 3), round(s['persistent_pct'], 3), int(s['active'])]
                frame_writer.writerow(row)
                frames_file.flush()
                marker, frames = '', frames + 1
                if args.duration and t >= args.duration:
                    break
                if args.headless:
                    continue
                cv2.imshow(window, draw(frame, monitor, out, loop_fps, monitor_ms))
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('s'):
                    path = output / f'capture_{frames}.jpg'
                    cv2.imwrite(str(path), draw(frame, monitor, out, loop_fps, monitor_ms))
                    print('saved', path)
                elif key == ord('r'):
                    log_events(monitor.relearn(t))
                    marker = 'relearn'
                    print('re-learning the empty scene: keep the zone clear')
                elif key in MARKERS:
                    marker = MARKERS[key]
                    print(f'marker {marker} at t={t:.1f}s')
        except Exception as exc:                 # recorded, then re-raised: a failure is never a clear scene
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            log_events(monitor.close(t))
            if monitor.blind_map is not None:
                monitor.blind_map.save(output, last_frame)   # blind_map.json + blind_map.png
                session['blind_map'] = monitor.blind_map.summary()
            capture.release()
            if not args.headless:
                cv2.destroyAllWindows()
            lat = np.asarray(latencies)
            session.update(
                status='error' if error else 'finished', error=error, ended_local=datetime.now().isoformat(),
                frames=frames, duration_s=round(t, 3),
                measured_loop_fps=round(frames / max(t, 1e-9), 2) if frames > 1 else None,
                monitor_ms=dict(median=round(float(np.median(lat)), 3), p95=round(float(np.percentile(lat, 95)), 3),
                                max=round(float(lat.max()), 3)) if len(lat) else None)
            write_session()
            print(f'Logs: {output}')


if __name__ == '__main__':
    main()
