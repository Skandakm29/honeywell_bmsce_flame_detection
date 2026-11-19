"""Phase A table. Reads phaseA_dets/*.json. Methods (all with the locked temporal rule, 10 fps):
  A full 640 @0.40 (current) | B full 640 @0.20 | C full @0.40 + zoom-on-candidates @0.40
Calibration is independent of the candle runs: tilt (and HFOV, setup 39 only) from the TAPE marks; contact height
from ONE mark (1 m) of ONE clip (m39_1to3_lighton); everything else is test."""
import os, sys, json, math
import numpy as np
import os as _os
from pathlib import Path as _P
_REPO = _P(__file__).resolve().parents[2]
H0 = _os.environ.get('W3_DATA', str(_REPO.parent)); R = str(_REPO)
sys.path[:0] = [R + '/src', R + '/experiments/temporal_final', R + '/experiments/yolov9_v10/workflow']
import temporal_final as TF
from geometry import Camera, intrinsics_from_hfov
SP = os.environ.get('W3_OUT', os.path.join(H0, 'w3_demo')); D = os.path.join(SP, 'phaseA_dets')
RULE = json.load(open(H0 + '/W3_starter_V12/rule_v12.json')); TF.BASE = 0.0
W, H, HGT = 1920, 1080, 1.5
TAPES = {'39': {1.0: (844, 949), 1.5: (853, 619), 2.0: (848, 392), 2.5: (858, 217), 3.0: (840, 87)},
         '44': {1.0: (830, 613), 1.5: (853, 291), 2.0: (862, 51), 0.5: None}}


def mkcam(t, hf, z=0.0):
    c = Camera((W, H), intrinsics_from_hfov(W, H, hf), pose={"height_m": HGT, "tilt_deg": t, "roll_deg": 0.0}); c.contact_height_m = z; return c


def dist_px(c, p, z=0.0):
    g = c.pixel_to_floor(np.array([p], float), z)[0]; return float(math.hypot(g[0], g[1]))


def fit(marks, hf=None):
    best = None
    for t in np.arange(25, 70, 0.05):
        for f in ([hf] if hf else np.arange(55, 75, 0.5)):
            c = mkcam(t, f); r = sum((dist_px(c, p) - d) ** 2 for d, p in marks.items() if p)
            if best is None or r < best[0]: best = (r, t, f)
    return best[1], best[2]


t39, hf = fit(TAPES['39']); t44, _ = fit(TAPES['44'], hf)
print(f"calibration from tapes: setup39 tilt {t39:.2f} hfov {hf:.1f} | setup44 tilt {t44:.2f} (hfov reused)")
CAM = {'39': mkcam(t39, hf), '44': mkcam(t44, hf)}
CLIPS = {  # name: (setup or None if camera moved / no tapes, marks list or None, has_flame)
    'm39_1to3_lighton': ('39', [1, 1.5, 2, 2.5, 3], 1), 'm39_1to3_lightoff': ('39', [1, 1.5, 2, 2.5, 3], 1),
    'm39_3to1_lightoff': ('39', [1, 1.5, 2, 2.5, 3], 1), 'm44_05to2_lighton': ('44', [0.5, 1, 1.5, 2], 1),
    'm44_05to2_lightoff': ('44', [0.5, 1, 1.5, 2], 1), 'u39_1to3_lighton': (None, None, 1), 'u39_1to3_lightoff': (None, None, 1),
    'u44_05to2_lighton': (None, None, 1), 'u44_05to2_lightoff': (None, None, 1),
    'nf_dim_room': ('39', None, 0), 'nf_light_room': ('39', None, 0), 'nf39_marking_lightoff': (None, None, 0),
    'nf44_empty_lighton': ('44', None, 0), 'nf44_empty_lightoff': ('44', None, 0), 'nf44_marking_lightoff': ('44', None, 0)}
METHODS = {'A full@0.40': lambda f: [b for b in f['full'] if b[4] >= .40],
           'B full@0.20': lambda f: [b for b in f['full'] if b[4] >= .20],
           'C full@0.40+zoom@0.40': lambda f: merge([b for b in f['full'] if b[4] >= .40], [b for b in f['zoom'] if b[4] >= .40]), 'D full@0.40+zoom@0.20': lambda f: merge([b for b in f['full'] if b[4] >= .40], [b for b in f['zoom'] if b[4] >= .20])}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def merge(full, zoom):
    out = list(full)
    for z in sorted(zoom, key=lambda b: -b[4]):
        if all(iou(z, o) < .3 for o in out): out.append(z)
    return out


def oracle(f):  # where the candle is: best box over everything at any score >= .10 (method independent)
    allb = [b for b in f['full'] + f['zoom'] if b[4] >= .10]
    return max(allb, key=lambda b: b[4]) if allb else None


def near(b, o): return o is not None and math.hypot((b[0] + b[2] - o[0] - o[2]) / 2, (b[1] + b[3] - o[1] - o[3]) / 2) < 60


# contact height from m39_1to3_lighton at the 1 m mark only (oracle boxes whose bottom is near the 1 m tape row)
cal = json.load(open(os.path.join(D, 'm39_1to3_lighton.json')))['frames']
bots = [((o[0] + o[2]) / 2, o[3]) for f in cal if (o := oracle(f)) and abs(o[3] - 949) < 120]
zs = np.linspace(0, 0.3, 301)
Z = float(zs[np.argmin([abs(np.median([dist_px(CAM['39'], p, z) for p in bots]) - 1.0) for z in zs])])
print(f"contact height (flame base above floor) from 1 m of m39_1to3_lighton: {Z*100:.1f} cm  ({len(bots)} frames)\n")

rows = []
for name, (setup, marks, flame) in CLIPS.items():
    p = os.path.join(D, name + '.json')
    if not os.path.exists(p): continue
    F = json.load(open(p))['frames']; times = [f['t'] for f in F]; orc = [oracle(f) for f in F]
    cam = CAM.get(setup)
    for m, sel in METHODS.items():
        sets = [sel(f) for f in F]
        per = TF.track([dict(t=t, dets=s) for t, s in zip(times, sets)], 640 / W, None)
        al = TF.alarm(times, TF.raw_flags(times, per, RULE['window_s'], RULE['min_hit_ratio'], RULE['min_mean_score']), RULE['alarm_on_after_s'])
        ms = np.mean([f['ms_full'] + (f['ms_zoom'] if m[0] in 'CD' else 0) for f in F])
        r = dict(clip=name, method=m, ms=round(float(ms), 1))
        if not flame:
            ev = sum(1 for i in range(len(al)) if al[i] and (i == 0 or not al[i - 1]))
            r.update(false_alarm_s=round(sum(al) * 0.1, 1), false_alarm_events=ev, boxes=sum(len(s) for s in sets), dur_s=round(times[-1], 1))
        else:
            seen = [i for i, o in enumerate(orc) if o is not None]
            first = seen[0] if seen else None
            fa = next((i for i in range(len(al)) if al[i]), None)
            r.update(det_pct=round(100 * np.mean([any(near(b, orc[i]) for b in sets[i]) for i in seen]), 1) if seen else None,
                     alarm_pct=round(100 * np.mean([al[i] for i in seen if times[i] >= times[first] + 3]), 1) if seen else None,
                     t_alarm_s=round(times[fa] - times[first], 1) if fa is not None and first is not None else None,
                     off_candle_boxes=sum(1 for i in range(len(F)) for b in sets[i] if not near(b, orc[i])))
            if marks and cam is not None:
                cam.contact_height_m = Z; pm = {}
                for i in seen:
                    o = orc[i]; d = dist_px(cam, ((o[0] + o[2]) / 2, o[3]), Z); k = min(marks, key=lambda x: abs(x - d))
                    if abs(k - d) > 0.25 * k: continue
                    q = pm.setdefault(k, dict(n=0, det=0, al=0, err=[]))
                    q['n'] += 1; hit = [b for b in sets[i] if near(b, o)]
                    q['det'] += bool(hit); q['al'] += bool(al[i])
                    if hit:
                        b = max(hit, key=lambda b: b[4]); q['err'].append((dist_px(cam, ((b[0] + b[2]) / 2, b[3]), Z) - k) / k)
                r['marks'] = {k: dict(frames=v['n'], det_pct=round(100 * v['det'] / v['n']), alarm_pct=round(100 * v['al'] / v['n']),
                                      med_err_pct=round(100 * float(np.median(v['err'])), 1) if v['err'] else None) for k, v in sorted(pm.items())}
        rows.append(r)
json.dump(rows, open(os.path.join(SP, 'phaseA_results_D.json'), 'w'), indent=1)
for r in rows: print(r)
