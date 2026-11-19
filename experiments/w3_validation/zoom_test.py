"""Zoom-crop test on fire_6: crop CxC around the brightest spot, resize to 640, run V12."""
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('W3_DATA', REPO.parent))          # folder holding W3_starter_V12/ and w3_demo/
MODEL = str(DATA / 'W3_starter_V12/models/flame_v12_full.onnx')
VIDEO = os.environ.get('W3_VIDEO', str(DATA / 'w3_demo/drive_videos/fire6.mp4'))
OUT = Path(os.environ.get('W3_OUT', DATA / 'w3_demo'))
import sys, cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, str(REPO / 'scripts')); from live_flame import prepare, decode
s = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
c = cv2.VideoCapture(VIDEO)
SEG = [(0, 9, 'no candle'), (11, 25, '0.5 m'), (26, 41, '1 m'), (42, 57, '1.5 m'), (58, 71, '2 m'), (72, 86, '2.5 m'), (88, 104, '3 m')]
CROPS = [1920, 960, 640, 427, 320]          # 1920 = normal full frame (zoom 1x at 640 input = 0.33 scale)
frames = {}
for a, b, lab in SEG:
    fr = []
    for t in np.arange(a + 0.5, b, 0.5):
        c.set(0, t * 1000); ok, f = c.read(); fr.append(f)
    frames[lab] = fr
print("crop px -> scale vs full-res:  " + "  ".join(f"{C}:{640/C:.2f}x" for C in CROPS))
print(f"{'position':10s}" + "".join(f"{('full' if C==1920 else str(C)):>14s}" for C in CROPS))
for lab, fr in frames.items():
    row = []
    for C in CROPS:
        hits = 0; scs = []
        for f in fr:
            if C == 1920:
                cr = f
            else:
                g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (9, 9), 0); _, _, _, (x, y) = cv2.minMaxLoc(g)
                x0 = min(max(0, x - C // 2), 1920 - C); y0 = min(max(0, y - C // 2), 1080 - C)
                cr = np.ascontiguousarray(f[y0:y0 + C, x0:x0 + C])
            xx, r = prepare(cr, 640)
            bx = decode(s.run(None, {s.get_inputs()[0].name: xx})[0], cr, r, 0.05, 'yolov9')
            m = max([z[4] for z in bx], default=0.0); scs.append(m); hits += m >= 0.40
        row.append(f"{hits:>3d}/{len(fr):<3d}({np.median(scs):.2f})")
    print(f"{lab:10s}" + "".join(f"{v:>14s}" for v in row))
