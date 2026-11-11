"""Export frozen V10 (YOLOv9-S-TI-Lite, 1 class) WITHOUT the decode head, at 640 and 416.

Outputs per stride s in (8, 16, 32), H = W = size / s:
  cls_s{s}  [1, 1, H, W]   class logit (apply sigmoid on the A53)
  reg_s{s}  [1, 64, H, W]  box regression logits: 4 sides (l, t, r, b) x reg_max 16 bins (DFL)
Check: the raw outputs decoded in numpy (softmax over 16 bins, expectation, x stride, anchors at cell centre)
must match the original full-graph model's boxes and scores.
  python export_raw.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V10 = HERE.parent / 'v10_package'
sys.path[:0] = [str(V10 / 'workflow')]
from model import Detector  # noqa: E402

CKPT = HERE / 'src' / 'FROZEN_V10' / 'selected.pth'
STRIDES = (8, 16, 32)
REG_MAX = 16


class RawHead(torch.nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, images):
        feats = self.det.neck(self.det.backbone(images))
        outs = []
        for head, x in zip(self.det.bbox_head.heads, feats):
            # raw conv outputs, before TI's Anchor2Vec (rearrange + softmax): channel = side * 16 + bin
            outs += [head.class_conv(x), head.anchor_conv(x)]
        return tuple(outs)


def decode(outs, size):
    boxes, scores = [], []
    for k, s in enumerate(STRIDES):
        cls, reg = outs[2 * k], outs[2 * k + 1]
        _, _, h, w = cls.shape
        r = reg.reshape(1, 4, REG_MAX, h, w)
        p = np.exp(r - r.max(2, keepdims=True)); p /= p.sum(2, keepdims=True)
        dist = (p * np.arange(REG_MAX).reshape(1, 1, REG_MAX, 1, 1)).sum(2) * s          # [1,4,h,w]
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        ax, ay = xx * s + s // 2, yy * s + s // 2
        l, t, rr, b = dist[0]
        boxes.append(np.stack([ax - l, ay - t, ax + rr, ay + b], -1).reshape(-1, 4))
        scores.append((1 / (1 + np.exp(-cls[0, 0]))).reshape(-1))
    return np.concatenate(boxes), np.concatenate(scores)


def main():
    import onnx
    import onnxruntime as ort
    det = Detector(str(V10 / 'tensorlab'))
    det.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False)['model'], strict=True)
    det.eval()
    raw = RawHead(det).eval()
    names = [f'{kind}_s{s}' for s in STRIDES for kind in ('cls', 'reg')]
    report = {}
    torch.manual_seed(0)
    for size in (640, 416):
        x = torch.rand(1, 3, size, size)
        path = HERE / f'flame_v10_raw_{size}.onnx'
        torch.onnx.export(raw, x, str(path), opset_version=17, input_names=['images'], output_names=names,
                          do_constant_folding=True, dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)))
        sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        outs = sess.run(None, {'images': x.numpy()})
        with torch.no_grad():
            ref = det(x)[0].numpy()                                      # [N, 5] xyxy + prob (full graph)
            torch_raw = [t.numpy() for t in raw(x)]
        b, sc = decode(outs, size)
        report[size] = dict(
            file=path.name, outputs={n: list(o.shape) for n, o in zip(names, outs)},
            max_abs_box_error_px=float(np.abs(b - ref[:, :4]).max()), max_abs_score_error=float(np.abs(sc - ref[:, 4]).max()),
            onnx_vs_torch_raw_max_abs=float(max(np.abs(a - t).max() for a, t in zip(outs, torch_raw))),
            ops=sorted({n.op_type for n in onnx.load(str(path)).graph.node}))
        print(size, json.dumps(report[size], indent=1))
    (HERE / 'export_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
