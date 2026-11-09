# Flame detector for AM62A: raw-head export (V10)

Model: **YOLOv9-S-TI-Lite, 1 class (flame)**, frozen V10 weights (`selected.pth`, SHA256 `d046dab6…`). The same weights as the current `flame.onnx`. Only the end of the graph is different.

## 1. Export without the decode head

`flame_v10_raw_640.onnx` and `flame_v10_raw_416.onnx`:
- opset 17, fixed shape, batch 1, simplified with onnx-simplifier
- ops: Conv, Relu, MaxPool, Resize, Concat, Add, Slice
- no anchors, DFL, softmax, sigmoid or decode

| output | 640 input | 416 input | meaning |
|---|---|---|---|
| `cls_s8` | [1, 1, 80, 80] | [1, 1, 52, 52] | flame logit, stride 8 |
| `reg_s8` | [1, 64, 80, 80] | [1, 64, 52, 52] | box regression logits, stride 8 |
| `cls_s16` | [1, 1, 40, 40] | [1, 1, 26, 26] | stride 16 |
| `reg_s16` | [1, 64, 40, 40] | [1, 64, 26, 26] | stride 16 |
| `cls_s32` | [1, 1, 20, 20] | [1, 1, 13, 13] | stride 32 |
| `reg_s32` | [1, 64, 20, 20] | [1, 64, 13, 13] | stride 32 |

Input: `images` [1, 3, S, S] float32.
- **reg_max = 16**, strides **8 / 16 / 32**, one anchor point per cell. Total cells: 8400 at 640, 3549 at 416.
- `reg` channel layout: `channel = side * 16 + bin`, with sides in the order **left, top, right, bottom**.

**Decode on the A53**, for each stride `s` and cell `(i = row, j = col)`:
```
score = sigmoid(cls[0, 0, i, j])
for side k in 0..3:  d_k = s * sum_b softmax(reg[0, k*16 : k*16+16, i, j])[b] * b
ax = j*s + s//2 ;  ay = i*s + s//2          # cell centre (integer division)
box = (ax - d_left, ay - d_top, ax + d_right, ay + d_bottom)   # xyxy, input pixels
```

Then keep score >= threshold, run NMS (IoU 0.65, at most 100 boxes), and undo the letterbox (section 3).

**Verified (measured):** the raw outputs, decoded in numpy, match the original full graph within 0.006 px (box) and 1e-9 (score) at both sizes. ONNX Runtime and PyTorch agree within 1e-3. Details are in `export_report.json` and the script is `export_raw.py`.

**TI export path:** this is our own adapter around TI's YOLOv9 ti-lite layers, not TI's mmdet deploy script. That script adds TIDL meta-architecture files; we haven't used it yet. The backbone, neck and head convs are TI's.

## 2. 416x416

Same weights, same settings, only the input size changes. **Caveat:** the model was trained and evaluated at 640 only, so 416 accuracy is not measured yet. Expect lower small-flame recall at 416. For a fair comparison, run the same validation set at both sizes.

## 3. Preprocessing (must match for calibration)

1. Read the image as BGR (OpenCV), then convert it to **RGB**.
2. **Letterbox, no stretch:** `r = min(S/h, S/w)`, resize to `(round(w*r), round(h*r))` with **INTER_LINEAR**.
3. Place the resized image at the **top-left** of an S x S canvas filled with **114** (all channels). Pad on the right and bottom only.
4. Scale pixels to the **0–1** range (divide by 255). **No mean or std.** Layout: NCHW float32.
5. Box restore: `x / (round(w*r)/w)`, `y / (round(h*r)/h)`, then clip to the image.

## 4. Output and evaluation details

- **Current `flame.onnx`:** output `detections` [1, 8400, 5] = **xyxy in input pixels** + flame probability (sigmoid, no objectness), pre-NMS.
- **1 class:** flame.
- **Evaluation:**
  - Score floor 0.001, **NMS IoU 0.65**, at most 100 detections, COCO AP.
  - Precision, recall and false alarms are measured at **score 0.40**, with IoU 0.5 one-to-one matching.

## 5. Data and float baseline

- `calibration/`: **90 images, all from TRAINING data**, so validation and test stay clean. `calibration_list.json` lists each one:
  - 24 small, 18 medium and 14 large flames
  - 10 smoke-only and 14 no-fire images (FASDD-CV v2 train)
  - 10 own phone frames with small flames
- **Validation split:** FASDD-CV v2 val (15,884 images, public on Kaggle: yuulind/fasdd-cv-coco). The image list and the nuisance set (hn_val, 2,078 images: lamps, screens, sunsets, welding…) and the nuisance videos are on the lab PC. We can send them on request.
- **Float baseline, V10 at 640 (measured):**

| set | AP50 | precision @.40 | recall @.40 | other |
|---|---|---|---|---|
| FASDD val (15,884) | 74.47% | 81.13% | 58.85% | small-flame recall 48.79% |
| FASDD test (run once) | 89.05% | 90.34% | 79.15% | 764 FP boxes; no-flame images flagged 1.84% |
| nuisance set hn_val | - | - | - | 2.31% of images flagged at .40 |

## 6. Versions

- Model code: TI **edgeai-tensorlab**, commit `f272548ef0fe96046f1bf2c3833e040a1ecf8e44` (edgeai-mmdetection YOLOv9 ti-lite layers), plus our 1-class adapter.
- Training: PyTorch 2.7.1 (CUDA 12.8). This export: PyTorch 2.11 (CPU), onnx 1.23.2, onnxruntime 1.31.1, onnxsim.
- **TIDL / Edge AI SDK: none chosen yet.** Use the SDK that matches your SK-AM62A-LP board image. We haven't compiled for the board.

**Note:** a newer fine-tuned model (V12) may replace V10 around 7 Nov. `export_raw.py` re-exports any checkpoint the same way, with the same output names.
