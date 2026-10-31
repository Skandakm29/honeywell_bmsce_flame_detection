# YOLOv9-S-TI-Lite flame experiment

Training completed on the lab RTX A5000; final validation/test metrics and export
are documented in [RESULTS_YOLOV9.md](RESULTS_YOLOV9.md). This page describes the
alternative Colab workflow. Do not interpret local engineering fixtures as benchmarks.
Local evidence is in `results/yolov9_engineering_checks.json`: eight focused tests,
a real-image 640-pixel CPU train-step and ONNX check, and simulated interruption
recovery with bitwise-identical model/EMA weights on the same CPU.

## Run in Colab

Open `experiments/yolov9_flame/YOLOv9_S_TI_Lite_Flame.ipynb` in Google Colab,
select a GPU runtime, and run the cells in order. Authorize the project Google
Drive account when prompted. The notebook embeds the reviewed experiment code
and six pinned TI source modules; it does not require a GitHub push. Keep the
training cell running. After a disconnect rerun setup, data, preflight and train;
matching checkpoints resume rather than starting over.
The latest Colab kernel may use Python 3.13; setup uses uv to create a separate
Python 3.11 environment for the pinned Torch 2.5.1 / CUDA 12.4 packages.

Outputs are under
`MyDrive/HoneywellTraining/experiments/yolov9_s_ti_lite_v1/`:

- `source_bundle.zip`: exact executable source and vendor modules.
- `manifests/`: frozen train, monitor validation, full validation, and test lists;
  source annotation hashes, automated decode/box checks, quarantine and duplicate report.
- `preflight/`: real-resolution train-step/export check and chosen microbatch.
- `training/pilot_lr_*/`: short learning-rate screening, selected on validation only.
- `training/full/`: model/optimizer/EMA/scaler/RNG checkpoints, recipe and history.
- `validation/`: full-validation COCO AP/AR, precision/recall/F1, predictions,
  generated report, and checked `export/flame.onnx` with input/output contract.

The notebook does not evaluate the test split automatically. The team has
already examined this Kaggle test; it is not a new independent benchmark.

## Model and source

TI `edgeai-tensorlab` revision `f272548ef0fe96046f1bf2c3833e040a1ecf8e44`.
The architecture follows `configs_edgeailite/yolov9/yolov9_s_coco_lite.py`:
stem 32/64, feature channels 128/192/256, max-pooling with kernel 3, main and
training-only auxiliary detection heads, TI Lite ReLU and split-pooling surgery.

`model.py` loads the official layer/backbone/neck/auxiliary/loss modules under a
private namespace, avoiding unrelated MMCV CUDA extensions. The auxiliary head's
unused `ConvModule` import is removed in memory; the source files are unchanged.
`vendor.json` checks each selected source file after normalizing CRLF to LF.
The small main-head container follows TI's published `YOLOV9Head.heads` layout.

Published COCO initialization is retrieved from TI's **r10.1** model catalog:

https://github.com/TexasInstruments/edgeai-modelzoo/blob/r10.1/models/vision/detection/coco/edgeai-mmdet/yolov9_s_coco_lite_640x640_20250219_checkpoint.pth.link

Verified SHA256:
`7d17806cda7bb4e1c94b6838e52d66c5fa93a421d94d2ada42f23da800b84559`.
The adapter rejects any other pretrained file. Its 1,758 matching tensors load;
only the six main/auxiliary final class convolutions (12 tensors) are initialized
for one class, with a 0.01 initial class probability prior.

Two explicit loss corrections are documented and tested:

1. Clamp the positive normalization denominator to at least one; batches without
   matched flames retain classification loss and zero localization loss.
2. Compute matched CIoU directly in paired singleton dimensions rather than
   constructing an NxN matrix and taking its diagonal. Values and gradients are
   checked against the vendor calculation. This removes quadratic temporary memory.

Preprocessing: aspect-preserving resize, top-left content and bottom/right padding
114; **RGB float32 0..1**. Restore coordinates using actual rounded horizontal and
vertical resize scales. This differs from the existing YOLOX BGR/raw-value contract.
The ONNX model outputs input-pixel xyxy boxes and flame probabilities before NMS;
it excludes the auxiliary branch and does not multiply by an objectness score.

## Recipe and selection

This is an experiment, not a promise of globally optimal hyperparameters.
`config.json` specifies the exact initial recipe. Three LR candidates
`1e-4 / 3e-4 / 1e-3` each run three epochs on the same seeded 8,192-image training
subset and monitor validation. Best monitor COCO AP50:95 selects the LR for a
fresh full-data run, with no pilot weight carryover and no test selection.

Full training: 640 input, up to 40 epochs, AdamW, weight decay 5e-4 excluding
biases/norms, effective batch 32, two-epoch warmup, cosine decay to 5% LR,
gradient clip 10, EMA, mixed precision on CUDA. Microbatch is chosen by a real
training-step memory check; accumulation maintains effective batch 32. A changed
batch/recipe requires a new run directory. A fresh full-resolution GPU preflight
and dependency verification are still required in Colab.

The AMP initial loss scale is 128. The default 65,536 overflowed during the T4
preflight; finite-loss and gradient checks remain enabled. The scaler can reduce
its scale and skip overflowing updates during training without updating EMA.
Dataset folder copies use 16 readers to reduce remote-cache per-file latency;
the automatic decode, annotation and hash checks still run after copying.

Augmentation: seeded mosaic probability 0.5, horizontal flip, moderate HSV.
Mosaic closes for the last 10 epochs. Training can stop after epoch 30 if
monitor AP has not improved for 10 epochs. Checkpoint selection always uses
EMA monitor AP50:95. The source SHA, manifests, initial weights and config are
checked on resume. Checkpoints at optimizer boundaries include optimizer,
EMA, AMP scaler and RNG state; the previous checkpoint is retained for recovery.
Resume means continuation of the same schedule, not a new warmup.

The recipes differ from the teammate's YOLOX-S recipe. Comparison is a practical
model/recipe comparison and cannot isolate architecture alone.

## Data and metrics

No manual annotation work is required. Match the supplied Kaggle annotation
files using `dataset_source.json`; a mismatched dataset version stops execution.
Preserve supplied splits and smoke-only/background negatives. Automated image
decoding, size and box checks quarantine invalid records. The <=1-pixel box
rounding allowance matches the existing audit. Report exact duplicates without
silently modifying the teammate comparison splits. Source/video independence
is not established by these checks. Inspect the preparation summary before
interpreting comparisons; use a new run if source or split policies change.

Report standard COCO bbox AP50:95, AP50, AP75, AP by COCO size, and AR; absent
size groups are null. Prediction score floor .001, NMS IoU .65, maxDets 100.
Report precision/recall/F1 and TP/FP/FN at fixed confidence .40, matching IoU .50,
with score-ordered one-to-one matching. False-positive boxes are not alarm events.
No classification accuracy or invented production metric is reported.

ONNX checker and CPU ONNX Runtime parity must pass. Boxes have an absolute
tolerance of .05 input pixel; probabilities have atol 1e-5 / rtol .001. Separate
units avoid meaningless relative tolerances near zero-valued box coordinates.
Maximum differences and tolerances are written to `export.json`.
AM62A compilation, latency, memory, INT8 quality and 1 TOPS compliance require
separate measurements. The inference graph, not training auxiliary parameters,
is the relevant deployed model.

## Local validation commands

With dependencies installed, from the repository root:

```powershell
python experiments/yolov9_flame/test_workflow.py --tensorlab .tools/edgeai-tensorlab --checkpoint .tools/yolov9_s_ti_lite_coco.pth --export
python experiments/yolov9_flame/build_notebook.py --tensorlab .tools/edgeai-tensorlab
```

The tests cover resize/coordinate restoration, RGB input, empty labels,
matching, COCO perfect/empty predictions, mixed/negative losses and gradients,
paired-CIoU equivalence, pretrained loading, EMA copy and ONNX parity. Use a
writable temporary directory on restricted Windows environments.
