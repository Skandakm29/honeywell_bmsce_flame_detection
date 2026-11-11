# Week 4 protocol: obstruction as a second class (fixed before training; never changed after seeing results)

**Decision (mentor, week 4):** detect camera obstruction by adding a class `obstruction` to the **same** YOLOv9-S-TI-Lite flame model (option A in `docs/WEEK2_SUMMARY.md` section 12). One model, one pass per frame, on one processor.

## Model
- **Start:** final V12 (`select_v12_rec/best.pth`, SHA256 `4395912b…`). V10 can be used instead by setting `W4_START_CHECKPOINT` and `W4_START_SHA256`.
- **Change:** only the last 1×1 classifier conv of the 6 class branches grows from 1 to 2 outputs (12 tensors).
  - Flame channel = the trained weights, copied exactly.
  - Obstruction channel = weight 0, bias for a 1% prior.
  - Before training, the grown model gives **bit-identical** flame boxes and scores (checked in `tests/test_obstruction_class.py`).
- **Output:** `[1, 8400, 6]` = x1, y1, x2, y2, flame probability, obstruction probability. NMS per class.

## Training (V12's recipe, `config_w4.json`)
- V10's training loop unchanged (`train_v10.train`): AdamW, EMA, AMP, resume, every epoch saved and measured.
- Data: `manifests_v12` training set (what V12 was trained on), V12 `b_zoom` augmentation.
- **Changes from V12:** LR 3e-5 → 5e-5 and 8 → 10 epochs, because the new classifier starts from zero.
- **Obstructions:** drawn on the fly on 20% of loaded images (before mosaic). Each one is made by `occluders.py`:
  - **Shapes:** entering from an edge or corner (60%), inside the view (25%), or covering the whole view (15%).
  - **Kinds:** cardboard, steel or metal sheet, plain board, cloth, skin (hand), dark lens cover, or a heavily defocused other photo.
  - **Near-camera cues:** soft defocused edge, blurred texture, lighting falloff, sensor noise.
  - **Labels:**
    - The box is the hidden region's bounding box.
    - Only obstructions covering at least 3% of the image are labelled.
    - A flame that is at least 50% hidden loses its flame label.

## Validation (monitor set; FASDD test never read)
- The 2,000 monitor images (unchanged, flame only).
- Plus **1,000 fixed synthetic obstructions** on `validation_full` images outside the monitor set, seed 4242. Their donors come from the same pool, so nothing touches train, hn_val or test.
- **Flame metrics** use V10's definitions on the clean images only, so they compare directly with V10 and V12.
- **Obstruction metrics:**
  - AP50 and AP50:95, and precision and recall at 0.40, on the obstructed images.
  - Image detection rate.
  - False-obstruction image rate at 0.40, on clean images and on hn_val.
  - Blind-share error: |union area of obstruction boxes ≥ 0.40 − true hidden share|.

## Selection (`select_w4.py`)
- **Reference:** the grown start model (epoch 0), measured by the same code.
- **Eligible if ALL hold:**
  - flame AP50 ≥ reference − 1 pt
  - flame recall at 0.40 ≥ reference − 1 pt
  - flame false-positive image rate on hn_val ≤ 1.5 × reference
  - obstruction false-alarm image rate ≤ 2% on clean images **and** ≤ 2% on hn_val
- **Best:** highest obstruction AP50. Ties within 0.5 pt go to higher flame AP50.
- **No eligible epoch:** nothing is adopted and the flame model stays as it was.

## Gates
- **Gate A (before training):** a person looks at `manifests_w4/preview_val.jpg` and writes `status_w4/gate_a.txt`.
- **Real-camera test (after selection, reported once):** see `docs/WEEK4_OBSTRUCTION.md`. Synthetic numbers alone do not show real-camera performance.

## Known limits
- A single frame has no memory of the clear view. The class learns what a near-camera blocking object looks like, so:
  - a far-away object that hides part of the scene but looks like normal scene content may be missed
  - a large plain surface that is part of the normal view (a close wall) may be flagged
  The real-camera test measures both.
- Blind share is box-based, so it over-estimates for diagonal or round obstructions.
- Board: the raw-head export (`experiments/edge_export/export_raw.py`) is 1-class only and must be re-run for 2 classes (`cls_s*` becomes [1, 2, H, W]). AM62A speed and int8 accuracy are not measured.
