# V11 protocol (fixed before training; never changed after seeing results)

**Model:** YOLOv9-S-TI-Lite, 640x640, 1 class.
- No architecture change.
- Export is a fixed-shape ONNX (TIDL-compatible path).
- INT8 and AM62A simulation come later.

**Data**
- Train is manifests_v10 train (same records and repeats), with these label fixes (`labelfix.py`):
  - V10 box score >= .60 and CLIP P(fire) >= .70: the box is added.
  - CLIP P(fire) between .30 and the add threshold: the image is dropped.
  - CLIP P(fire) < .30: the image is left as it is.
  - Gate A: if fewer than 80% of 72 random added boxes are real flames, all add_box images are dropped instead.
- Dense own frames (step 1b/1c): the 4 train small-flame clips (diya_01, farflame_01/02, flame_occlusion) are sampled every 0.33 s. V10 or the colour rule proposes each box; every box is reviewed; only `ok` frames are used, x3. These add volume from the same 4 scenes, not variety. Eval clips and ONFIRE stay held out.
- WhatsApp clips (owner decision, 6 Nov): all 6 scored clips go to TRAINING; they are no longer a test set. Flame clips: boxes proposed from V10 seeds plus tracking, every box reviewed. Non-fire clips: 1 frame/s as negatives. All x3.
- Optional extra small-flame frames from training locations only, repeated x3.
- val, validation_full, hn_val and test are byte-identical copies of V10's.

**Training (`config_v11.json`)**
- COCO TI weights, 40 epochs, AdamW LR 1e-4, 2 warm-up epochs, cosine decay to 0.05.
- Batch 16 x 2, EMA, AMP.
- Mosaic 0.5, switched off for the last 10 epochs.
- Augmentation added on top of V10's: zoom-out (p .5, scale .25-1, no box under 4 px), small-flame paste (p .25, 6-32 px, crops from the training set only), and camera degradations.
- No early stopping; the EMA checkpoint of every epoch is saved.

**Selection (`select_v11.py`)**
- Candidates: epochs 21-40.
- Reference: frozen V10, measured with the same code.
- Eligible if all of these hold:
  - AP50 >= V10 - 1 pt
  - recall@.40 >= V10 - 1 pt
  - real-nuisance score <= 1.5 x V10
- Best: highest small-flame recall (fixed 564-image validation slice). Ties within 0.5 pt go to higher recall, then higher AP50:95.
- V11 is adopted only if its small-flame recall is >= V10 + 2 pts. Otherwise V10 stays.
- The FASDD test set runs once, after selection is final (owner decision 6 Nov: end-to-end run). Nothing is changed after seeing it.
