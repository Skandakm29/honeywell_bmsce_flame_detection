# V12 protocol (fixed before training; never changed after seeing results)

**Model:** YOLOv9-S-TI-Lite, 640x640, 1 class. No architecture change. Fixed-shape ONNX export. INT8 / AM62A later.

**Why V12 (from V11, measured):** training from COCO with strong augmentation gave +0.8 pt small-flame recall and 4.3x V10's false alarms. V12 starts from frozen V10 instead and tests one change at a time.

**Data**
- Training: `manifests_v11/train.json` (V10 train + V11 label fixes + dense own frames + 6 WhatsApp clips). Built and leak-checked by V11 step 2.
- Evaluation: val, validation_full, hn_val, test are byte-identical to V10 (preflight checks this).
- Owner recordings (step 4), split by place:
  - every place except corridor goes to training (x5).
  - corridor is the distance test only.
  - Every box is reviewed. Boxes come from flame positions marked by Claude (AI) in `recordings_seeds.json`, followed frame by frame; clips without seeds use V10 or a warm, flickering region.
  - Corridor clips have several diyas: one box covers the cluster, and a detection counts as a hit if it overlaps the box or its centre lies inside it.

**Step 1: size-aware threshold (no training, validation only)**
- Rule: a box counts if its score >= .40, or if its longest side at the 640 input is < S px and its score >= t.
- Grid: S in {12, 16, 24, 32}, t in {.15, .20, .25, .30, .35}.
- Pick: the highest small-flame recall with precision >= base - 1 pt and nuisance <= 1.5 x V10. Otherwise keep .40.

**Step 2: three fine-tunes from frozen V10**
- Each uses V10's own fine-tune settings: AdamW, LR 3e-5, 300 warm-up updates, cosine decay to 0.1, batch 16 x 2, EMA, AMP. 8 epochs, mosaic .5 off for the last 2.
- Variants:
  - `a_data`: V10's pipeline, nothing new.
  - `b_zoom`: mild zoom-out (p .3, scale .5-1, no box under 6 px) plus camera degradations (resample, blur and noise at p .1, JPEG at p .15).
  - `c_zoom_paste`: b plus small-flame paste (p .15, 1-2 flames at 8-32 px, train-only crops).

**Selection (validation only; frozen V10 measured with the same code)**
- Candidates: every epoch of the 3 variants (24 checkpoints).
- Eligible if all hold:
  - AP50 >= V10 - 1 pt
  - recall@.40 >= V10 - 1 pt
  - real-nuisance score <= 1.5 x V10
- Best: highest small-flame recall (564-image slice). Ties within .5 pt go to higher recall, then higher AP50:95.
- Adopted over V10 only if eligible and small-flame recall >= V10 + 1 pt.
- The size-aware threshold is chosen again for the best checkpoint, with the same grid and limits.
- Start model for the recordings run: the best eligible checkpoint, or V10 if none is eligible.

**Step 4: recordings fine-tune**
- From the start model, same augmentation as its variant. LR 2e-5, 4 epochs, mosaic off for the last epoch.
- Selection uses the same rule over the start model plus its 4 epochs, so the recordings can never force a worse model.

**Step 5: test, exactly once**
- Runs only after every selection is final. Nothing is changed after seeing it.
- FASDD test at .40 and under the chosen threshold rule.
- Distance test (corridor): the share of frames with the flame detected (IoU >= .1 with the reviewed box), and the share of no-flame frames with a false alarm. Both for V10 and the final model.

**Leakage:** nothing from validation, hn_val, test, ONFIRE, held-out own clips or corridor is ever trained on. The code checks this.
