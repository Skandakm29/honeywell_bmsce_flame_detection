# Review: full negative-pool CLIP audit

Reviewed 1 November 2026. Source: teammate's nine-page `CLIP_Hard_Negatives_Full_Dataset_Report.pdf`, SHA256 `e50d17850cec8ff8d50399be168200ce7334503f6a58cb05ab7b6e95edfcfba1`.

## Overall judgement

This is useful, substantially stronger **coverage evidence** than the initial 500-image pilot. It includes reviewed verdicts, image IDs, splits, descriptions, possible cases and examples. Welding is now supported by actual dataset examples. It is not a detector-performance study, exhaustive visual census or proof that the models reject these conditions.

The report says it embedded the entire **39,199-image no-fire/no-smoke-annotation pool**, not every FASDD image. The full Kaggle copy contains 95,314 images, including 23,414 smoke-only images outside that searched pool. Top50 candidates per condition were visually reviewed, extended to top100 for heaters/screens; images below those ranks were not visually checked. The PDF does not include the new full-run notebook/logs, embedding cache or complete verification/ranking CSVs, so the claimed full embedding run and exact scores were not independently reproduced here.

## What was independently checked

- Read all nine pages and rendered the report for visual review.
- Extracted all **106 reported confirmed rows** from the detailed tables. The list contains 106 distinct split/filename records and reproduces the reported split totals: **54 train, 18 validation, 34 test**.
- Matched every row to the original preserved dataset and our frozen YOLOv9 manifests: filename, source image ID, split, empty flame targets and raw image SHA256 all match.
- Visually inspected the report's illustrated examples, all seven welding and five sparks candidates from the original archive, and all four reported unlabelled-flame originals. This is a semantic spot check, not independent re-review of every one of the 106 condition assignments.
- Thus the 54 training records were indeed available as negatives in the existing YOLOv9 training dataset. This does not require a new dataset or prove effective learning. Do not automatically transfer this exact membership claim to Nano's different original training split.

## Reported coverage and training membership

Counts below are retrieved/reviewed examples, not total condition prevalence. The condition names preserve the report's categories; see taxonomy cautions below.

| Condition | Reported confirmed across splits | Of these, in existing YOLOv9 train |
|---|---:|---:|
| Welding | 7 | 3 |
| Grinding/cutting sparks (report calls this grinding) | 5 | 3 |
| Sunlight through windows/openings | 7 | 4 |
| Reflections on metal/glass | 7 | 4 |
| Vehicle headlights/tail lights | 34 | 11 |
| LED indicator lights/signals | 9 | 4 |
| Red/orange clothing | 37 | 25 |
| Heaters/glowing elements | 0 confirmed in top100 | 0 confirmed |
| Screens displaying fire | 0 confirmed in top100 | 0 confirmed |
| Total | 106 | 54 |

The LED list includes traffic signals, displays and dashboard indicators as well as router LEDs. That supports a broad visible-light category, not specifically flashing industrial indicators. Still images cannot establish flashing or flicker. Phone-flashlight coverage was not a separate audited category, and clothing is not equivalent to every orange fabric/object geometry seen in webcam tests.

## Corrections/cautions to apply

1. **Welding is present.** Replace the earlier pilot-only statement that welding was not confirmed. Keep the old pilot result as historical evidence, not the current conclusion.
2. **Heaters and fire-on-screen remain not confirmed, not absent.** The page7 suggestion that the dataset has "few or none" is not justified by unsuccessful top100 retrieval. Searching/scoring every image with CLIP does not guarantee a relevant image reaches top100.
3. **Do not call all five sparks examples mechanical grinding.** Three visibly involve grinders. `neitherFireNorSmoke_CV038401.jpg` (train) and `neitherFireNorSmoke_CV026563.jpg` (test) involve cutting-torch/thermal-cutting scenes. Keep them in a separate cutting/sparks subcategory and review whether a physical torch flame is visible. A gas flame can be a positive even when industrial sparks are a nuisance; electric welding arcs are a different phenomenon. Do not oversample questionable thermal-cutting images as verified flame-free negatives before agreeing the policy.
4. **Category counts do not measure scene diversity.** Welding `CV000400` (val) and `CV013518` (train) visibly show the same construction scene with similar composition. The report itself notes the relationship. This is a scene-overlap warning, not a demonstrated exact-byte duplicate. The three training welding examples are not evidence of broad welding coverage.
5. **The new pilot comparison column needs provenance.** The original saved 500-image notebook had zero recorded reviews. The PDF may refer to later manual review; obtain that table if making historical pilot-count comparisons. This does not invalidate the new listed image examples.
6. **Coverage is not hardness.** A CLIP match is semantically relevant; an actual flame detector may reject it correctly. Use our trained detector's mistakes to decide which confirmed training examples deserve additional weight.

## Four label errors: important, but not training examples

The four original images visibly show real flames, while our corresponding annotation/manifests contain no flame boxes:

| Filename (prefix omitted here) | Split | Visible content |
|---|---|---|
| CV031809.jpg | test | Two burning lamps |
| CV032004.jpg | validation | Blue torch-lighter flame |
| CV032861.jpg | test | Flames behind a firefighter |
| CV035908.jpg | test | Burning vehicle visible from inside another car |

All use the `neitherFireNorSmoke_` filename prefix. They were **not training negatives in our YOLOv9 Kaggle-split run**, because none is in train. They are evaluation-label errors: a correct flame box can be counted as a false positive. Their impact on published scores is not measured, and four images do not explain the whole validation/test performance gap.

Do not silently edit the frozen dataset, drop difficult cases from published results, or move these val/test images into training. Preserve original reports. If creating a corrected evaluation set, annotate flames properly (including required smoke policy), version a correction manifest and evaluate all compared models on that same corrected set; report original and corrected scores separately. Keep ambiguous cases separate pending review.

## What to request from the audit teammate

One run-output bundle containing the full-run notebook/config and completion logs, sample/index manifest, per-category ranking CSVs and the human-verification CSV (verdict, reviewer, comments and image split/ID). Embedding cache is useful for continued searches but not required just to review the listed images. Remove credential outputs before sharing notebooks. A second reviewer should confirm torch-cutting policy and label errors.

These files would make the new full-pool execution reproducible; the PDF already gives usable image-level evidence of category presence.

## How this should guide future development

1. **No wholesale rebuild or immediate retraining.** Keep original YOLOv9 epoch20 as the current reference: the completed three-epoch unchanged-data refinement did not beat its AP selection criterion.
2. **Evaluate nuisance conditions before reweighting.** Run our model on the 54 reported training examples after policy checks; use the actual false detections to prioritise. Report scores on these as training diagnostics, not independent accuracy. Keep the 18 val and 34 test records in their original evaluation roles.
3. **One targeted training experiment, if justified.** Increase exposure to confirmed, actually confusing training cases while retaining diverse real flames and other negatives. Limit repetition of near-duplicate scenes. Add verified varied flashlight/heater/screen examples only where required by observed failures; do not assume the audited pool contains them or that a handful of frames establishes robustness.
4. **Protect small/bright flames.** Our same-video test found visible small-flame misses; suppressing all bright/white/orange regions would harm the target. Include legitimate difficult flame positives alongside nuisance work.
5. **Independent video evidence remains essential.** Neither CLIP nor these still-image counts measure false alarm events, detection delay, flashing-light rejection or generalisation to the intended camera/industrial scene. Keep development and evaluation recordings separate.
6. **Keep hardware cost stable where possible.** Data selection/reweighting with an unchanged network does not add inference layers or a second detector. It still requires new export/parity, TIDL/quantized-quality and actual AM62A checks before deployment. Do not claim 1 TOPS compliance from dataset coverage or ONNX file size.

## Safe meeting wording

"A CLIP-assisted audit has identified examples of welding, sparks, sunlight, reflections, vehicle lights, LEDs and red/orange clothing in FASDD. We verified 54 of the listed examples are already negative training images for our YOLOv9 run. Heater and fire-on-screen coverage remains unconfirmed. Presence in the dataset does not establish reliable rejection, so we are testing those conditions directly. Four missing flame labels were also found in validation/test and will be tracked through a versioned correction process."

Machine-readable evidence: `results/fasdd_full_audit/reported_confirmed.csv`, `label_issues.csv`, `verification_summary.json`. No training data, annotations or model weights were changed by this review.
