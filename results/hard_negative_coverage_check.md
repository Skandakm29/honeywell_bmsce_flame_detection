# FASDD-CV hard-negative coverage check — 31 October 2026

Sources: original FASDD paper https://doi.org/10.1080/10095020.2024.2347922 and official record https://doi.org/10.57760/sciencedb.j00104.00103 . Read local preserved Kaggle ZIP annotations/train.json and archive member names; no dataset/training modifications.

Original authors describe confusing non-fire/non-smoke images and variation in distance, indoor/outdoor scenes and daylight/night conditions. Annotation guidance addresses partial occlusion, color differences, small targets and reflections. This does not establish complete coverage of a deployment checklist or per-condition counts.

Raw Kaggle training-image groups, computed directly from annotation class membership:

| Image group | Count |
|---|---:|
| Flame and smoke | 13,435 |
| Flame only | 8,368 |
| Smoke only | 15,610 |
| Neither annotated flame nor smoke | 26,133 |

Total raw train 63,546; fire-positive 21,803; negative for our single-class objective 41,743. Two flame-positive invalid records are quarantined by current YOLOv9 preparation, giving 63,544 final train images. These counts describe annotation membership, not exhaustive semantic verification.

Reproducibly sampled 32 images from the no-annotations training group, random seed71. IDs and original metadata: hard_negative_sample_ids.json. Thumbnails: hard_negatives_check_1.jpg and hard_negatives_check_2.jpg. Visually inspected both sheets. Confirmed red/orange clothing in neitherFireNorSmoke_CV032843.jpg; other samples show red lipstick, red peppers, red packaging/graphics, orange/red foliage, daylight outdoor and indoor classroom scenes. Do not infer that samples are sufficiently representative or that all negative labels are correct.

Smoke-only presence is established by annotation counts and prior inspected smoke examples. Red/orange clothing presence is confirmed by direct sample inspection. Exact requested subconditions sunlight through windows, sun reflections on metal/glass, welding, grinding sparks, vehicle lights, flashing LEDs, heaters, polished-surface reflections and monitor/video showing fire were NOT established by this check. Mark them unconfirmed, not absent. Filenames provide image group and numeric ID only; COCO has fire/smoke box classes, without an inventory of these nuisance conditions. A still-image archive does not by itself verify flashing/flicker or real-time false-alarm performance. Different fuels and low-contrast/clutter coverage/quantities are also not established by metadata.

Conclusion: do not assert FASDD lacks hard negatives or recommend wholesale dataset replacement. Some checklist coverage already exists; exact remaining coverage and deployment performance need targeted verification. Current lab run remains unchanged.
