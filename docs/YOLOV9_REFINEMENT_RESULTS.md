# YOLOv9 three-epoch refinement: measured outcome

Source: user-supplied `09zgrcx3643jppvi.zip`, containing lab results archive and checksum. Copied/extracted to the owner's ignored `results/finetune_backup_20261101/` directory. Inner archive SHA256 `6a2ca6080dc90ff9a3af56afe8f6e1d91337a3917dd9aa7313b24e5fa4bf675b` matches the lab checksum; ZIP CRC passed. Available source/package hashes match; the lab bundle omits original-manifest copies and some launcher files, which remain in the separately retained initial package/baseline backup.

## Completed experiment

- RTX A5000, Python 3.11.14, PyTorch 2.5.1+cu124.
- Strictly initialized from original trained epoch20: 1,770 tensors loaded, zero reinitialized.
- Three additional epochs; input640, batch16/accum2, AdamW LR1e-5, warmup100updates, mosaic0, AMP/EMA retained.
- No dataset/label changes. Full training/monitor/full-validation/test manifest record contents, including image hashes and order, match the original after excluding absolute image paths.
- Epoch durations: 1,373.43 / 1,378.93 / 1,379.09 seconds. Total approximately **68.9 minutes**, including per-epoch monitor validation; setup and final evaluations/export are additional.

## Monitor validation — fixed 2,000-image subset

| Checkpoint | AP50:95 | AP50 | Precision | Recall | F1 | FP boxes | FN boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original (refinement epoch0) | 43.34% | 73.81% | 81.04% | 57.15% | 67.03% | 215 | 689 |
| Additional epoch1 | 42.67% | 73.66% | 80.43% | 57.77% | 67.25% | 226 | 679 |
| Additional epoch2 | 43.09% | 73.42% | 80.46% | 58.89% | 68.01% | 230 | 661 |
| Additional epoch3 | 42.76% | 73.58% | 80.22% | 58.52% | 67.67% | 232 | 667 |

P/R/F1: confidence0.40, matchingIoU0.50. These are actual reported box metrics, not alarm events. Epoch2 improved monitor recall/F1 but increased FP boxes and did not exceed original AP50:95. All additional epochs had lower monitor AP than the original. Falling training loss alone did not establish a better detector.

## Selection and full-validation interpretation

The predefined selection criterion was AP50:95, with the original eligible as a fallback. Therefore **epoch0 was retained**. It represents the already trained epoch20, not an untrained model or a training failure.

The full-validation baseline and supplied `candidate_validation` report both evaluate the original weights. Their checkpoint file hashes differ because of serialization, but the tensor values are identical. Do not claim full-validation results for an actual refined epoch from that candidate report. Full-validation performance of newly refined weights remains unestablished by this archive.

The retained model's full-validation metrics reproduce the original: AP50:95 44.31%, AP50 74.60%, precision80.65%, recall58.73%, F1 67.97%.

The selected ONNX checksum is exactly the original `332cb480c4a4f3d8cdfbb3f5545a2e08211f6ae3db492d1d08d22318e2c78544`. Lab ONNX checker/parity passed on three inputs. There is no new selected model to substitute for the current deployment/demo file. No new test-set or video quality claim follows from these results.

## Backup and next decision

Keep the trained last/previous checkpoints, history and original baseline. Before cleanup, verify the copied checkpoints load and that selected/initial tensors equal the original. Local verification evidence is `results/finetune_backup_20261101/backup_verification.json`.

Do not extend training blindly: this short unchanged-data/mosaic-off experiment did not win on the agreed metric. Epoch2's recall/F1 trade-off is worth recording, but its exact weights were not retained as a separate best-refined checkpoint by this selection policy. The last/previous checkpoints remain available for diagnostic evaluation; report their exact epoch/batch and EMA/raw choice if used. Do not relabel an epoch3 checkpoint as epoch2.

Next useful work is targeted hard-case training/validation coverage and small/bright-flame plus nuisance evaluation on independent clips. Any new criterion or data change must be a separately named experiment, not a retrospective declaration that this run succeeded. Hardware/TIDL/INT8/1TOPS checks remain pending.
