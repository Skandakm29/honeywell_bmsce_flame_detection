# A5000 short fine-tune

1. Transfer `Honeywell-Refine.zip` to the lab PC using Chrome Remote Desktop file transfer or team Drive.
2. Extract it to **E:\Honeywell-Refine**. Do not place it inside an existing Honeywell-Lab folder.
3. Ensure 80 GB free on E: for data, environment, caches and checkpoints. The first launch downloads the same pinned Kaggle dataset version2; internet is required. If you already have the exact raw dataset ZIP, copy it to `E:\Honeywell-Refine\dataset.zip` before launching to avoid downloading it.
4. Open PowerShell on the lab PC and run:

```powershell
Set-Location E:\Honeywell-Refine
powershell -NoProfile -ExecutionPolicy Bypass -File .\lab_setup.ps1
```

Keep this terminal open. This requires the RTX A5000. It installs the pinned Python3.11/CUDA PyTorch environment inside this folder, verifies the package and dataset, strictly loads trained epoch20, and trains at most three additional epochs. No LR pilots. Dataset/labels/input640 unchanged; LR1e-5, mosaic0, batch16/accum2. First-time setup/download/audit adds time before training.

The three-hour training budget is checked at successful optimizer-update boundaries and includes measured resumed training time/validation. Setup and final evaluation/export are additional. A complete run is expected to take a few hours on the earlier A5000; this is not a deadline guarantee. If no valid new checkpoint beats the initial model, the original is retained. Epoch0 in this experiment means the original trained epoch20, not untrained weights.

After interruption, rerun the same command from this folder: matching last/previous checkpoints resume without restarting pilots. Do not edit files/config during a run. The OS lock prevents two launchers in the same folder; do not start another job in a copied folder while this one is active.

## Watch progress in another terminal

```powershell
Get-Content E:\Honeywell-Refine\setup.log -Tail 30 -Wait
```

The setup transcript contains dependency output and training logs. Structured history/config/environment/checkpoints are under `outputs/full`. Full-validation baseline/candidate reports and `comparison.json` show measured trade-offs. `selected_export` contains the validation-selected ONNX and parity report. No test/video claims are produced automatically and the held-out test is not used to choose weights.

## Finish and cleanup

The run creates `Honeywell-finetune-results.zip` and `Honeywell-finetune-results.sha256` and checks ZIP CRC. Copy both to the owner's laptop/team storage. Readable-checkpoint/hash verification on that separate copy must happen **before deleting any lab files**. Keep the original baseline backup. Do not delete account-wide caches or unrelated files. No cleanup is automated.

No GPU training was performed by creating this package. Board/TIDL/INT8/1-TOPS suitability and independent-video evaluation remain separate checks.
