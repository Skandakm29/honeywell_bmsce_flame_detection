# Run on the Windows RTX A5000 lab PC

1. Transfer Honeywell-Lab.zip to the lab PC using your existing remote desktop file transfer or your Drive. The ZIP contains source only; no dataset or account credentials.
2. Extract it to a short local path, for example E:\Honeywell-Lab. Use a local SSD with at least 80 GB free. Do not run from inside the ZIP or a cloud-synced folder.
3. Open PowerShell in the extracted folder containing START_TRAINING.cmd. Run:

   `cmd /c .\START_TRAINING.cmd`

No administrator session or preinstalled Python is required. The launcher downloads pinned official uv/Python and CUDA PyTorch into this folder; it does not change PATH or the NVIDIA driver. Network access to GitHub, Python/PyTorch package sources, TI and Kaggle is required. Stop if lab policy blocks those downloads; do not bypass institutional controls.

The launcher automatically downloads the pinned FASDD dataset version 2 and checksum-verifies its annotations and the TI pretrained weights. Dataset access is public but if Kaggle requires account authorization, complete that normally and report the error. All eight engineering tests and a GPU train/export preflight must pass before long training. The initial throughput report includes EMA but omits validation/checkpoint/setup time. No morning deadline or detection accuracy is guaranteed.

The recipe remains 640 RGB input, effective batch 32 (A5000 first tries microbatch 16 and falls back on OOM), three 3-epoch LR trials on 8,192 training images followed by fresh full-data training for 30–40 epochs, AdamW, AMP, EMA, warmup/cosine and validation-based checkpoint selection. Model architecture and existing splits are retained. Exact duplicate caveats remain in the audit report. The old Colab checkpoint is not resumed: this is a separately identified run because code and local manifest paths differ. Do not copy Colab checkpoints into this run directory.

Keep the PowerShell window open and the lab PC powered on. Disconnecting the remote viewer does not end the Windows process; signing out, restarting or shutting down does. The launcher requests wakefulness only while it runs; institutional power policies can override it. Do not allow another person to sign out that session.

To inspect progress from a second PowerShell window:

`Get-Content .\run.log -Tail 30 -Wait`

Training logs appear every 20 batches, followed by each epoch's measured validation scores and elapsed seconds. If anything fails, send the last 30 lines from run.log (or setup.log if setup failed). Relaunching the same unchanged folder resumes valid checkpoints; do not edit config or move the folder during a run. A process lock prevents two training pipelines here. The Colab monitor cannot inspect a different remote computer automatically.

Successful completion writes full validation and test mAP@50, mAP@50:95, AP75, precision/recall/F1 with stated thresholds, prediction files, best/last/previous checkpoints, ONNX plus numeric parity report, recipe/source provenance and logs. AM62A board compilation and performance remain separate.

Copy Honeywell-results.zip and Honeywell-results.sha256 back to your own machine/Drive. Check Get-FileHash -Algorithm SHA256 against the saved hash, extract the copy and confirm checkpoints, reports and ONNX are readable. Only then remove this dedicated Honeywell-Lab folder from the lab PC, using normal file management. Do not uninstall shared software, touch other lab files or delete the only copy of results. Cleanup is deliberately not automatic.

Installation references: https://pytorch.org/get-started/previous-versions/ and https://docs.astral.sh/uv/getting-started/installation/ . Official uv 0.9.5 Windows archive SHA256 is pinned in lab_setup.ps1.
