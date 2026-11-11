# Week 4 on the lab PC: flame + obstruction (one model)

Rules: `PROTOCOL_W4.md`.

**Copy** this folder to `D:\Honeywell-W4`. It reuses `D:\Honeywell-V10\.venv` and the V10 code, V12's `workflow12\data11.py` (all read-only) and the work folder `D:\Honeywell-V10-work`.

**Needs:**
- `D:\Honeywell-V10-work\manifests_v12\` (from V12 step 4b)
- `D:\Honeywell-V10-work\select_v12_rec\best.pth` (final V12)

| Step | Command | Time (estimate) | What it does |
|---|---|---|---|
| 0 | `.\run.ps1 0` | 1 min | checks GPU, code, manifests, start checkpoint SHA |
| 1 | `.\run.ps1 1` | 5–10 min | builds `manifests_w4` and `preview_val.jpg` |
| Gate A | look at the preview | 5 min | write your name and the date into `D:\Honeywell-V10-work\status_w4\gate_a.txt` |
| 2 | `.\run.ps1 2` | about 5–6 h (10 epochs) | training; rerun the same command to resume |
| 3 | `.\run.ps1 3` | 10 min | selection + ONNX + report → `outbox\w4_results.zip` |

**Gate A:**
- **Pass** if the blue boxes tightly cover the drawn obstructions, flames hidden behind them have no red box, and visible flames keep theirs.
- **Do not start step 2 otherwise.** Report what looks wrong.

**Different paths:** set `$env:V10_PACKAGE`, `$env:V12_PACKAGE` or `$env:V10_WORK` before `run.ps1`.

**Start from V10 instead of V12:** set
- `$env:W4_START_CHECKPOINT = 'D:\Honeywell-V10-work\FROZEN_V10\selected.pth'`
- `$env:W4_START_SHA256 = 'd046dab613faa72ec650d37dc40d0740ff50ac09df4eb97919d10cfd532b721d'`

**Watch a long step:** `Get-Content D:\Honeywell-V10-work\w4_step_2.log -Tail 30 -Wait`

**Send back:** `outbox\w4_results.zip`. It contains `REPORT_W4.md`, `selection.json`, `history.json`, the build report, the preview, and the ONNX with its contract.
