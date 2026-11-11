# V10 on the lab PC (RTX A5000): terminal-only steps

This folder is pulled from GitHub. **Do not edit any file in it**: step 0 checks every file's hash.
Data, checkpoints and results go to a separate work folder, by default `..\Honeywell-V10-work` next to this clone.
Nothing is pushed from the lab PC. Results come back to the laptop as zip files from `outbox\`.
Answers from the laptop arrive in `inbox\` through `git pull`.

Run everything from **64-bit PowerShell** inside this folder. If a step fails, rerun the same command to resume. If it fails again, send the step's log file (named in the red message).

## Needs
**Current lab machine (4 Nov): RTX PRO 4000 Blackwell, drives C: and D: only.** Put this folder at `D:\Honeywell-V10` (work folder becomes `D:\Honeywell-V10-work`). The old `E:\Honeywell-Lab` files are not on this machine: copy V9 `best.pth` from the shared Drive and let step 1 download FASDD (leave out `--fasdd-root`).

- RTX A5000, internet, **≥ 150 GB free** on the drive that holds the work folder
- The original V9 weights: `E:\Honeywell-Lab\outputs\full\best.pth` or `E:\Honeywell-Refine\initial_best.pth` (step 0 checks the SHA256 starts with `48d8e654`)
- The existing extracted FASDD from the V9 run, if it's still there: `E:\Honeywell-Lab\data` (otherwise step 1 downloads it from Kaggle, ~12 GB)
- The recorded clips in one folder with `clips.csv` (copy `clips_template.csv` and fill one row per clip)
- Kaggle access for D-Fire (same as the earlier FASDD download)

## Steps

| Step | Command | Time (estimate) | Send back |
|---|---|---|---|
| setup | `powershell -ExecutionPolicy Bypass -File .\setup.ps1` | 10–20 min | only if it fails |
| 0 preflight | `.\run.ps1 0 --v9 E:\Honeywell-Lab\outputs\full\best.pth` | 5 min (downloads CLIP weights) | only if it fails |
| 1 data | `.\run.ps1 1 --fasdd-root E:\Honeywell-Lab\data --clips E:\V10_clips` | 2–4 h (Open Images CSVs ~5 GB + ~20k images) | - |
| 2 mining | `.\run.ps1 2` | 30–60 min | **`outbox\step2_review.zip`** → STOP 1 |
| 2b | `git pull` then `.\run.ps1 2b` | 1 min | - |
| 3 manifests | `.\run.ps1 3` | 30–60 min | **`outbox\step3_manifest_report.zip`** → STOP 2 |
| 4 V9 baseline | `.\run.ps1 4` | 1–2 h | - |
| 5 training | `.\run.ps1 5` | ~9–10 h, overnight | - |
| 6 results | `.\run.ps1 6` | 2–3 h | **`outbox\step6_results.zip`** → STOP 3 |
| 7 test | `git pull` (brings `inbox\APPROVE_TEST.txt`) then `.\run.ps1 7` | 15 min | `bench\v9\test_summary.json`, `bench\v10\test_summary.json` |

Optional evaluation sets (test only, never trained on), if you can download them:
- **ONFIRE** contest videos (request access on the MIVIA ONFIRE page): give their folder to step 1 with `--onfire <folder>`. The folder needs an `onfire.csv` with columns `clip,fire_start_frame`, with the frame left empty for non-fire videos. Fire start times may be given in seconds instead, as `fire_start_s`.
- **MIVIA** fire videos: give their folder to step 1 with `--mivia <folder>`. The folder needs a `mivia.csv` with the same columns as `clips.csv`.
- **BoWFire** images: give their folder with `--bowfire <folder>`.

At each **STOP**, wait for the go-ahead before running the next step.

The work folder location can be changed by setting `$env:V10_WORK = 'D:\somewhere'` before every `run.ps1` call.
Keep the same value for all steps.

## Watching a long step from another PowerShell window
```powershell
Get-Content ..\Honeywell-V10-work\step_5.log -Tail 30 -Wait
```

## Rules
- FASDD **test** is used only in step 7, and only after approval.
- Never move val/test images into training. Never delete the work folder until the results zip has been copied and checked.
- Mining reviews are done with CLIP plus AI visual review on the laptop. Nobody calls them human-verified.

## What each step does
- **1 data:** loads FASDD (annotation hashes checked; 63,544 / 15,884 / 15,884 images).
  - Downloads D-Fire from Kaggle (fire boxes kept, smoke boxes dropped).
  - Downloads Open Images V7 images for 31 fire-like nuisance labels; anything tagged fire, candle, fireplace etc. is excluded.
  - Splits your clips 70/30 by whole clip and extracts non-fire frames at 1 fps.
  - Makes ~2,000 synthetic "fire on a screen" negatives.
- **2 mining:** runs V9 at confidence ≥ 0.20 on all candidate negatives, and checks hits with CLIP.
  - Unclear hits, plus FASDD train "no flame" images where V9 is confident, go into review sheets.
- **3 manifests:**
  - removes near-duplicates of any evaluation image
  - groups near-duplicates so hold-out sets don't share scenes with training
  - builds the V10 training list: mined hard negatives ×3, own clip frames ×3, flame images kept at ≥ 30%
  - builds `hn_val`, the held-out hard-negative check
- **4 / 6 benchmarks:** for V9 and for the selected V10:
  - FASDD val: AP, precision/recall, per flame size
  - `hn_val` false positives per category
  - your held-out clips (and MIVIA): frame hits, plus alarm events with the temporal layer (event recall, time to alarm, false alarms/hour)
  - diagnosis: false-positive clusters, D-RISE saliency maps, colour/blur/occlusion tests
- **5 training:** starts from V9 and trains 10 epochs at LR 3e-5, saving every epoch.
- **6 selection rule (fixed before training):** a candidate must keep val AP50 and recall within 1 point of V9. Among those, the lowest `hn_val` false-positive rate wins. If no epoch qualifies, V9 is kept.
