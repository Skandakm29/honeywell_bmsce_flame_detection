# V11 on the lab PC (RTX PRO 4000 Blackwell)

V11 = same YOLOv9-S-TI-Lite 640, retrained from COCO weights for 40 epochs on the V10 training set plus label fixes, with zoom-out, small-flame paste and camera-degradation augmentation. The evaluation sets are byte-identical to V10's. The rules are in `PROTOCOL_V11.md`; the monitoring agent's brief is in `AGENT_V11.md`.

Unzip to `D:\Honeywell-V11`. It reuses `D:\Honeywell-V10\.venv`, the V10 code (read-only) and `D:\Honeywell-V10-work`. Do not edit package files: every step checks their hashes.

| Step | Command | Time (estimate) | Gate |
|---|---|---|---|
| 0 | `.\run.ps1 0` | 5 min | - |
| 1 | `.\run.ps1 1` | 20-40 min | **Gate A:** spot check (below) |
| 2 | `.\run.ps1 2` | 20-40 min | look at `outbox\v11_step2_build.zip` preview |
| 3 | `.\run.ps1 3` | 5 min | log the estimate (training budget 72 h) |
| 4 | `.\run.ps1 4` | about 16-20 h | rerun the same command to resume |
| 5 | `.\run.ps1 5` | 3-5 h | results in `outbox\v11_results.zip` |
| 6 | `.\run.ps1 6` | 15 min | FASDD test, exactly once (the code refuses a rerun) |

**Gate A (spot check).** Open `D:\Honeywell-V10-work\labelfix_v11\spotcheck_add_box_1.jpg` and `_2.jpg` (72 crops).
- Count crops that show a **real physical flame** inside the red box.
- Screens, lamps, reflections, sparks and sunsets are NOT flames.
- Write `inbox\labelfix_spotcheck.json`:

```json
{"add_box_total": 72, "add_box_real_flame": <count>, "reviewer": "lab Claude (AI, not human)"}
```

The result follows the fixed rule automatically: at least 80% real flames means boxes are added, otherwise those images are dropped.

Watch a long step: `Get-Content D:\Honeywell-V10-work\v11_step_4.log -Tail 30 -Wait`

**Step 1b also covers the 6 WhatsApp clips in `whatsapp\`** (owner decision: all go to TRAINING). The 3 flame clips get proposals for review; the 3 non-fire clips give 1 frame/s as negatives.

**Gate B (dense boxes).** Open every `D:\Honeywell-V10-work\dense_v11
eview_XX.jpg`. Each crop is numbered `id N`.
- Mark `ok` only if the red box tightly covers a real physical flame. Otherwise (no flame visible, box on a lamp or reflection, box badly off) mark `bad`.
- Write `inbox\dense_review.csv` with the header `id,decision` and one row for EVERY id.
- Note in the agent log that the review was done by an AI, not a human.
