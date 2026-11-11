# V12 on the lab PC

V12 = fine-tunes from frozen V10, testing one change at a time, then a fine-tune on the owner's far-flame recordings. Rules: `PROTOCOL_V12.md`. Agent brief: `AGENT_V12.md`.

**Extract `Honeywell-V12.zip` to `D:\`** (it creates `D:\Honeywell-V12`). It reuses `D:\Honeywell-V10\.venv`, the V10 code (read-only), `D:\Honeywell-V10-work`, and V11's `manifests_v11` (read-only). Start only after V11 step 6 is done.

| Step | Command | Time (estimate) | Needs |
|---|---|---|---|
| 0 | `.\run.ps1 0` | 2 min | V11 finished |
| 1 | `.\run.ps1 1` | 15-30 min | - |
| 2 | `.\run.ps1 2` | about 10 h (3 x 8 epochs, about 25 min each) | rerun the same command to resume |
| 3 | `.\run.ps1 3` | about 1.5 h | gives `outbox\v12_results.zip` |
| 4a | `.\run.ps1 4a` | 15-30 min | recordings in `D:\Honeywell-V10-work\recordings_v12\`. **Can run during step 2.** |
| 4b | `.\run.ps1 4b` | 5 min | **Gate C:** `inbox\rec_review.csv`. **Can run during step 2.** |
| 4c | `.\run.ps1 4c` | about 1.5-2 h | needs steps 3 and 4b |
| 4d | `.\run.ps1 4d` | about 1 h | - |
| 5 | `.\run.ps1 5` | 30 min | runs once; the code refuses a rerun |

**Recordings.** Unzip the owner's Drive zip(s) into `D:\Honeywell-V10-work\recordings_v12\` (sub-folders are fine).
- Name flame clips `<place>_<candle|diya|lighter>_<N>m[_locked].mp4` and no-flame clips `<place>_noflame.mp4`.
- Any place name (lowercase, underscores ok). `corridor` is the distance TEST; every other place goes to training.
- Clips listed in `recordings_seeds.json` (the owner's 6 Nov clips) are boxed from flame positions marked by Claude (AI) and then followed frame by frame; other clips use V10 or the flicker rule. Every box is still reviewed at Gate C.
- Step 4a lists badly named files in `rec_v12\propose_summary.json`. Report those to the owner; never rename them yourself.

**Gate C (boxes).** View every `D:\Honeywell-V10-work\rec_v12\review_XX.jpg`.
- Mark `ok` only if the red box tightly covers the real candle, diya or lighter flame. Otherwise mark `bad`: no flame visible, box on a lamp, reflection or window, or box clearly off. When in doubt, `bad`.
- Write `inbox\rec_review.csv` with the header `id,decision` and one row for every id.
- Log that the review was done by an AI, not a human.

**Without recordings:** step 5 can run after step 3 only if the OWNER writes `inbox\skip_recordings.txt`.

Watch a long step: `Get-Content D:\Honeywell-V10-work\v12_step_2.log -Tail 30 -Wait`
