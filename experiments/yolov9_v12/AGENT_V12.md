# Brief for the lab agent: run V12

**Goal:** steps 0 → 1 → 2 → 3 run back to back without stopping. When the owner's recordings arrive, run 4a → Gate C → 4b → 4c → 4d → 5. No data leakage and no rule changes.

## Where things are
- Package: `D:\Honeywell-V12`
- V10 code and venv: `D:\Honeywell-V10` (never edit)
- Work folder: `D:\Honeywell-V10-work`
- V11 outputs (`manifests_v11`, `train_v11`, `select_v11` ...) are read-only inputs. Never edit or delete them.
- V12 writes only these:
  - `*_v12` folders, `bench\v12*`, `temporal\v12*`, `diagnose\v12`
  - `status_v12`, `REPORT_V12.md`, `outbox\v12_*.zip`
- Logs: `v12_step_<n>.log`. Done markers: `status_v12\step_<n>.done`. Rules: `PROTOCOL_V12.md`.

## Run loop
1. **Start condition:** V11 step 6 done (`status_v11\step_6.done`) and nothing else using the GPU.
2. Find the next step without a `.done` marker and start it: `cd D:\Honeywell-V12; .\run.ps1 <step>`. Run steps 1, 2, 3, 4c and 4d in the background.
3. Check every 30-60 min. During step 2, note the variant, epoch, loss, and monitor recall and hn_val numbers from `train_v12\<variant>\history.json`.
4. **Recordings, in parallel to save time:** as soon as `D:\Honeywell-V10-work\recordings_v12\` holds clips, run step 4a, do Gate C and run step 4b, even while step 2 is training. They use little GPU. Never stop or restart step 2 for them.
5. **After step 3:** write the selection reason, the best variant and epoch, and the table from `REPORT_V12.md` to the log. If step 4b is done, start step 4c right away, then 4d and 5. Do not run step 5 without the recordings, unless the owner has written `inbox\skip_recordings.txt`.
6. **Gate C, after step 4a.** View every `rec_v12\review_XX.jpg`.
   - `ok` only if the box tightly covers a real candle, diya or lighter flame. Lamps, reflections, windows, no visible flame, or a box clearly off are `bad`. When in doubt, `bad`.
   - Write `inbox\rec_review.csv` (`id,decision`, one row for every id).
   - Log: "review by lab Claude (AI, not human)", with the counts per clip.
   - Report the badly named files (`ignored_names` in `rec_v12\propose_summary.json`) to the owner. Never rename them.
7. After step 5, write a final summary to the log:
   - the selection reason lines (step 3 and step 4d)
   - the test table and the distance-test table from `REPORT_V12.md`
   - the path `outbox\v12_results.zip`

## When something fails
- First, rerun the same command. Every step resumes.
- Fix environment problems: free disk space (delete only `measure_v12` caches and temp files), restart after a reboot, driver error or CUDA out-of-memory error.
- **Code bugs** (crash, exception, wrong path, wrong shape): you may fix code under `workflow12\`.
  - Make the smallest fix that does what the docstring says.
  - Then run `D:\Honeywell-V10\.venv\Scripts\python.exe workflow12\rehash.py "<file> | <error> | <fix>"` and rerun the step.
- **Never:**
  - change `config_v12.json`, `PROTOCOL_V12.md`, thresholds, epochs, augmentation values, place splits or the selection rule
  - edit V10 code, V10 or V11 manifests, or V11 outputs
  - delete checkpoints or `train_v12`
  - train on val / hn_val / test / ONFIRE / held-out clips / corridor
  - rerun step 5
  - write `inbox\skip_recordings.txt` yourself (only the owner may)
- If the only way forward breaks one of these rules, write the problem to `V12_AGENT_LOG.md` and stop.

## Every check
Append one line to `D:\Honeywell-V10-work\V12_AGENT_LOG.md`:

`time | step | state | key numbers | action taken`

Never invent numbers: copy them from the logs and JSON files.
