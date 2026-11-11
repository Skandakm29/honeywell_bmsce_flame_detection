# Brief for the lab agent: run V11 end to end, without stopping

**Goal:** steps 0 → 1 → 1b → 1c → 2 → 3 → 4 → 5 → 6 all finish, with no data leakage and no rule changes. The owner wants the whole process done, so do not pause for approval.

## Where things are
- Package: `D:\Honeywell-V11`
- V10 code and venv: `D:\Honeywell-V10` (never edit)
- Work folder: `D:\Honeywell-V10-work`. V11 writes only these:
  - `*_v11` folders
  - `bench\v11`, `temporal\v11`, `diagnose\v11`
  - `status_v11`
  - `outbox\v11_*.zip`
- Logs: `D:\Honeywell-V10-work\v11_step_<n>.log`
- Done markers: `status_v11\step_<n>.done`
- Rules: `PROTOCOL_V11.md`

## Run loop
1. Find the next step without a `.done` marker. If nothing is running, start it: `cd D:\Honeywell-V11; .\run.ps1 <step>`. Long steps (1, 1b, 4, 5) go in the background.
2. While it runs, check every 30–60 minutes.
   - **Step 4:** note the epoch, loss and the per-epoch recall and hn_val numbers from `train_v11\history.json`.
3. **Gates you do yourself, then continue immediately:**
   - **Gate A, after step 1.** View `labelfix_v11\spotcheck_add_box_1.jpg` and `_2.jpg`.
     - Count crops that show a real physical flame inside the box.
     - Write `inbox\labelfix_spotcheck.json` with `add_box_total`, `add_box_real_flame` and `"reviewer": "lab Claude (AI, not human)"`.
   - **Gate B, after step 1b.** View every `dense_v11\review_XX.jpg`.
     - `ok` only if the box tightly covers a real physical flame; otherwise `bad`. Screens, lamps, lantern glow, reflections, sparks and sunsets are `bad`. When in doubt, `bad`.
     - Write `inbox\dense_review.csv` (`id,decision`, one row for every id).
   - **After step 2:** look at `manifests_v11\augment_preview.jpg`. Boxes should sit on flames, including small pasted ones.
     - If the boxes are clearly shifted or the images are garbage, treat it as a code bug (next section).
   - **After step 3:** record the pilot estimate in the log and continue. The training budget is 72 h.
4. When step 5 finishes, run step 6 (the FASDD test, exactly once; the code refuses a rerun).
   - Then write a final summary in the log: the selection reason line, the V10 vs V11 numbers from `REPORT_V11.md`, and the paths `outbox\v11_results.zip` and `bench\v11\test_summary.json`.

## When something fails
- First, rerun the same command (every step resumes).
- Fix environment problems: free disk space (delete `select_v11\cache` and temp files only), restart after a reboot or driver error, restart after a CUDA out-of-memory error.
- **Code bugs** (crash, exception, wrong path, wrong shape): you may fix package code under `workflow11\`.
  - Make the smallest fix that does what the docstring says.
  - Then run `D:\Honeywell-V10\.venv\Scripts\python.exe workflow11\rehash.py "<file> | <error> | <fix>"` and rerun the step.
- **Never:**
  - change `config_v11.json`, `PROTOCOL_V11.md`, thresholds, epochs, augmentation values, data choices or the selection rule
  - edit V10 code or V10 manifests
  - delete checkpoints, manifests or `train_v11`
  - move any val / hn_val / test / ONFIRE / held-out clip image into training
  - rerun the FASDD test
  - start over in a new folder
- If the only way forward breaks one of these rules, write the problem to `V11_AGENT_LOG.md` and stop. This is the only reason to stop.

## No leakage (check and log these)
- Step 2 must report 0 evaluation images in training. The code checks this. If it fails, it is a real problem: stop and report.
- Only these go into training: training-split own clips, the 4 dense clips, the 6 WhatsApp clips and V10's training manifest. Held-out own clips, hn_val, validation, test and ONFIRE never do.
- The FASDD test runs only after selection is final, and nothing is changed after seeing it.

## Every check
Append one line to `D:\Honeywell-V10-work\V11_AGENT_LOG.md`:

`time | step | state | key numbers | action taken`

Never invent numbers: copy them from the logs and JSON files.
