# Week 2 code index (branch `feature/week2-final`)

Built on `feature/v10-hard-negatives`. The full narrative and every number are in **`docs/WEEK2_SUMMARY.md`**.

| Folder | What it is |
|---|---|
| `experiments/yolov9_v10/` | V10 fine-tune workflow (hard negatives). Updated to the final as-run versions (`mine.py`, `report.py`, `weightops.py`, `START_HERE.md`). Own-clip frames are not included (data); only `own_flame/own_boxes.json`. |
| `experiments/yolov9_v11/` | V11 package as run on the lab PC (COCO init, 40 epochs, zoom/paste/degrade augmentation). **Rejected**: false alarms 4.3× V10. `BUGFIX_LOG.md` has the lab fix. |
| `experiments/yolov9_v12/` | V12 package as run: 3 fine-tune variants from V10 + a recordings fine-tune. `recordings_seeds.json` holds the flame positions marked for the 6 Nov videos. |
| `experiments/temporal_final/` | Temporal alarm layer: tracking, 2 s persistence, 3 s debounce, small-box threshold. Run 3 (ONFIRE duplicate fix). Protocol + spec. |
| `experiments/edge_export/` | Raw-head ONNX export for the AM62A (`export_raw.py`), board README, final rule note, list of the 15,884 FASDD v2 val images. ONNX files are **not** committed; re-create them with the script. |
| `experiments/field_tests/` | Scripts for our own videos: `compare_v10box_v12.py` (side-by-side grids, same alarm rule on both models), `flame_contrast.py` and `lowconf_daylight.py` (daylight analysis), `compare3.py` (V9/V10/A1, 5 Nov). Paths are hard-coded to the laptop layout; edit `ROOT` to run elsewhere. |
| `src/geometry/` | **W3** spatial calibration (`camera.json`, pixel to floor metres, range rings); see `docs/W3_SPATIAL_CALIBRATION.md`. CLI in `scripts/calibrate.py`. |
| `results/week2/` | Reports (V11, V12, temporal), locked `rule.json`, selections, V12 test numbers, field-test summaries. |

**Not in git:** weights (`.pth`), ONNX, videos and datasets. They live on the laptop and lab PC; see section 10 of `WEEK2_SUMMARY.md`.
