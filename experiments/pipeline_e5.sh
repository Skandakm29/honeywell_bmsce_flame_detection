#!/usr/bin/env bash
# End-to-end, resumable background job on the Colab GPU (5-epoch YOLOX-S-Lite + baseline comparison):
#   1 manifests  2 smoke tests  3 batch size  4 training (resume-aware)  5 curves
#   6 S-Lite validation (threshold) + test  7 baseline validation (threshold) + test
#   8 ONNX exports  9 comparison report
# Each stage is skipped when its result already exists on Drive; any failure stops the job and
# leaves the error in logs/pipeline.log. A training checkpoint is never ignored: if last.pth exists
# it is verified and resumed (train.py also refuses a fresh start next to an existing last.pth).
set -euo pipefail
export MPLBACKEND=Agg
: "${PY:?}" "${REPO:?}" "${KAGGLE:?}" "${DRIVE_EXP:?}" "${CHECKPOINT:?}"
BASELINE_CKPT="${BASELINE_CKPT:-}"
EXP="$REPO/experiments"
MANIFESTS="$DRIVE_EXP/manifests_v2"
SMOKE="$DRIVE_EXP/smoke_v3"
CONFIGS="$DRIVE_EXP/configs"; LOGS="$DRIVE_EXP/logs"; RUN="$DRIVE_EXP/checkpoints/s_lite_e5_v1"
VAL="$DRIVE_EXP/validation"; TEST="$DRIVE_EXP/test"; PLOTS="$DRIVE_EXP/plots"; EXPORTS="$DRIVE_EXP/exports"
mkdir -p "$SMOKE" "$CONFIGS" "$LOGS" "$RUN" "$VAL" "$TEST" "$PLOTS" "$EXPORTS" "$DRIVE_EXP/reports"
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }

if [ ! -f "$MANIFESTS/prepare_report.json" ]; then
  log "STAGE 1/9 manifests from the Kaggle splits (validation of annotations, paths, boxes, classes)"
  "$PY" "$EXP/prepare_kaggle.py" --root "$KAGGLE" --output "$MANIFESTS" --workers 16
fi

if [ ! -f "$SMOKE/PASSED" ]; then
  log "STAGE 2/9 smoke test: weights, labels, forward, loss, backward, save/reload, output shape"
  "$PY" "$EXP/smoke_test.py" --data "$MANIFESTS" --checkpoint "$CHECKPOINT" --output "$SMOKE" --batch 16 --device cuda
  LOOP="$SMOKE/loop_$(date +%Y%m%d_%H%M%S)"
  COMMON=(--config "$EXP/configs/smoke.json" --data "$MANIFESTS/smoke" --checkpoint "$CHECKPOINT" --output "$LOOP" --device cuda)
  log "STAGE 2/9 smoke training loop: interrupt after 2 updates, verify checkpoint, resume, finish, export, evaluate"
  STATUS_DIR="$LOOP/logs" "$PY" "$EXP/train_s.py" "${COMMON[@]}" --stop-after-updates 2 || true
  [ -f "$LOOP/last.pth" ] || { log "SMOKE FAILED: no checkpoint after interruption"; exit 1; }
  "$PY" "$EXP/verify_checkpoint.py" --path "$LOOP/last.pth" --arch s --kind train
  STATUS_DIR="$LOOP/logs" "$PY" "$EXP/train_s.py" "${COMMON[@]}" --resume "$LOOP/last.pth"
  "$PY" "$EXP/verify_checkpoint.py" --path "$LOOP/best.pth" --arch s --kind weights
  "$PY" "$EXP/evaluate_s.py" --arch s --checkpoint "$LOOP/best.pth" --data "$MANIFESTS/smoke" --split val --size 640 \
    --output "$LOOP/eval_val" --latency-images 20 --review 8
  for f in best.pth last.pth export.json logs/status.json logs/resume_events.json eval_val/metrics.json eval_val/curves.png; do
    [ -f "$LOOP/$f" ] || { log "SMOKE FAILED: missing $f"; exit 1; }
  done
  "$PY" -c "import json,sys; r=json.load(open('$SMOKE/smoke_report.json')); sys.exit(0 if r.get('passed') else 1)"
  echo "$LOOP" > "$SMOKE/PASSED"
  log "SMOKE PASSED (ONNX parity on the 4-step smoke model: $("$PY" -c "import json; print(json.load(open('$LOOP/export.json')).get('parity_passed'))"))"
fi

if [ ! -f "$CONFIGS/run_settings.json" ]; then
  log "STAGE 3/9 batch size: 16 requested, checked against the measured smoke-test memory"
  "$PY" - "$SMOKE/smoke_report.json" "$CONFIGS/run_settings.json" <<'EOF'
import json, sys, torch
report, target = sys.argv[1], sys.argv[2]
smoke = json.load(open(report))
peak16 = smoke['speed']['peak_gpu_memory_MiB']
total = torch.cuda.get_device_properties(0).total_memory/2**20
batch, accumulate = (16,1) if peak16 <= .85*total else (8,2)
settings = {'batch':batch,'accumulate':accumulate,'measured_peak_MiB_at_batch':{str(smoke['batch']):peak16},
            'gpu_total_MiB':total,'gpu':torch.cuda.get_device_name(0),
            'rule':'batch 16 if the measured batch-16 training peak <= 85% of GPU memory, else 8 x accumulate 2 (same effective batch)'}
json.dump(settings, open(target,'w'), indent=2)
print(settings)
EOF
  cp "$EXP/configs/e5_v1.json" "$CONFIGS/e5_v1.json"
fi
BATCH=$("$PY" -c "import json; print(json.load(open('$CONFIGS/run_settings.json'))['batch'])")
ACCUMULATE=$("$PY" -c "import json; print(json.load(open('$CONFIGS/run_settings.json'))['accumulate'])")

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 60 >> "$LOGS/gpu_usage.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

if [ ! -f "$RUN/export.json" ]; then
  RESUME=()
  if [ -f "$RUN/last.pth" ]; then
    log "STAGE 4/9 verifying existing checkpoint before resuming (never restart from pretrained weights)"
    "$PY" "$EXP/verify_checkpoint.py" --path "$RUN/last.pth" --arch s --kind train --record "$LOGS/resume_checkpoint_check.json"
    RESUME=(--resume "$RUN/last.pth")
  fi
  log "STAGE 4/9 training 5 epochs: batch $BATCH x accumulate $ACCUMULATE ${RESUME[*]:-(fresh start from TI COCO weights)}"
  STATUS_DIR="$LOGS" "$PY" "$EXP/train_s.py" --config "$EXP/configs/e5_v1.json" --data "$MANIFESTS" --checkpoint "$CHECKPOINT" \
    --output "$RUN" --batch "$BATCH" --accumulate "$ACCUMULATE" "${RESUME[@]}" 2>&1 | tee -a "$LOGS/train.log"
fi
"$PY" "$EXP/verify_checkpoint.py" --path "$RUN/last.pth" --arch s --kind train --record "$LOGS/final_last_checkpoint.json" > /dev/null
"$PY" "$EXP/verify_checkpoint.py" --path "$RUN/best.pth" --arch s --kind weights --record "$LOGS/best_checkpoint.json" > /dev/null
log "checkpoints verified: $RUN/last.pth, $RUN/best.pth"

log "STAGE 5/9 training curves"
"$PY" "$EXP/plot_history.py" --run "$RUN" --plots "$PLOTS"

if [ ! -f "$VAL/s_lite/metrics.json" ]; then
  log "STAGE 6/9 S-Lite full validation (threshold selection)"
  "$PY" "$EXP/evaluate_s.py" --arch s --checkpoint "$RUN/best.pth" --data "$MANIFESTS" --split validation_full --size 640 --output "$VAL/s_lite"
fi
if [ ! -f "$TEST/s_lite/metrics.json" ]; then
  log "STAGE 6/9 S-Lite final test with its validation-selected threshold"
  "$PY" "$EXP/evaluate_s.py" --arch s --checkpoint "$RUN/best.pth" --data "$MANIFESTS" --split test --size 640 \
    --output "$TEST/s_lite" --final-test --threshold-from "$VAL/s_lite/metrics.json"
fi

if [ -n "$BASELINE_CKPT" ] && [ -f "$BASELINE_CKPT" ]; then
  if ! grep -q '"found": true' "$CONFIGS/baseline_checkpoint.json" 2>/dev/null; then
    log "STAGE 7/9 verifying the baseline checkpoint is YOLOX-Nano-ti-lite (read-only)"
    "$PY" "$EXP/verify_checkpoint.py" --path "$BASELINE_CKPT" --arch nano --kind weights --record "$CONFIGS/baseline_checkpoint.json"
    "$PY" -c "import json; p='$CONFIGS/baseline_checkpoint.json'; d=json.load(open(p)); d['found']=True; json.dump(d,open(p,'w'),indent=2)"
  fi
  if [ ! -f "$VAL/baseline/metrics.json" ]; then
    log "STAGE 7/9 baseline full validation (threshold selection, 416 px)"
    "$PY" "$EXP/evaluate_s.py" --arch nano --checkpoint "$BASELINE_CKPT" --data "$MANIFESTS" --split validation_full --size 416 --output "$VAL/baseline"
  fi
  if [ ! -f "$TEST/baseline/metrics.json" ]; then
    log "STAGE 7/9 baseline final test with its validation-selected threshold"
    "$PY" "$EXP/evaluate_s.py" --arch nano --checkpoint "$BASELINE_CKPT" --data "$MANIFESTS" --split test --size 416 \
      --output "$TEST/baseline" --final-test --threshold-from "$VAL/baseline/metrics.json"
  fi
else
  log "STAGE 7/9 baseline checkpoint not available: baseline metrics will be N/A"
  "$PY" -c "import json; json.dump({'found':False,'path':'$BASELINE_CKPT','reason':'baseline best.pth not found in Google Drive at the configured path'},open('$CONFIGS/baseline_checkpoint.json','w'),indent=2)"
fi

if [ ! -f "$EXPORTS/s_lite/export.json" ]; then
  log "STAGE 8/9 ONNX export: S-Lite"
  "$PY" "$EXP/export_model.py" --arch s --checkpoint "$RUN/best.pth" --data "$MANIFESTS" --size 640 --output "$EXPORTS/s_lite"
fi
if [ -n "$BASELINE_CKPT" ] && [ -f "$BASELINE_CKPT" ] && [ ! -f "$EXPORTS/baseline/export.json" ]; then
  log "STAGE 8/9 ONNX export: baseline (copy written to exports/, original untouched)"
  "$PY" "$EXP/export_model.py" --arch nano --checkpoint "$BASELINE_CKPT" --data "$MANIFESTS" --size 416 --output "$EXPORTS/baseline"
fi

log "STAGE 9/9 comparison report"
"$PY" "$EXP/compare.py" --root "$DRIVE_EXP" --run s_lite_e5_v1
log "PIPELINE COMPLETE"
