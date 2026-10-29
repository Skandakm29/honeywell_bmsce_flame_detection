#!/usr/bin/env bash
# Whole experiment as one resumable background job on the Colab GPU:
#   manifests -> smoke tests -> batch size from measured memory -> training ->
#   curves -> full-validation evaluation (threshold selection) -> final test -> report.
# Every stage is skipped when its result already exists on Drive, so relaunching after a
# disconnect continues where it stopped. Any failure stops the job with the error in the log.
set -euo pipefail
export MPLBACKEND=Agg
: "${PY:?}" "${REPO:?}" "${KAGGLE:?}" "${DRIVE_EXP:?}" "${CHECKPOINT:?}"
EXP="$REPO/experiments/yolox_s_lite_flame"
MANIFESTS="$DRIVE_EXP/manifests_v2"
SMOKE="$DRIVE_EXP/smoke_v2"
RUN="$DRIVE_EXP/runs/full_v1"
cd "$REPO"
mkdir -p "$SMOKE" "$RUN"
log() { echo "[$(date '+%F %T')] $*"; }

if [ ! -f "$MANIFESTS/prepare_report.json" ]; then
  log "STAGE 1/7 manifests from the Kaggle splits"
  "$PY" "$EXP/prepare_kaggle.py" --root "$KAGGLE" --output "$MANIFESTS" --workers 16
fi

if [ ! -f "$SMOKE/PASSED" ]; then
  log "STAGE 2/7 smoke test: weights, labels, loss, gradients, save/reload, output shape"
  "$PY" "$EXP/smoke_test.py" --data "$MANIFESTS" --checkpoint "$CHECKPOINT" --output "$SMOKE" --batch 8 --device cuda
  LOOP="$SMOKE/loop_$(date +%Y%m%d_%H%M%S)"
  COMMON=(--config "$EXP/configs/smoke.json" --data "$MANIFESTS/smoke" --checkpoint "$CHECKPOINT" --output "$LOOP" --device cuda)
  log "STAGE 2/7 smoke training loop: interrupt after 2 updates, resume, finish, export, evaluate"
  # The deliberate early stop can make data-loader workers abort on exit; success is judged by last.pth.
  "$PY" "$EXP/train_s.py" "${COMMON[@]}" --stop-after-updates 2 || true
  [ -f "$LOOP/last.pth" ] || { log "SMOKE FAILED: no checkpoint after interruption"; exit 1; }
  "$PY" "$EXP/train_s.py" "${COMMON[@]}" --resume "$LOOP/last.pth"
  "$PY" "$EXP/evaluate_s.py" --arch s --checkpoint "$LOOP/best.pth" --data "$MANIFESTS/smoke" --split val --size 640 \
    --output "$LOOP/eval_val" --latency-images 20 --review 8
  for f in best.pth last.pth export.json eval_val/metrics.json eval_val/curves.png; do
    [ -f "$LOOP/$f" ] || { log "SMOKE FAILED: missing $f"; exit 1; }
  done
  "$PY" -c "import json,sys; r=json.load(open('$SMOKE/smoke_report.json')); sys.exit(0 if r.get('passed') else 1)"
  echo "$LOOP" > "$SMOKE/PASSED"
  log "SMOKE PASSED (ONNX parity: $("$PY" -c "import json; print(json.load(open('$LOOP/export.json')).get('parity_passed'))"))"
fi

if [ ! -f "$RUN/run_settings.json" ]; then
  log "STAGE 3/7 batch size from measured smoke-test memory"
  "$PY" - "$SMOKE/smoke_report.json" "$RUN/run_settings.json" <<'EOF'
import json, sys, torch
report, target = sys.argv[1], sys.argv[2]
peak8 = json.load(open(report))['speed']['peak_gpu_memory_MiB']
total = torch.cuda.get_device_properties(0).total_memory/2**20
batch, accumulate = (16,1) if 2*peak8 <= .75*total else (8,2)
settings = {'batch':batch,'accumulate':accumulate,'peak_MiB_at_batch_8':peak8,'gpu_total_MiB':total,
            'gpu':torch.cuda.get_device_name(0),
            'rule':'16x1 if 2*peak(batch 8) <= 0.75*GPU memory, else 8x2; effective batch 16 either way'}
json.dump(settings, open(target,'w'), indent=2)
print(settings)
EOF
fi
BATCH=$("$PY" -c "import json; print(json.load(open('$RUN/run_settings.json'))['batch'])")
ACCUMULATE=$("$PY" -c "import json; print(json.load(open('$RUN/run_settings.json'))['accumulate'])")

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 60 >> "$RUN/gpu_usage.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

if [ ! -f "$RUN/export.json" ]; then
  RESUME=()
  [ -f "$RUN/last.pth" ] && RESUME=(--resume "$RUN/last.pth")
  log "STAGE 4/7 training: batch $BATCH x accumulate $ACCUMULATE ${RESUME[*]:-(fresh start)}"
  "$PY" "$EXP/train_s.py" --config "$EXP/configs/full_v1.json" --data "$MANIFESTS" --checkpoint "$CHECKPOINT" \
    --output "$RUN" --batch "$BATCH" --accumulate "$ACCUMULATE" "${RESUME[@]}"
fi
log "STAGE 5/7 training curves"
"$PY" "$EXP/plot_history.py" --run "$RUN"
if [ ! -f "$RUN/eval_validation_full/metrics.json" ]; then
  log "STAGE 6/7 full validation evaluation and threshold selection"
  "$PY" "$EXP/evaluate_s.py" --arch s --checkpoint "$RUN/best.pth" --data "$MANIFESTS" --split validation_full \
    --size 640 --output "$RUN/eval_validation_full"
fi
if [ ! -f "$RUN/eval_test/metrics.json" ]; then
  log "STAGE 7/7 final test with the validation-selected threshold"
  "$PY" "$EXP/evaluate_s.py" --arch s --checkpoint "$RUN/best.pth" --data "$MANIFESTS" --split test --size 640 \
    --output "$RUN/eval_test" --final-test --threshold-from "$RUN/eval_validation_full/metrics.json"
fi
"$PY" "$EXP/make_report.py" --run "$RUN" --manifests "$MANIFESTS" --smoke-report "$SMOKE/smoke_report.json"
cp "$RUN/report.md" "$DRIVE_EXP/report.md"
log "PIPELINE COMPLETE"
