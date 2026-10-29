#!/usr/bin/env bash
# Continuation of the finished 5-epoch YOLOX-S-Lite run: 15 more epochs at the same learning rate
# (3e-4, warmup 200, cosine to 0.1x; the schedule restarts), starting from checkpoints/s_lite_e5_v1/best.pth, then every
# model is reported at confidence 0.40 fixed in advance (not tuned on any split):
#   1 checks  2 training (resume-aware)  3 curves  4 S-Lite 5+15 val/test  5 S-Lite e5 + baseline
#   val/test at 0.40  6 ONNX export  7 comparison report (reports_e20/; reports/ is left as it is)
# Each stage is skipped when its result already exists on Drive; any failure stops the job and
# leaves the error in logs/pipeline_e5p15.log. The 5-epoch run and the baseline are only read.
set -euo pipefail
export MPLBACKEND=Agg
: "${PY:?}" "${REPO:?}" "${DRIVE_EXP:?}" "${BASELINE_CKPT:?}"
EXP="$REPO/experiments/yolox_s_lite_flame"
MANIFESTS="$DRIVE_EXP/manifests_v2"
CONFIGS="$DRIVE_EXP/configs"; LOGS="$DRIVE_EXP/logs/e5p15"
E5="$DRIVE_EXP/checkpoints/s_lite_e5_v1"; RUN="$DRIVE_EXP/checkpoints/s_lite_e5p15_v1"
VAL="$DRIVE_EXP/validation"; TEST="$DRIVE_EXP/test"; PLOTS="$DRIVE_EXP/plots"; EXPORTS="$DRIVE_EXP/exports"
CONFIDENCE=0.40
mkdir -p "$CONFIGS" "$LOGS" "$RUN" "$VAL" "$TEST" "$PLOTS" "$EXPORTS"
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }

log "STAGE 1/7 checks: 5-epoch S-Lite best.pth and baseline best.pth load strictly"
"$PY" "$EXP/verify_checkpoint.py" --path "$E5/best.pth" --arch s --kind weights --record "$CONFIGS/e5p15_init_checkpoint.json" > /dev/null
"$PY" "$EXP/verify_checkpoint.py" --path "$BASELINE_CKPT" --arch nano --kind weights > /dev/null
cp "$EXP/configs/e5p15_v1.json" "$CONFIGS/e5p15_v1.json"

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 60 >> "$LOGS/gpu_usage.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

if [ ! -f "$RUN/export.json" ]; then
  RESUME=()
  if [ -f "$RUN/last.pth" ]; then
    log "STAGE 2/7 verifying existing checkpoint before resuming (never restart from the 5-epoch weights)"
    "$PY" "$EXP/verify_checkpoint.py" --path "$RUN/last.pth" --arch s --kind train --record "$LOGS/resume_checkpoint_check.json"
    RESUME=(--resume "$RUN/last.pth")
  fi
  log "STAGE 2/7 training 15 more epochs, lr 3e-4 (same as the 5-epoch run): ${RESUME[*]:-(start from $E5/best.pth)}"
  INIT_SOURCE="continuation from $E5/best.pth (5-epoch run s_lite_e5_v1)" STATUS_DIR="$LOGS" \
    "$PY" "$EXP/train_s.py" --config "$EXP/configs/e5p15_v1.json" --data "$MANIFESTS" --checkpoint "$E5/best.pth" \
    --output "$RUN" "${RESUME[@]}" 2>&1 | tee -a "$LOGS/train.log"
fi
"$PY" "$EXP/verify_checkpoint.py" --path "$RUN/best.pth" --arch s --kind weights --record "$LOGS/best_checkpoint.json" > /dev/null
log "checkpoint verified: $RUN/best.pth"

log "STAGE 3/7 training curves"
mkdir -p "$PLOTS/e5p15"
"$PY" "$EXP/plot_history.py" --run "$RUN" --plots "$PLOTS/e5p15"

evaluate() {  # name arch checkpoint size
  if [ ! -f "$VAL/$1/metrics.json" ]; then
    log "evaluating $1 on full validation at confidence $CONFIDENCE"
    "$PY" "$EXP/evaluate_s.py" --arch "$2" --checkpoint "$3" --data "$MANIFESTS" --split validation_full --size "$4" \
      --output "$VAL/$1" --confidence "$CONFIDENCE"
  fi
  if [ ! -f "$TEST/$1/metrics.json" ]; then
    log "evaluating $1 on test at confidence $CONFIDENCE (final test)"
    "$PY" "$EXP/evaluate_s.py" --arch "$2" --checkpoint "$3" --data "$MANIFESTS" --split test --size "$4" \
      --output "$TEST/$1" --final-test --confidence "$CONFIDENCE"
  fi
}
log "STAGE 4/7 S-Lite 5+15 epochs"
evaluate s_lite_e20_c040 s "$RUN/best.pth" 640
log "STAGE 5/7 S-Lite 5 epochs and baseline at the same confidence"
evaluate s_lite_e5_c040 s "$E5/best.pth" 640
evaluate baseline_c040 nano "$BASELINE_CKPT" 416

if [ ! -f "$EXPORTS/s_lite_e20/export.json" ]; then
  log "STAGE 6/7 ONNX export: S-Lite 5+15"
  "$PY" "$EXP/export_model.py" --arch s --checkpoint "$RUN/best.pth" --data "$MANIFESTS" --size 640 --output "$EXPORTS/s_lite_e20"
fi

log "STAGE 7/7 comparison report"
"$PY" "$EXP/compare.py" --root "$DRIVE_EXP" --run s_lite_e5_v1 --run s_lite_e5p15_v1 \
  --model baseline_c040 baseline "YOLOX-Nano-ti-lite (baseline, 416, 20 ep)" \
  --model s_lite_e5_c040 s_lite "YOLOX-S-ti-lite (640, 5 ep)" \
  --model s_lite_e20_c040 s_lite_e20 "YOLOX-S-ti-lite (640, 5+15 ep)" \
  --reports reports_e20 --plot-prefix comparison_e20 \
  --title "Flame detection: baseline vs YOLOX-S-ti-lite after 5 and 5+15 epochs (confidence 0.40)" \
  --threshold-rule "Every model uses confidence **0.40**, fixed in advance (the baseline's reported operating point); it was not tuned on validation or test. Test was evaluated once per model." \
  --limitation "S-Lite 5+15 is a continuation: 5 epochs (lr 3e-4) then 15 epochs from that best.pth with the same lr 3e-4 (schedule restarted: warmup 200, cosine to 0.1x), a fresh optimizer and EMA. It is not a single 20-epoch schedule." \
  --limitation "Early stopping (patience 5 epochs on monitor AP50:95) may end the continuation before 15 epochs; the epochs completed are listed below." \
  --limitation "The training loop disables augmentation for its last two epochs (inherited from the baseline recipe)." \
  --limitation "Per-epoch monitor precision/recall in history.json use each epoch's max-F1 confidence (train.py, unchanged), not 0.40; AP is threshold-free."
log "PIPELINE E5P15 COMPLETE"
