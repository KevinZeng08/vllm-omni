#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-robbyant/lingbot-va-posttrain-robotwin}
OBS_DIR=${OBS_DIR:-robotwin_obs}
PROMPT_FILE=${PROMPT_FILE:-robotwin_prompt.txt}
NUM_CHUNKS=${NUM_CHUNKS:-10}
CFG_PARALLEL_SIZE=${CFG_PARALLEL_SIZE:-1}
SEED=${SEED:-42}
OUT_VIDEO=${OUT_VIDEO:-vllm_robotwin_i2va.mp4}
OUT_ACTIONS=${OUT_ACTIONS:-vllm_robotwin_i2va_actions.npy}

PROMPT_ARGS=(--prompt-file "$PROMPT_FILE")
if [[ -n "${PROMPT:-}" ]]; then
  PROMPT_ARGS=(--prompt "$PROMPT")
fi

python end2end.py \
  --model "$MODEL_PATH" \
  --obs-dir "$OBS_DIR" \
  "${PROMPT_ARGS[@]}" \
  --num-chunks "$NUM_CHUNKS" \
  --cfg-parallel-size "$CFG_PARALLEL_SIZE" \
  --seed "$SEED" \
  --output-video "$OUT_VIDEO" \
  --output-actions "$OUT_ACTIONS"

echo "Done."
echo "Video: $OUT_VIDEO"
echo "Actions: $OUT_ACTIONS"
