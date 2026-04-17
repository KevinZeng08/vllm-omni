#!/usr/bin/env bash
set -euo pipefail

cd /root/vllm-omni
source /root/vllm-omni/.venv/bin/activate

MODEL_PATH=${MODEL_PATH:-/root/vllm-omni/lingbot-va/lingbot-va-posttrain-robotwin}
OBS_DIR=${OBS_DIR:-examples/offline_inference/lingbot_va/robotwin_obs}
PROMPT_FILE=${PROMPT_FILE:-examples/offline_inference/lingbot_va/robotwin_prompt.txt}
NUM_CHUNKS=${NUM_CHUNKS:-10}
SEED=${SEED:-42}
OUT_VIDEO=${OUT_VIDEO:-examples/offline_inference/lingbot_va/vllm_robotwin_i2va.mp4}
OUT_ACTIONS=${OUT_ACTIONS:-examples/offline_inference/lingbot_va/vllm_robotwin_i2va_actions.npy}

PROMPT_ARGS=(--prompt-file "$PROMPT_FILE")
if [[ -n "${PROMPT:-}" ]]; then
  PROMPT_ARGS=(--prompt "$PROMPT")
fi

if [[ "$MODEL_PATH" == /* ]] && [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: local MODEL_PATH does not exist: $MODEL_PATH" >&2
  echo "Hint: set MODEL_PATH to a valid local checkpoint directory or a HF repo id (e.g. owner/repo)." >&2
  exit 1
fi

/root/vllm-omni/.venv/bin/python examples/offline_inference/lingbot_va/end2end.py \
  --model "$MODEL_PATH" \
  --obs-dir "$OBS_DIR" \
  "${PROMPT_ARGS[@]}" \
  --num-chunks "$NUM_CHUNKS" \
  --seed "$SEED" \
  --output-video "$OUT_VIDEO" \
  --output-actions "$OUT_ACTIONS"

echo "Done."
echo "Video: $OUT_VIDEO"
echo "Actions: $OUT_ACTIONS"
