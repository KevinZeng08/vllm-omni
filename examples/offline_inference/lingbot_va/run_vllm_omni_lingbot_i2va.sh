#!/usr/bin/env bash
set -euo pipefail

cd /root/vllm-omni
source /root/vllm-omni/.venv/bin/activate

MODEL_PATH=${MODEL_PATH:-/root/vllm-omni/lingbot-va/lingbot-va-posttrain-robotwin}
OBS_DIR=${OBS_DIR:-/root/kevinzeng/lingbot-va/example/robotwin}
PROMPT=${PROMPT:-"Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack."}
NUM_CHUNKS=${NUM_CHUNKS:-10}
SEED=${SEED:-42}
OUT_VIDEO=${OUT_VIDEO:-examples/offline_inference/lingbot_va/vllm_robotwin_i2va.mp4}
OUT_ACTIONS=${OUT_ACTIONS:-examples/offline_inference/lingbot_va/vllm_robotwin_i2va_actions.npy}

/root/vllm-omni/.venv/bin/python examples/offline_inference/lingbot_va/end2end.py \
  --model "$MODEL_PATH" \
  --obs-dir "$OBS_DIR" \
  --prompt "$PROMPT" \
  --num-chunks "$NUM_CHUNKS" \
  --seed "$SEED" \
  --output-video "$OUT_VIDEO" \
  --output-actions "$OUT_ACTIONS"

echo "Done."
echo "Video: $OUT_VIDEO"
echo "Actions: $OUT_ACTIONS"
