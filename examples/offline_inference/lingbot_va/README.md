# LingBot-VA Offline I2VA Example

Run LingBot-VA Image-to-Video-Action offline inference in vllm-omni.

## 1) Download Model

Model: `robbyant/lingbot-va-posttrain-robotwin`

```bash
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin
```

## 2) Example Assets

For reproducibility, this example keeps the default obs and prompt next to the
python script:

- `examples/offline_inference/lingbot_va/robotwin_obs/`
- `examples/offline_inference/lingbot_va/robotwin_prompt.txt`

Observation directory should contain 3 camera images. The script supports both
LingBot canonical names and short names (png/jpg/jpeg):

- `observation.images.cam_high` or `cam_high`
- `observation.images.cam_left_wrist` or `cam_left_wrist`
- `observation.images.cam_right_wrist` or `cam_right_wrist`

## 3) Run

```bash
cd examples/offline_inference/lingbot_va/
python end2end.py \
  --obs-dir robotwin_obs \
  --prompt-file robotwin_prompt.txt \
  --num-chunks 10
```

or use the helper script:

```bash
cd examples/offline_inference/lingbot_va/
bash run_vllm_omni_lingbot_i2va.sh
```

Outputs:
- `lingbot_i2va.mp4`
- `lingbot_actions.npy`
