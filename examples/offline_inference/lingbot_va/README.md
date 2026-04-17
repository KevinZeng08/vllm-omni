# LingBot-VA Offline I2VA Example

Run LingBot-VA Image-to-Video-Action offline inference in vllm-omni.

```bash
python examples/offline_inference/lingbot_va/end2end.py \
  --model /root/vllm-omni/lingbot-va/lingbot-va-posttrain-robotwin \
  --obs-dir /root/vllm-omni/lingbot-va/example/robotwin \
  --num-chunks 2
```

Outputs:
- `lingbot_i2va.mp4`
- `lingbot_actions.npy`

Observation directory should contain 3 camera images. The script supports both
LingBot canonical names and short names (png/jpg/jpeg):

- `observation.images.cam_high` or `cam_high`
- `observation.images.cam_left_wrist` or `cam_left_wrist`
- `observation.images.cam_right_wrist` or `cam_right_wrist`
