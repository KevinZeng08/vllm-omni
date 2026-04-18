# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LingBot-VA I2VA offline inference example."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers.utils import export_to_video

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


DEFAULT_EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OBS_DIR = DEFAULT_EXAMPLE_DIR / "robotwin_obs"
DEFAULT_PROMPT_FILE = DEFAULT_EXAMPLE_DIR / "robotwin_prompt.txt"
DEFAULT_MODEL = "robbyant/lingbot-va-posttrain-robotwin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LingBot-VA I2VA offline inference")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="LingBot-VA checkpoint path or Hugging Face repo id",
    )
    parser.add_argument(
        "--obs-dir",
        type=str,
        default=str(DEFAULT_OBS_DIR),
        help="Directory containing robotwin camera images",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Task instruction. If not provided, read from --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(DEFAULT_PROMPT_FILE),
        help="Path to prompt text file used when --prompt is not provided.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-chunks", type=int, default=2, help="Number of autoregressive chunks")
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="CFG parallel size (requires matching GPU count when > 1).",
    )
    parser.add_argument("--output-video", type=str, default="lingbot_i2va.mp4", help="Output video path")
    parser.add_argument("--output-actions", type=str, default="lingbot_actions.npy", help="Output actions path")
    return parser.parse_args()


def _resolve_obs_image(obs_dir: Path, canonical_name: str, short_name: str) -> str:
    candidates = [
        f"{canonical_name}.png",
        f"{canonical_name}.jpg",
        f"{canonical_name}.jpeg",
        f"{short_name}.png",
        f"{short_name}.jpg",
        f"{short_name}.jpeg",
    ]
    for candidate in candidates:
        p = obs_dir / candidate
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Could not find image for {canonical_name} in {obs_dir}. Tried: {', '.join(candidates)}"
    )


def build_prompt(obs_dir: Path, prompt: str) -> dict:
    images = {
        "observation.images.cam_high": _resolve_obs_image(
            obs_dir, "observation.images.cam_high", "cam_high"
        ),
        "observation.images.cam_left_wrist": _resolve_obs_image(
            obs_dir, "observation.images.cam_left_wrist", "cam_left_wrist"
        ),
        "observation.images.cam_right_wrist": _resolve_obs_image(
            obs_dir, "observation.images.cam_right_wrist", "cam_right_wrist"
        ),
    }
    return {
        "prompt": prompt,
        "multi_modal_data": {
            "images": images,
        },
    }


def resolve_prompt(prompt: Optional[str], prompt_file: Path) -> str:
    if prompt is not None and prompt.strip():
        return prompt

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    content = prompt_file.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return content


def main() -> None:
    args = parse_args()

    obs_dir = Path(args.obs_dir)
    prompt_text = resolve_prompt(args.prompt, Path(args.prompt_file))
    prompt_data = build_prompt(obs_dir, prompt_text)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    parallel_config = DiffusionParallelConfig(
        cfg_parallel_size=args.cfg_parallel_size,
    )
    omni = Omni(
        model=args.model,
        model_class_name="LingBotVAPipeline",
        parallel_config=parallel_config,
    )

    outputs = omni.generate(
        prompt_data,
        OmniDiffusionSamplingParams(
            generator=generator,
            seed=args.seed,
            guidance_scale=5.0,
            extra_args={"num_chunks_to_infer": args.num_chunks},
        ),
    )

    out = outputs[0]

    video = out.images
    if isinstance(video, list):
        video = video[0]
    video_np = np.asarray(video)
    if video_np.ndim == 5:
        video_np = video_np[0]

    if np.issubdtype(video_np.dtype, np.integer):
        video_np_f = video_np.astype(np.float32) / 255.0
    else:
        video_np_f = video_np.astype(np.float32)

    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(list(video_np_f), str(output_video), fps=10)

    actions = out.multimodal_output.get("actions")
    if actions is None:
        raise RuntimeError("No actions found in multimodal output")
    actions_np = np.asarray(actions)

    output_actions = Path(args.output_actions)
    output_actions.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_actions, actions_np)

    print(f"Video saved to: {output_video}")
    print(f"Actions saved to: {output_actions}")
    print(f"Video shape: {video_np.shape}")
    print(f"Actions shape: {actions_np.shape}")
    print(f"CFG parallel size: {args.cfg_parallel_size}")

    omni.close()


if __name__ == "__main__":
    main()
