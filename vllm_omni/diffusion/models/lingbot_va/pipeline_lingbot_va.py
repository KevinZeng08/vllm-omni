# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import load_image
from einops import rearrange
from torch import Tensor
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt

from .modeling.lingbot_va_transformer import WanTransformer3DModel
from .modeling.scheduler import FlowMatchScheduler
from .modeling.streaming_vae import WanVAEStreamingWrapper
from .modeling.utils import data_seq_to_patch, get_mesh_id
from .state_lingbot_va import LingBotVAState

logger = logging.getLogger(__name__)


_DEFAULT_ROBOTWIN_Q01 = [
    -0.06172713458538055,
    -3.6716461181640625e-05,
    -0.08783501386642456,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -0.3547105032205582,
    -1.3113021850585938e-06,
    -0.11975435614585876,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
] + [0.0] * 16

_DEFAULT_ROBOTWIN_Q99 = [
    0.3462600058317184,
    0.39966784834861746,
    0.14745532035827624,
    1.0,
    1.0,
    1.0,
    1.0,
    0.034201726913452024,
    0.39142737388610793,
    0.1792279863357542,
    1.0,
    1.0,
    1.0,
    1.0,
] + [0.0] * 14 + [1.0, 1.0]


def _load_transformer_config(model_path: str, local_files_only: bool = True) -> dict[str, Any]:
    config_path = os.path.join(model_path, "transformer", "config.json")
    if local_files_only and os.path.exists(config_path):
        import json

        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_lingbot_va_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(output: Tensor | dict[str, Any]):
        if isinstance(output, dict):
            return output
        return {"video": output}

    return post_process_func


def _to_numpy_rgb(img_obj: Any) -> np.ndarray:
    if isinstance(img_obj, np.ndarray):
        arr = img_obj
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    if isinstance(img_obj, PIL.Image.Image):
        return np.array(img_obj.convert("RGB"))
    if isinstance(img_obj, str):
        return np.array(load_image(img_obj).convert("RGB"))
    raise TypeError(f"Unsupported image object type: {type(img_obj)!r}")


def get_lingbot_va_pre_process_func(od_config: OmniDiffusionConfig):
    model_cfg = od_config.model_config or {}
    obs_cam_keys = model_cfg.get(
        "obs_cam_keys",
        [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
    )

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)

            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            multi_modal_data = prompt.get("multi_modal_data", {})
            obs_payload = prompt["additional_information"].get("obs")
            if obs_payload is None:
                if "obs" in multi_modal_data:
                    obs_payload = multi_modal_data["obs"]
                elif "images" in multi_modal_data and isinstance(multi_modal_data["images"], dict):
                    obs_dict = {
                        key: _to_numpy_rgb(multi_modal_data["images"][key])
                        for key in obs_cam_keys
                        if key in multi_modal_data["images"]
                    }
                    if obs_dict:
                        obs_payload = [obs_dict]
                elif "image" in multi_modal_data:
                    img_arr = _to_numpy_rgb(multi_modal_data["image"])
                    obs_payload = [{k: img_arr for k in obs_cam_keys}]

            if obs_payload is not None:
                if isinstance(obs_payload, dict):
                    obs_payload = [obs_payload]
                prompt["additional_information"]["obs"] = obs_payload

            request.prompts[i] = prompt
        return request

    return pre_process_func


class LingBotVAPipeline(nn.Module, CFGParallelMixin):
    support_image_input = True
    color_format = "RGB"

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.model_config = od_config.model_config or {}
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(self.device)

        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.streaming_vae_half = None

        tf_cfg = _load_transformer_config(model, local_files_only=local_files_only)
        kwargs = {
            "patch_size": tuple(tf_cfg.get("patch_size", (1, 2, 2))),
            "num_attention_heads": tf_cfg.get("num_attention_heads", 24),
            "attention_head_dim": tf_cfg.get("attention_head_dim", 128),
            "in_channels": tf_cfg.get("in_channels", 48),
            "out_channels": tf_cfg.get("out_channels", 48),
            "action_dim": tf_cfg.get("action_dim", self.model_config.get("action_dim", 30)),
            "text_dim": tf_cfg.get("text_dim", 4096),
            "freq_dim": tf_cfg.get("freq_dim", 256),
            "ffn_dim": tf_cfg.get("ffn_dim", 14336),
            "num_layers": tf_cfg.get("num_layers", 30),
            "cross_attn_norm": tf_cfg.get("cross_attn_norm", True),
            "eps": tf_cfg.get("eps", 1e-6),
            "rope_max_seq_len": tf_cfg.get("rope_max_seq_len", 1024),
            "pos_embed_seq_len": tf_cfg.get("pos_embed_seq_len", None),
            "attn_mode": tf_cfg.get("attn_mode", "torch"),
        }
        self.transformer = WanTransformer3DModel(**kwargs).to(self.device, dtype=self.dtype)

        self.scheduler = FlowMatchScheduler(
            shift=self.model_config.get("snr_shift", 5.0), sigma_min=0.0, extra_one_step=True
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=self.model_config.get("action_snr_shift", 1.0), sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.state = LingBotVAState(cache_name=self.model_config.get("cache_name", "pos"))

        self.patch_size = tuple(self.model_config.get("patch_size", (1, 2, 2)))
        self.env_type = self.model_config.get("env_type", "robotwin_tshape")
        self.height = int(self.model_config.get("height", 256))
        self.width = int(self.model_config.get("width", 320))
        self.frame_chunk_size = int(self.model_config.get("frame_chunk_size", 2))
        self.action_per_frame = int(self.model_config.get("action_per_frame", 16))
        self.action_dim = int(self.model_config.get("action_dim", kwargs["action_dim"]))
        self.attn_window = int(self.model_config.get("attn_window", 72))
        self.obs_cam_keys = self.model_config.get(
            "obs_cam_keys",
            [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ],
        )
        self.used_action_channel_ids = self.model_config.get(
            "used_action_channel_ids",
            list(range(0, 7)) + [28] + list(range(7, 14)) + [29],
        )
        inverse_ids = [len(self.used_action_channel_ids)] * self.action_dim
        for i, j in enumerate(self.used_action_channel_ids):
            if j < self.action_dim:
                inverse_ids[j] = i
        self.inverse_used_action_channel_ids = self.model_config.get("inverse_used_action_channel_ids", inverse_ids)
        self.action_norm_method = self.model_config.get("action_norm_method", "quantiles")
        self.norm_stat = self.model_config.get("norm_stat", {"q01": _DEFAULT_ROBOTWIN_Q01, "q99": _DEFAULT_ROBOTWIN_Q99})

        self.guidance_scale = float(self.model_config.get("guidance_scale", 5.0))
        self.action_guidance_scale = float(self.model_config.get("action_guidance_scale", 1.0))
        self.num_inference_steps = int(self.model_config.get("num_inference_steps", 25))
        self.action_num_inference_steps = int(self.model_config.get("action_num_inference_steps", 50))
        self.video_exec_step = int(self.model_config.get("video_exec_step", -1))
        self.num_chunks_to_infer = int(self.model_config.get("num_chunks_to_infer", 10))

        self.video_processor = VideoProcessor(vae_scale_factor=1)

        if self.env_type == "robotwin_tshape":
            vae_half = DistributedAutoencoderKLWan.from_pretrained(
                model,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=local_files_only,
            ).to(self.device)
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def _get_t5_prompt_embeds(
        self,
        prompt,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(p) if isinstance(p, str) else p for p in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), mask.to(self.device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(self, latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action: np.ndarray):
        action_model_input = torch.from_numpy(action)
        action_model_input_paded = F.pad(action_model_input, [0, 0, 0, 0, 0, 1], mode="constant", value=0)
        action_model_input = action_model_input_paded[self.inverse_used_action_channel_ids]
        if self.action_norm_method != "quantiles":
            raise NotImplementedError(f"Unsupported action norm method: {self.action_norm_method}")
        action_model_input = (action_model_input - self.actions_q01) / (self.actions_q99 - self.actions_q01 + 1e-6) * 2.0 - 1.0
        return action_model_input.unsqueeze(0).unsqueeze(-1)

    def postprocess_action(self, action: torch.Tensor):
        action = action.cpu()[0, ..., 0]
        if self.action_norm_method != "quantiles":
            raise NotImplementedError(f"Unsupported action norm method: {self.action_norm_method}")
        action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 + 1e-6) + self.actions_q01
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.used_action_channel_ids]

    def _repeat_input_for_cfg(self, input_dict: dict[str, Any]):
        out = dict(input_dict)
        if self.use_cfg:
            out["noisy_latents"] = out["noisy_latents"].repeat(2, 1, 1, 1, 1)
            out["text_emb"] = torch.cat(
                [self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()],
                dim=0,
            )
            out["grid_id"] = out["grid_id"][None].repeat(2, 1, 1)
            out["timesteps"] = out["timesteps"][None].repeat(2, 1)
        else:
            out["grid_id"] = out["grid_id"][None]
            out["timesteps"] = out["timesteps"][None]
        return out

    def _prepare_latent_input(
        self,
        latent_model_input,
        action_model_input,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
    ):
        input_dict = {}
        if latent_model_input is not None:
            input_dict["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=self.device) * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // self.patch_size[0],
                    latent_model_input.shape[-2] // self.patch_size[1],
                    latent_model_input.shape[-1] // self.patch_size[2],
                    0,
                    1,
                    frame_st_id,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_model_input is not None:
            input_dict["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=self.device) * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                input_dict["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict["action_res_lst"]["timesteps"][0:1] *= 0
            input_dict["action_res_lst"]["noisy_latents"][:, ~self.action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs: list[dict[str, np.ndarray]]):
        images = obs if isinstance(obs, list) else [obs]
        if len(images) < 1:
            return None

        videos = []
        for k_i, key in enumerate(self.obs_cam_keys):
            if self.env_type == "robotwin_tshape":
                if k_i == 0:
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(np.stack([each[key] for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(
                history_video_k,
                size=(height_i, width_i),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == "robotwin_tshape" and self.streaming_vae_half is not None:
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            vae_dtype = next(self.streaming_vae.vae.parameters()).dtype
            enc_out_high = self.streaming_vae.encode_chunk(videos_high.to(vae_device).to(vae_dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(vae_dtype)
            )
            enc_out = torch.cat(
                [torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1), enc_out_high],
                dim=-2,
            )
        else:
            videos_cat = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            vae_dtype = next(self.streaming_vae.vae.parameters()).dtype
            enc_out = self.streaming_vae.encode_chunk(videos_cat.to(vae_device).to(vae_dtype))

        mu, _ = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt: str):
        self.state.reset()
        self.transformer.clear_cache(self.state.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half is not None:
            self.streaming_vae_half.clear_cache()

        self.use_cfg = (self.guidance_scale > 1) or (self.action_guidance_scale > 1)

        if self.env_type == "robotwin_tshape":
            self.latent_height, self.latent_width = ((self.height // 16) * 3) // 2, self.width // 16
        else:
            self.latent_height, self.latent_width = self.height // 16, (self.width // 16) * len(self.obs_cam_keys)

        latent_token_per_chunk = (
            self.frame_chunk_size * self.latent_height * self.latent_width
        ) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        action_token_per_chunk = self.frame_chunk_size * self.action_per_frame

        self.transformer.create_empty_cache(
            self.state.cache_name,
            self.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=self.dtype,
            device=self.device,
            batch_size=2 if self.use_cfg else 1,
        )

        self.action_mask = torch.zeros([self.action_dim]).bool()
        self.action_mask[self.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.norm_stat["q01"], dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.norm_stat["q99"], dtype=torch.float32).reshape(-1, 1, 1)

        self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=self.guidance_scale > 1,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=self.device,
            dtype=self.dtype,
        )

    def _infer_one_chunk(self, obs: list[dict[str, np.ndarray]], frame_st_id: int, generator: torch.Generator):
        if frame_st_id == 0:
            self.state.init_latent = self._encode_obs(obs)

        latents = randn_tensor(
            (
                1,
                48,
                self.frame_chunk_size,
                self.latent_height,
                self.latent_width,
            ),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        actions = randn_tensor(
            (
                1,
                self.action_dim,
                self.frame_chunk_size,
                self.action_per_frame,
                1,
            ),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        self.scheduler.set_timesteps(self.num_inference_steps)
        self.action_scheduler.set_timesteps(self.action_num_inference_steps)

        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        if self.video_exec_step != -1:
            timesteps = timesteps[: self.video_exec_step]

        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                last_step = i == len(timesteps) - 1
                latent_cond = self.state.init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id,
                )
                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.state.cache_name,
                    action_mode=False,
                )

                if not last_step or self.video_exec_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.patch_size,
                        video_noise_pred,
                        self.frame_chunk_size,
                        self.latent_height,
                        self.latent_width,
                        batch_size=2 if self.use_cfg else 1,
                    )
                    if self.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.guidance_scale * (
                            video_noise_pred[:1] - video_noise_pred[1:]
                        )
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred, t, latents)
                if frame_st_id == 0 and latent_cond is not None:
                    latents[:, :, 0:1] = latent_cond

            for i, t in enumerate(action_timesteps):
                last_step = i == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [1, self.action_dim, 1, self.action_per_frame, 1],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    if frame_st_id == 0
                    else None
                )

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id,
                )
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.state.cache_name,
                    action_mode=True,
                )

                if not last_step:
                    action_noise_pred = rearrange(
                        action_noise_pred,
                        "b (f n) c -> b c f n 1",
                        f=self.frame_chunk_size,
                    )
                    if self.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.action_guidance_scale * (
                            action_noise_pred[:1] - action_noise_pred[1:]
                        )
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred, t, actions)
                if frame_st_id == 0 and action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~self.action_mask] *= 0
        actions_np = self.postprocess_action(actions)
        return actions_np, latents

    def decode_one_video(self, latents: torch.Tensor, output_type: str = "np"):
        latents = latents.to(self.vae.dtype)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device,
            latents.dtype,
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device,
            latents.dtype,
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        return self.video_processor.postprocess_video(video, output_type=output_type)

    def _resolve_obs(self, prompt_obj: Any, req: OmniDiffusionRequest):
        if isinstance(prompt_obj, str):
            return None
        additional = prompt_obj.get("additional_information", {})
        if "obs" in additional:
            return additional["obs"]
        extra_args = req.sampling_params.extra_args or {}
        return extra_args.get("obs")

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("LingBotVAPipeline currently supports a single prompt per request")

        prompt_obj = req.prompts[0]
        prompt_text = prompt_obj if isinstance(prompt_obj, str) else prompt_obj.get("prompt", "")
        obs = self._resolve_obs(prompt_obj, req)
        if obs is None:
            raise ValueError("LingBotVAPipeline requires observation images in prompt.additional_information['obs']")

        if isinstance(obs, dict):
            obs = [obs]

        extra_args = req.sampling_params.extra_args or {}
        num_chunks = int(extra_args.get("num_chunks_to_infer", self.num_chunks_to_infer))

        if req.sampling_params.guidance_scale_provided:
            self.guidance_scale = req.sampling_params.guidance_scale

        seed = req.sampling_params.seed if req.sampling_params.seed is not None else 42
        generator = req.sampling_params.generator
        if generator is None:
            generator = torch.Generator(device=self.device.type).manual_seed(seed)

        self._reset(prompt_text)

        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(num_chunks):
            frame_st_id = chunk_id * self.frame_chunk_size
            actions_np, latents = self._infer_one_chunk(obs, frame_st_id=frame_st_id, generator=generator)
            pred_latent_lst.append(latents)
            pred_action_lst.append(torch.from_numpy(actions_np))

        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        actions_seq = pred_action.transpose(0, 1).cpu().numpy()

        self.transformer.clear_cache(self.state.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half is not None:
            self.streaming_vae_half.clear_cache()

        decoded_video = self.decode_one_video(pred_latent, output_type="np")[0]

        return DiffusionOutput(
            output={"video": decoded_video, "actions": actions_seq},
            custom_output={"latents": pred_latent.detach().cpu()},
        )
