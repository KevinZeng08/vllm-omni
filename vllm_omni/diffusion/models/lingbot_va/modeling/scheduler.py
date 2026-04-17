# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LingBot-VA scheduler primitives."""

import math

import torch


class FlowMatchScheduler:
    """Ported from LingBot-VA with minimal adaptation for vllm-omni."""

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
        dynamic_shift_len: int | None = None,
    ) -> None:
        if shift is not None:
            self.shift = shift

        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)

        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])

        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            if mu is None:
                raise ValueError("exponential_shift_mu must be set when exponential_shift=True")
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)

        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)

        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas

        self.timesteps = self.sigmas * self.num_train_timesteps

        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighting = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighting
            self.training = True
        else:
            self.training = False

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor | float, sample: torch.Tensor, to_final: bool = False):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]

        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = self.sigmas[timestep_id + 1]

        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor, t_dim: int = 2):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep = timestep[None]
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def calculate_shift(
        self,
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b
