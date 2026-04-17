# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LingBot-VA persistent state across requests."""

from dataclasses import dataclass

import torch

# TODO: separate kv_cache management and other stateful information as needed.
@dataclass
class LingBotVAState:
    cache_name: str = "pos"
    frame_st_id: int = 0
    init_latent: torch.Tensor | None = None

    def reset(self) -> None:
        self.frame_st_id = 0
        self.init_latent = None
