# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner.v1.ops import rms_norm, zero_centered_rms_norm
from xtuner.v1.utils.profile import attach_l2_backward, profile_range


class RMSNorm(nn.Module):
    """RMSNorm is equivalent to T5LayerNorm.

    ``l2_name`` records ``xtuner.L2.{l2_name}.*`` for the model-level final
    norm only. Per-layer RMSNorms leave it unset.
    """

    weight: torch.Tensor
    zero_centered: bool

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        type: Literal["default", "zero_centered"] = "default",
        l2_name: str | None = None,
    ):
        super().__init__()

        self.zero_centered = type == "zero_centered"
        # Zero-centered RMSNorm (e.g. HF Qwen3NextRMSNorm) computes x * (1 + weight), so the
        # raw weight must start at 0 to make the effective scale start at 1.
        init_value = 0.0 if self.zero_centered else 1.0
        self.weight = nn.Parameter(torch.full((hidden_size,), init_value))
        self.variance_epsilon = eps
        self._type = type
        self._l2_name = l2_name

        if type == "default":
            self.rms_norm_fn = rms_norm
        elif type == "zero_centered":
            self.rms_norm_fn = zero_centered_rms_norm
        else:
            raise ValueError(f"Unsupported RMSNorm type: {type}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        if self._l2_name is None:
            return self.rms_norm_fn(hidden_states, weight, epsilon=self.variance_epsilon)  # type: ignore[operator]

        # Inside the module so FSDP unshard stays outside the L2 range.
        with profile_range(2, f"{self._l2_name}.forward"):
            hidden_states = self.rms_norm_fn(hidden_states, weight, epsilon=self.variance_epsilon)  # type: ignore[operator]
        return attach_l2_backward(hidden_states, self._l2_name)

    def init_weights(self):
        init_value = 0.0 if self.zero_centered else 1.0
        self.weight.data.fill_(init_value)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, type={self._type}, eps={self.variance_epsilon}"
