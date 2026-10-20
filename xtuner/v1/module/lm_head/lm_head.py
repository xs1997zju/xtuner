from typing import Any, Sequence, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from typing_extensions import overload

from xtuner.v1.loss.ce_loss import LMHeadLossContext
from xtuner.v1.utils.profile import attach_l2_backward, profile_range


Loss: TypeAlias = torch.Tensor
Logits: TypeAlias = torch.Tensor
Weight: TypeAlias = torch.Tensor | DTensor
Bias: TypeAlias = torch.Tensor | DTensor | None
HiddenStates: TypeAlias = torch.Tensor
LossHiddenStates: TypeAlias = torch.Tensor | tuple[torch.Tensor, Sequence[torch.Tensor]]
Labels: TypeAlias = torch.Tensor


class LMHead(nn.Linear):
    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: None = None
    ) -> tuple[None, tuple[Logits | None, dict[str, Any]]]: ...

    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: LossHiddenStates, loss_ctx: LMHeadLossContext
    ) -> tuple[Loss, tuple[Logits | None, dict[str, Any]]]: ...

    def forward(  # type: ignore[override]
        self, hidden_states: LossHiddenStates, loss_ctx: LMHeadLossContext | None = None
    ) -> tuple[Loss | None, tuple[Logits | None, dict[str, Any]]]:
        """Forward pass of the language model head."""
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                assert isinstance(self.bias, DTensor), "Bias should be a DTensor if weight is a DTensor"
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        with profile_range(2, "lm_head.forward"):
            if loss_ctx is None:
                assert isinstance(hidden_states, torch.Tensor)
                logits = F.linear(hidden_states, w, b)
            else:
                loss, rest = loss_ctx.forward(hidden_states, w, b)  # type: ignore[arg-type]
        if loss_ctx is None:
            return None, (attach_l2_backward(logits, "lm_head").float(), {})
        return attach_l2_backward(loss, "lm_head"), rest

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: None = None
    ) -> tuple[None, tuple[Logits | None, dict[str, Any]]]: ...

    @overload  # type: ignore
    def __call__(
        self, hidden_states: LossHiddenStates, loss_ctx: LMHeadLossContext
    ) -> tuple[Loss, tuple[Logits | None, dict[str, Any]]]: ...

    __call__ = nn.Module.__call__
