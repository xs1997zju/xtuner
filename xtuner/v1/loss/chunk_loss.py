from typing import Any, Callable

import torch
from torch.autograd import grad

from xtuner.v1.utils.profile import profile_range


class ChunkLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_forward: Callable,
        loss_kwargs_chunks: list[Any],
        chunk_size: int,
    ):
        # ChunkLoss is a `torch.autograd.Function`; without an explicit range
        # it is recorded as an opaque `cpu_op` and its GPU work is unattributed.
        with profile_range(2, "loss.chunk_forward"):
            if head_bias is not None:
                raise NotImplementedError("head_bias is not supported in ChunkLoss")

            device = hidden_states.device
            accumulated_loss = torch.tensor(0.0, device=device)
            grad_inputs = torch.empty_like(hidden_states)
            weight_requires_grad = head_weight.requires_grad
            grad_weight = torch.zeros_like(head_weight) if weight_requires_grad else None

            grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
            hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)

            from xtuner.v1.model.utils import ModelForwardExtraLogInfo

            chunked_extra_info = ModelForwardExtraLogInfo()
            for i in range(len(hidden_states_chunks)):
                hidden_states_chunk = hidden_states_chunks[i]
                grad_inputs_chunk = grad_inputs_chunks[i]

                with torch.enable_grad():
                    hidden_states_chunk.requires_grad_()
                    chunk_loss, (_, extra_info) = loss_forward(
                        hidden_states_chunk, head_weight, None, loss_kwargs_chunks[i]
                    )
                    if weight_requires_grad:
                        chunk_grad_input, chunk_grad_weight = grad(
                            chunk_loss, (hidden_states_chunk, head_weight), allow_unused=True
                        )
                    else:
                        chunk_grad_input = grad(chunk_loss, (hidden_states_chunk,), allow_unused=True)[0]

                accumulated_loss.add_(chunk_loss)
                grad_inputs_chunk.copy_(chunk_grad_input)
                if grad_weight is not None:
                    grad_weight.add_(chunk_grad_weight)

                chunked_extra_info.append(extra_info)

            ctx.weight_requires_grad = weight_requires_grad
            if grad_weight is None:
                ctx.save_for_backward(grad_inputs)
            else:
                ctx.save_for_backward(grad_inputs, grad_weight)
            return accumulated_loss, chunked_extra_info

    @staticmethod
    def backward(ctx, *grad_output):
        # The autograd engine runs this on a worker thread; the `xtuner.`
        # prefix lets profile_speed.py attribute GPU events from that thread.
        with profile_range(2, "loss.chunk_backward"):
            grad_input = ctx.saved_tensors[0]
            grad_weight = ctx.saved_tensors[1] if ctx.weight_requires_grad else None
            if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
                grad_input = grad_input * grad_output[0]
                if grad_weight is not None:
                    grad_weight = grad_weight * grad_output[0]

            return grad_input, grad_weight, None, None, None, None
