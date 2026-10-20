import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, cast, overload

import torch
from torch import Tensor, nn
from torch.autograd import Function

from xtuner.v1.utils.device import get_torch_device_module
from xtuner.v1.utils.logger import get_logger


logger = get_logger()

# Open/close pairs for layer backward ranges must run on the same autograd thread.
_BW_RANGE_STACK: ContextVar[list[AbstractContextManager[Any]] | None] = ContextVar(
    "xtuner_profile_bw_stack", default=None
)


@torch.compiler.disable(recursive=False)
def layer_profile_name(layer_idx: int) -> str:
    """Return ``layer{idx}.forward`` or ``layer{idx}.recompute`` during
    checkpoint replay."""
    from xtuner.v1.model.utils.checkpointing import is_checkpoint_replay

    phase = "recompute" if is_checkpoint_replay() else "forward"
    return f"layer{layer_idx + 1}.{phase}"


@torch.compiler.disable(recursive=False)
def mtp_profile_name(mtp_idx: int) -> str:
    """Return ``mtp{idx}.forward`` or ``mtp{idx}.recompute`` during checkpoint
    replay."""
    from xtuner.v1.model.utils.checkpointing import is_checkpoint_replay

    phase = "recompute" if is_checkpoint_replay() else "forward"
    return f"mtp{mtp_idx + 1}.{phase}"


def profile_range(level: int, name: str):
    """Return a profiler range named ``xtuner.L{level}.{name}``.

    ``profile_speed.py`` groups these ranges by the ``xtuner.L{{n}}.`` prefix.
    Level 1 is a training phase, level 2 is a layer or loss block, level 3 is a
    finer block inside a layer. Nested calls should follow that order.
    """
    return torch.profiler.record_function(f"xtuner.L{level}.{name}")


@torch.compiler.disable(recursive=False)
def profiled_item(tensor: Tensor, name: str = "item") -> Any:
    """Call ``tensor.item()`` under ``xtuner.L3.host_sync.{name}``.

    ``.item()`` is a device-to-host sync. Without a range the profiler records
    only ``aten::_local_scalar_dense`` with no parent annotation.
    """
    with profile_range(3, f"host_sync.{name}"):
        return tensor.item()


class _ProfileBwOpen(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, name: str) -> torch.Tensor:
        ctx.name = name
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        cm = torch.profiler.record_function(f"xtuner.L2.{ctx.name}.backward")
        cm.__enter__()
        stack = _BW_RANGE_STACK.get()
        if stack is None:
            stack = []
            _BW_RANGE_STACK.set(stack)
        stack.append(cm)
        return grad, None


class _ProfileBwClose(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        stack = _BW_RANGE_STACK.get()
        if stack:
            stack.pop().__exit__(None, None, None)
        return grad


@overload
def profile_bw_close(hidden: torch.Tensor) -> torch.Tensor: ...


@overload
def profile_bw_close(hidden: list[torch.Tensor]) -> list[torch.Tensor]: ...


def profile_bw_close(hidden: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    """Mark layer inputs so backward can close the L2.backward range."""
    if isinstance(hidden, list):
        return [_ProfileBwClose.apply(tensor) for tensor in hidden]
    return _ProfileBwClose.apply(hidden)


@overload
def profile_bw_open(hidden: torch.Tensor, name: str) -> torch.Tensor: ...


@overload
def profile_bw_open(hidden: list[torch.Tensor], name: str) -> list[torch.Tensor]: ...


def profile_bw_open(hidden: torch.Tensor | list[torch.Tensor], name: str) -> torch.Tensor | list[torch.Tensor]:
    """Mark layer outputs so backward can open
    ``xtuner.L2.{name}.backward``."""
    if isinstance(hidden, list):
        return [_ProfileBwOpen.apply(tensor, name) for tensor in hidden]
    return _ProfileBwOpen.apply(hidden, name)


# Separate from decoder-layer bw stack so module hooks never pop layer ranges.
_MODULE_BW_STACK: ContextVar[list[AbstractContextManager[Any]] | None] = ContextVar(
    "xtuner_profile_module_bw_stack", default=None
)


class _ModuleBwOpen(Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, name: str) -> Tensor:
        ctx.name = name
        return tensor

    @staticmethod
    def backward(ctx, grad: Tensor):
        cm = torch.profiler.record_function(f"xtuner.L2.{ctx.name}.backward")
        cm.__enter__()
        stack = _MODULE_BW_STACK.get()
        if stack is None:
            stack = []
            _MODULE_BW_STACK.set(stack)
        stack.append(cm)
        return grad, None


def attach_l2_backward(tensor: Tensor, name: str) -> Tensor:
    """Open ``xtuner.L2.{name}.backward`` and close it after ``tensor``'s
    producer.

    The open sits after ``tensor`` in the graph so it runs before the producer
    backward; a hook on the original ``grad_fn`` closes after that producer.
    """
    node = tensor.grad_fn
    out = cast(Tensor, _ModuleBwOpen.apply(tensor, name))
    if node is not None:

        def _close(_grad_inputs: Any, _grad_outputs: Any) -> None:
            stack = _MODULE_BW_STACK.get()
            if stack:
                stack.pop().__exit__(None, None, None)

        node.register_hook(_close)
    return out


def profile_embedding(
    embed_tokens: nn.Module | Callable[[Tensor], Tensor],
    input_ids: Tensor,
    name: str = "embed_tokens",
) -> Tensor:
    """Record ``xtuner.L2.{name}.forward`` around an embedding call.

    Dynamo inlines ``nn.Embedding.forward``, so a Python wrapper on that method
    is skipped. The range must sit at the call site in uncompiled model code.

    Args:
        embed_tokens (nn.Module | Callable[[Tensor], Tensor]): Embedding module
            or callable, typically ``embed_tokens``.
        input_ids (Tensor): Token ids.
        name (str): L2 group name.

    Returns:
        Tensor: Embedding output, with backward opening ``{name}.backward``.
    """
    with profile_range(2, f"{name}.forward"):
        hidden = embed_tokens(input_ids)
    return attach_l2_backward(hidden, name)


@contextmanager
def profile_time(desc):
    start_t = time.time()

    yield

    cost_time = time.time() - start_t
    logger.success(f"{desc} Elapsed time {cost_time:.2f} seconds")


@contextmanager
def profile_time_and_memory(desc):
    torch_device = get_torch_device_module()
    start_t = time.time()
    torch_device.reset_peak_memory_stats()

    yield

    max_memory = torch_device.max_memory_allocated()
    cost_time = time.time() - start_t

    logger.success(f"{desc} Elapsed time {cost_time:.2f} seconds, peak gpu memory {max_memory / 1024**3:.1f}G")


# Adapted from https://github.com/volcengine/verl/blob/main/verl/utils/profiler/performance.py
@contextmanager
def timer(name: str, timer_dict: dict[str, float]):
    # TODO: install codetiming in xtuner latest images
    from codetiming import Timer

    with Timer(name=name, logger=None) as t:
        yield
    if name not in timer_dict:
        timer_dict[name] = 0.0
    timer_dict[name] += t.last


def timer_logger(time_dict: dict[str, float]):
    report_lines = [f"  - {name:<25}: {duration:.2f}s" for name, duration in time_dict.items()]
    total_duration = sum(time_dict.values())
    report_lines.append(f"  - {'Total':<25}: {total_duration:.2f}s")
    return "\n".join(report_lines)
