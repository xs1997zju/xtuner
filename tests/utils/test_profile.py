import torch

from xtuner.v1.module.lm_head import LMHead
from xtuner.v1.module.rms_norm import RMSNorm
from xtuner.v1.utils.profile import profile_embedding, profiled_item


def test_profile_embedding_records_forward_and_backward():
    embed = torch.nn.Embedding(8, 4)
    tokens = torch.tensor([[1, 2, 3]])

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        hidden = profile_embedding(embed, tokens)
        hidden.sum().backward()

    events = list(profiler.events())
    names = {event.name for event in events}
    assert "xtuner.L2.embed_tokens.forward" in names
    assert "xtuner.L2.embed_tokens.backward" in names
    backward = next(event for event in events if event.name == "xtuner.L2.embed_tokens.backward")
    embedding_bwd = next(event for event in events if event.name == "EmbeddingBackward0")
    # Close happens on EmbeddingBackward0's node hook, so the range must cover it.
    assert backward.cpu_time_total >= embedding_bwd.cpu_time_total


def test_final_rmsnorm_records_l2_ranges_and_layer_norm_does_not():
    layer_norm = RMSNorm(4)
    final_norm = RMSNorm(4, l2_name="norm")
    hidden = torch.randn(2, 4, requires_grad=True)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        layer_norm(hidden).sum().backward()

    assert not any(event.name.startswith("xtuner.L2.norm.") for event in profiler.events())

    hidden = torch.randn(2, 4, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        final_norm(hidden).sum().backward()

    names = {event.name for event in profiler.events()}
    assert "xtuner.L2.norm.forward" in names
    assert "xtuner.L2.norm.backward" in names


def test_profiled_item_records_host_sync_range():
    value = torch.tensor(3.5)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        assert profiled_item(value, "loss") == 3.5
    names = {event.name for event in profiler.events()}
    assert "xtuner.L3.host_sync.loss" in names


def test_lm_head_records_l2_ranges():
    head = LMHead(4, 8)
    hidden = torch.randn(2, 4, requires_grad=True)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        _, (logits, _) = head(hidden, None)
        assert logits is not None
        logits.sum().backward()

    names = {event.name for event in profiler.events()}
    assert "xtuner.L2.lm_head.forward" in names
    assert "xtuner.L2.lm_head.backward" in names
