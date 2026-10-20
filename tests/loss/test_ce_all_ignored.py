import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.loss.ce_loss import CELossConfig, CELossContext, CELossKwargs


@pytest.mark.parametrize("mode", ["eager", "chunk"])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_ce_loss_matches_unconditional_cross_entropy(mode: str, all_ignored: bool) -> None:
    labels = torch.tensor([[-100, -100, -100, -100] if all_ignored else [1, -100, 2, -100]])
    loss_weight = (labels != -100).float() / max(int((labels != -100).sum()), 1)
    hidden = torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(7), requires_grad=True)
    weight = torch.randn(5, 3, generator=torch.Generator().manual_seed(8), requires_grad=True)

    ctx = CELossContext(
        CELossConfig(mode=mode, chunk_size=2),
        CELossKwargs(shifted_labels=labels, loss_weight=loss_weight),
    )
    loss = ctx.forward(hidden, weight)[0]
    grad_hidden, grad_weight = torch.autograd.grad(loss, (hidden, weight))

    expected = (
        F.cross_entropy(F.linear(hidden, weight).float().flatten(0, 1), labels.flatten(), reduction="none")
        * loss_weight.flatten()
    ).sum()
    expected_grad_hidden, expected_grad_weight = torch.autograd.grad(expected, (hidden, weight))
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(grad_hidden, expected_grad_hidden)
    torch.testing.assert_close(grad_weight, expected_grad_weight)
