from types import SimpleNamespace

import pytest

from xtuner.v1.datasets import config as config_module
from xtuner.v1.datasets.config import DataloaderConfig, build_dataloader


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_default_prefetch_preserves_native_default(workers):
    config = DataloaderConfig(num_workers=workers)
    assert config.prefetch_factor is None
    assert config.model_dump()["prefetch_factor"] is None


@pytest.mark.parametrize("factor", [1, 2])
def test_explicit_prefetch_serializes_for_paired_controls(factor):
    config = DataloaderConfig(num_workers=2, prefetch_factor=factor)
    assert config.model_dump()["prefetch_factor"] == factor


@pytest.mark.parametrize("factor", [0, -1])
def test_prefetch_rejects_nonpositive_factor(factor):
    with pytest.raises(ValueError):
        DataloaderConfig(num_workers=1, prefetch_factor=factor)


def test_explicit_prefetch_requires_workers():
    with pytest.raises(ValueError, match="requires num_workers"):
        DataloaderConfig(num_workers=0, prefetch_factor=1)


@pytest.mark.parametrize("factor", [None, 1, 2])
@pytest.mark.parametrize("builder", ["legacy", "config"])
def test_both_builders_forward_prefetch_without_starting_workers(monkeypatch, factor, builder):
    captured = {}

    def capture_loader(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(config_module, "TorchDataLoader", capture_loader)
    monkeypatch.setattr(config_module, "Dataloader", capture_loader)
    monkeypatch.setattr(config_module, "build_datasets", lambda *args, **kwargs: [range(4)])
    config = DataloaderConfig(dataset_config_list=[], pack_level="none", num_workers=2, prefetch_factor=factor)
    if builder == "legacy":
        build_dataloader(config, [range(4)], global_batch_size=1, micro_batch_size=1, seed=7)
    else:
        config.build(None, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=7)
    assert captured["prefetch_factor"] == factor
    assert captured["num_workers"] == 2
