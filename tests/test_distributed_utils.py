import pytest
import torch

from src.distributed_utils import DistributedContext, unwrap_model


def test_non_torchrun_environment_is_single_process(monkeypatch):
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)

    context = DistributedContext.initialize()

    assert not context.enabled
    assert context.is_main_process
    assert context.rank == 0
    assert context.world_size == 1


def test_incomplete_torchrun_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="Incomplete torchrun environment"):
        DistributedContext.initialize()


def test_unwrap_model_keeps_plain_module_unchanged():
    model = torch.nn.Linear(2, 2)
    assert unwrap_model(model) is model
