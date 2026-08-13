import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @classmethod
    def initialize(cls) -> "DistributedContext":
        distributed_keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        present = [key in os.environ for key in distributed_keys]
        if any(present) and not all(present):
            missing = [
                key for key, is_present in zip(distributed_keys, present) if not is_present
            ]
            raise RuntimeError(
                "Incomplete torchrun environment; missing " + ", ".join(missing)
            )
        if not all(present):
            return cls(enabled=False, rank=0, local_rank=0, world_size=1)

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                f"Invalid distributed ranks: rank={rank}, world_size={world_size}"
            )

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} "
                    "CUDA devices are visible"
                )
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        return cls(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        if self.world_size == 1:
            return model
        if device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        return DistributedDataParallel(model)

    def reduce_sums(
        self, values: list[float], device: torch.device
    ) -> list[float]:
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model
