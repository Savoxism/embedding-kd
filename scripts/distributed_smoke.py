import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distributed_utils import DistributedContext, unwrap_model


def main() -> None:
    context = DistributedContext.initialize()
    device = (
        torch.device("cuda", context.local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.manual_seed(11)
    model = context.wrap_model(nn.Linear(4, 2).to(device), device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.full((3, 4), float(context.rank + 1), device=device)
    anchor_output = model(inputs)
    candidate_output = unwrap_model(model)(inputs * 0.5)
    loss = anchor_output.square().mean() + candidate_output.square().mean()
    loss.backward()
    optimizer.step()

    reduced = context.reduce_sums([float(context.rank + 1)], device)[0]
    expected = context.world_size * (context.world_size + 1) / 2
    assert reduced == expected
    assert all(not key.startswith("module.") for key in unwrap_model(model).state_dict())
    if context.is_main_process:
        print(
            f"DISTRIBUTED_SMOKE_OK backend_device={device.type} "
            f"world_size={context.world_size} rank_sum={reduced:.0f}"
        )
    context.barrier()
    context.close()


if __name__ == "__main__":
    main()
