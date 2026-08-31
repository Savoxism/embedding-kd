"""Finite-value checks and the gradient-norm ceiling.

Pure functions over tensors and modules: nothing here touches the distiller's
state, which is why they live outside it.
"""

import torch
from torch import nn

# Global gradient-norm ceiling, applied after `scaler.unscale_`. Not configuration:
# 1.0 is the standard convention for transformer fine-tuning (Devlin et al., 2019;
# it is also the HuggingFace `Trainer` default), applied here without modification.
# The step used to have no ceiling at all -- only a finiteness check that skipped
# the update outright -- so a single anchor whose softmax sat at a per-row bandwidth
# near tau_min could take an arbitrarily large step, and the only evidence of it
# would be the loss curve afterwards.
MAX_GRAD_NORM = 1.0


def is_finite(x: torch.Tensor) -> bool:
    return torch.is_tensor(x) and torch.isfinite(x).all().item()


def nonfinite_details(name: str, tensor: torch.Tensor) -> str:
    if not torch.is_tensor(tensor):
        return f"{name}: expected tensor, got {type(tensor).__name__}"
    if tensor.is_floating_point() or tensor.is_complex():
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
    else:
        nan_count = 0
        inf_count = 0
    return (
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, nan_count={nan_count}, inf_count={inf_count}"
    )


def assert_module_parameters_finite(module: nn.Module, module_name: str) -> None:
    finite_status = None
    for parameter in module.parameters():
        current = torch.isfinite(parameter).all()
        finite_status = current if finite_status is None else finite_status & current

    if finite_status is None or bool(finite_status.item()):
        return

    for name, parameter in module.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise RuntimeError(
                f"{module_name} parameters became NaN/Inf: "
                f"{nonfinite_details(name, parameter)}"
            )


def grads_are_finite(optim) -> bool:
    # Accumulated on device and read back once. Testing each gradient in a Python
    # `if` forces a host sync per parameter, which is ~200 stalls per step on a
    # BERT-base student -- for a check that is almost always True.
    finite_status = None
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            current = torch.isfinite(p.grad).all()
            finite_status = (
                current if finite_status is None else finite_status & current
            )
    return finite_status is None or bool(finite_status.item())
