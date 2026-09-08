"""Finite-value checks and the gradient-norm diagnostic.

Pure functions over tensors and modules: nothing here touches the distiller's
state, which is why they live outside it.

**There is no gradient-norm ceiling.** There used to be one, at 1.0, applied
after ``scaler.unscale_``. Measured over the 45,365 logged steps in
``ablations/runs`` it bound on 100.00% of steps of every arm of the primary pair
(median grad_norm 2.03, max 4.34), so it was not the safety net its comment
described -- it was an unconditional gradient normalization, and one that two
ablation arms escaped (``batch_local`` at median 0.67 and ``uniform_corpus`` at
0.26 sat below it) while the other 41 runs were renormalized to exactly 1.0.
That made the arms it did bind differ from the arms it did not by an effective
step size rather than only by the thing being ablated. The norm is still
computed and logged, because that measurement is what the diagnostic is for; it
just no longer changes the update.
"""

import torch
from torch import nn


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
    """Raise if any parameter is NaN/Inf, naming the first one that is.

    Called at load time and nowhere in the training loop. It used to run after
    every optimizer step, where it cost ~200 kernel launches plus a host sync
    per step to check something that a non-finite gradient norm already reports
    one step earlier -- and which fired zero times in 45,365 logged steps.
    """
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


def total_grad_norm(optim) -> torch.Tensor:
    """L2 norm over every gradient the optimizer will apply, as a device tensor.

    Every parameter group, not just the student's: a diagnostic that skips a
    param group does not describe the update that is about to happen.

    Returned unread, so the caller decides when to pay the one host sync. That
    single read also answers "are the gradients finite" -- the norm is NaN or Inf
    exactly when some gradient is -- which is why there is no longer a separate
    ``grads_are_finite`` walking every gradient with its own sync.
    """
    grads = [
        p.grad
        for group in optim.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    if not grads:
        return torch.zeros(())
    # One fused kernel over the whole list rather than one launch per tensor.
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
