"""The talas training step.

The criterion, the SAM optimizer and the schedule are built before training
starts (see ``src/methods/talas.py``); this used to create all three here, on
the first batch, because it read the student's layer count off that batch.

Reads from the distiller context: config, device_s, model_student, criterion,
optimizer, scaler, scheduler, current_epoch, current_step
"""

import math

import torch
from torch.amp import autocast

from src.distill.numerics import is_finite, total_grad_norm
from src.loss import info_nce


def restore_without_update(ctx) -> int:
    """Undo ``SAM.first_step`` when the second pass is abandoned.

    ``first_step`` moves the weights to ``w + eps(w)`` and stashes ``w`` in the
    optimizer state; only ``second_step`` puts them back, and it also applies the
    wrapped update. A pass-2 skip must restore *without* updating -- otherwise
    training silently continues from adversarially perturbed weights for the
    remainder of the run.
    """
    restored = 0
    for group in ctx.optimizer.param_groups:
        for p in group["params"]:
            old = ctx.optimizer.state.get(p, {}).get("old_p")
            if old is not None:
                p.data.copy_(old)
                restored += 1
    if restored == 0:
        print(
            "Warning: SAM skip restored no weights (no 'old_p' in optimizer "
            "state); training may continue from perturbed weights."
        )
    return restored


def step(ctx, batch: dict) -> tuple[torch.Tensor, dict]:
    cfg = ctx.config
    batch_s = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith("_stu") or k == "labels" or k == "teacher_cls":
            batch_s[k] = v.to(ctx.device_s, non_blocking=True)

    # ========== FIRST PASS ==========
    with autocast("cuda", enabled=torch.cuda.is_available()):
        teacher_cls = batch_s["teacher_cls"]

        s_out1 = ctx.model_student(
            input_ids=batch_s["input_ids1_stu"],
            attention_mask=batch_s["attention_mask1_stu"],
            output_hidden_states=True,
            return_dict=True,
        )
        s_out2 = ctx.model_student(
            input_ids=batch_s["input_ids2_stu"],
            attention_mask=batch_s["attention_mask2_stu"],
            output_hidden_states=False,
            return_dict=True,
        )

        S_last1 = s_out1.last_hidden_state
        S_last2 = s_out2.last_hidden_state
        S_cls1 = S_last1[:, 0, :]
        S_cls2 = S_last2[:, 0, :]

        loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)

        student_outputs = {
            "hidden_states": s_out1.hidden_states,
            "last_hidden_state": S_last1,
        }

        loss, metrics = ctx.criterion(
            student_outputs=student_outputs,
            teacher_cls=teacher_cls,
            task_loss=loss_task,
        )

        loss = loss.float()

    # Backward pass 1 (this will init gradients for first_step)
    ctx.scaler.scale(loss).backward()

    # Check gradients
    ctx.scaler.unscale_(ctx.optimizer)
    metrics["grad_norm"] = float(total_grad_norm(ctx.optimizer))
    if not math.isfinite(metrics["grad_norm"]):
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.scaler.update()
        ctx.scheduler.step()
        return loss, {**metrics, "skip": "grad_inf_p1"}

    # SAM first step
    ctx.optimizer.first_step(zero_grad=True)

    # ========== SECOND PASS ==========
    with autocast("cuda", enabled=torch.cuda.is_available()):
        s_out1_2 = ctx.model_student(
            input_ids=batch_s["input_ids1_stu"],
            attention_mask=batch_s["attention_mask1_stu"],
            output_hidden_states=True,
            return_dict=True,
        )
        s_out2_2 = ctx.model_student(
            input_ids=batch_s["input_ids2_stu"],
            attention_mask=batch_s["attention_mask2_stu"],
            output_hidden_states=False,
            return_dict=True,
        )

        S_last1_2 = s_out1_2.last_hidden_state
        S_last2_2 = s_out2_2.last_hidden_state
        S_cls1_2 = S_last1_2[:, 0, :]
        S_cls2_2 = S_last2_2[:, 0, :]

        loss_task_2, _ = info_nce(S_cls1_2, S_cls2_2, temperature=cfg.temperature)

        student_outputs_2 = {
            "hidden_states": s_out1_2.hidden_states,
            "last_hidden_state": S_last1_2,
        }

        loss_2, _ = ctx.criterion(
            student_outputs=student_outputs_2,
            teacher_cls=teacher_cls,
            task_loss=loss_task_2,
        )

        loss_2 = loss_2.float()

    if not is_finite(loss_2):
        raise RuntimeError(
            f"loss_2 NaN/Inf at epoch={ctx.current_epoch} step={ctx.current_step}"
        )

    # Backward pass 2 - IMPORTANT: Do NOT scale (plain backward)
    loss_2.backward()

    # Check gradients again. The SAM ascent point has its own gradient, so this
    # is a second measurement rather than a repeat of the pass-1 one.
    metrics["grad_norm_p2"] = float(total_grad_norm(ctx.optimizer))
    if not math.isfinite(metrics["grad_norm_p2"]):
        # first_step() already perturbed the weights; put them back before
        # abandoning the step.
        restore_without_update(ctx)
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.scaler.update()
        # Same policy as the ggpkd skip path: the schedule was sized for
        # len(train_loader) * epochs steps, so a skipped update must still
        # advance it or the LR runs permanently behind its own curve.
        ctx.scheduler.step()
        return loss, {**metrics, "skip": "grad_inf_p2"}

    # SAM second step
    ctx.optimizer.second_step(zero_grad=True)
    # `second_step()` performs the wrapped AdamW update directly, so
    # PyTorch's scheduler wrapper never observes an `optimizer.step()` call.
    # Mark the real update before advancing the schedule; otherwise it emits
    # a false "scheduler before optimizer" warning on the first TALAS step.
    ctx.optimizer._opt_called = True
    ctx.scaler.update()
    ctx.scheduler.step()

    # Clean up
    del s_out1, s_out2, s_out1_2, s_out2_2
    del student_outputs, student_outputs_2

    return loss, metrics
