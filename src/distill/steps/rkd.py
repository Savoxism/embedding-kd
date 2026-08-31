"""The rkd training step.

Reads from the distiller context: config, device_s, model_student, criterion, optimizer, scaler, scheduler, current_epoch, current_step
"""

import torch
from torch.amp import autocast

from src.distill.numerics import grads_are_finite, is_finite
from src.loss import info_nce


def step(ctx, batch: dict) -> tuple[torch.Tensor, dict]:
    cfg = ctx.config
    batch_s = {
        key: value.to(ctx.device_s, non_blocking=True)
        for key, value in batch.items()
        if torch.is_tensor(value)
        and (key.endswith("_stu") or key in {"labels", "teacher_cls"})
    }
    ctx.optimizer.zero_grad(set_to_none=True)

    with autocast("cuda", enabled=torch.cuda.is_available()):
        student_output = ctx.model_student(
            input_ids=batch_s["input_ids1_stu"],
            attention_mask=batch_s["attention_mask1_stu"],
            return_dict=True,
        )
        student_cls = student_output.last_hidden_state[:, 0, :]

        task_loss = None
        if cfg.w_task > 0:
            if "input_ids2_stu" not in batch_s:
                raise ValueError("RKD with w_task > 0 requires a second text view")
            student_output2 = ctx.model_student(
                input_ids=batch_s["input_ids2_stu"],
                attention_mask=batch_s["attention_mask2_stu"],
                return_dict=True,
            )
            student_cls2 = student_output2.last_hidden_state[:, 0, :]
            task_loss, _ = info_nce(
                student_cls,
                student_cls2,
                temperature=cfg.temperature,
            )

        loss, metrics = ctx.criterion(
            student=student_cls,
            teacher=batch_s["teacher_cls"],
            task_loss=task_loss,
        )
        loss = loss.float()

    if not is_finite(loss):
        raise RuntimeError(
            f"RKD loss NaN/Inf at epoch={ctx.current_epoch} step={ctx.current_step}"
        )

    ctx.scaler.scale(loss).backward()
    ctx.scaler.unscale_(ctx.optimizer)
    if not grads_are_finite(ctx.optimizer):
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.scaler.update()
        ctx.scheduler.step()
        return loss, {**metrics, "skip": "grad_inf"}

    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    ctx.scheduler.step()
    return loss, metrics
