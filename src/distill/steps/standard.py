"""The shared training step for methods that run the teacher inside the step.

Unlike ggpkd/rkd/talas -- which read a cached teacher -- these methods encode
with both models every step, so they share a prologue (teacher forward, student
forward, pooling, task loss) and an epilogue (backward, gradient norm, step),
and differ only in the KD term. That term used to be a four-way `if` on the
method name in the middle of this file; it is now `spec.kd_loss`, and each
method's version lives in its own module under `src/methods/`.

Reads from the distiller context: method, config, device_s, device_t,
model_student, model_teacher, criterion, proj_s2t, task_head, tok_student,
tok_teacher, optimizer, scaler, scheduler, current_epoch, current_step, and
whatever the method's own `kd_loss` reads.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.amp import autocast

from src.distill.numerics import is_finite, total_grad_norm
from src.loss import info_nce
from src.pooling import last_token_pool


@dataclass
class StepTensors:
    """Everything the prologue computed, handed to the method's KD term.

    A dataclass rather than the enclosing function's locals: the KD terms read
    an overlapping but not identical subset of these, and passing them
    explicitly is what let the four branches move out of this file.
    """

    batch: dict
    batch_s: dict
    batch_t: dict
    s_out1: Any
    s_out2: Any
    S_last1: Any
    S_last2: Any
    S_cls1: torch.Tensor
    S_cls2: Any
    T_last1: torch.Tensor
    T_last2: Any
    T_cls1: torch.Tensor
    T_atts: Any
    T_atts2: Any
    loss_task: torch.Tensor
    task_metrics: dict


def step(ctx, batch: dict) -> tuple[torch.Tensor, dict]:
    cfg = ctx.config
    spec = ctx.method
    batch_s, batch_t = {}, {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith("_stu") or k == "labels":
            batch_s[k] = v.to(ctx.device_s, non_blocking=True)
        if k.endswith("_tea"):
            batch_t[k] = v.to(ctx.device_t, non_blocking=True)

    ctx.optimizer.zero_grad(set_to_none=True)

    with autocast("cuda", enabled=torch.cuda.is_available()):
        need_atts = spec.needs_attentions
        # Defined on every path: the KD terms take one bundle, and a method that
        # does not read attentions still has to be handed something.
        T_atts = T_atts2 = T_last2 = None
        # `no_grad`, not `inference_mode`: DSKD and EMO feed these teacher
        # tensors through trainable projections, and an inference tensor cannot
        # be saved for backward. It only worked because `.to(device_s)` copies
        # when the two devices differ -- on a single-device (CPU-only) run that
        # move is a no-op, the inference flag survives, and the backward pass
        # raises. Same values, same cost here; the teacher forward is already
        # the cheap half of this step.
        with torch.no_grad():
            t_out1 = ctx.model_teacher(
                input_ids=batch_t["input_ids1_tea"],
                attention_mask=batch_t["attention_mask1_tea"],
                output_attentions=need_atts,
                return_dict=True,
            )
            T_last1 = t_out1.last_hidden_state
            T_cls1 = last_token_pool(T_last1, batch_t["attention_mask1_tea"])

            T_last1 = T_last1.to(ctx.device_s, non_blocking=True)
            T_cls1 = T_cls1.to(ctx.device_s, non_blocking=True)

            if need_atts:
                T_atts = tuple(
                    att.to(ctx.device_s, non_blocking=True) for att in t_out1.attentions
                )
                if "input_ids2_tea" in batch_t:
                    t_out2 = ctx.model_teacher(
                        input_ids=batch_t["input_ids2_tea"],
                        attention_mask=batch_t["attention_mask2_tea"],
                        output_attentions=True,
                        return_dict=True,
                    )
                    T_last2 = t_out2.last_hidden_state.to(
                        ctx.device_s, non_blocking=True
                    )
                    T_atts2 = tuple(
                        attention.to(ctx.device_s, non_blocking=True)
                        for attention in t_out2.attentions
                    )

        # Student forward. The signature differs by what the method needs back.
        if spec.student_returns_pooled:
            # Such a student does not accept output_attentions or return_dict.
            s_out1 = ctx.model_student(
                input_ids=batch_s["input_ids1_stu"],
                attention_mask=batch_s["attention_mask1_stu"],
            )
            s_out2 = ctx.model_student(
                input_ids=batch_s["input_ids2_stu"],
                attention_mask=batch_s["attention_mask2_stu"],
            )
        elif need_atts:
            s_out1 = ctx.model_student(
                input_ids=batch_s["input_ids1_stu"],
                attention_mask=batch_s["attention_mask1_stu"],
                output_attentions=True,
                return_dict=True,
            )
            s_out2 = None
            if "input_ids2_stu" in batch_s:
                s_out2 = ctx.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    output_attentions=True,
                    return_dict=True,
                )
        else:
            # A plain encoder: one hidden-state stream per view.
            s_out1 = ctx.model_student(
                input_ids=batch_s["input_ids1_stu"],
                attention_mask=batch_s["attention_mask1_stu"],
                return_dict=True,
            )
            s_out2 = ctx.model_student(
                input_ids=batch_s["input_ids2_stu"],
                attention_mask=batch_s["attention_mask2_stu"],
                return_dict=True,
            )
        S_last1 = S_last2 = None
        if not spec.student_returns_pooled:
            S_last1 = s_out1.last_hidden_state
            S_last2 = None if s_out2 is None else s_out2.last_hidden_state
            S_cls1 = S_last1[:, 0, :]
            S_cls2 = None if S_last2 is None else S_last2[:, 0, :]
        else:
            S_cls1 = s_out1["pooled"]
            S_cls2 = s_out2["pooled"]

        if spec.supervised_task_loss:
            loss_task, task_metrics = ctx.compute_task_loss(S_cls1, S_cls2, batch_s)
        else:
            loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)
            task_metrics = {}

        # The only part that differs between these methods.
        loss, metrics = spec.kd_loss(
            ctx,
            StepTensors(
                batch=batch,
                batch_s=batch_s,
                batch_t=batch_t,
                s_out1=s_out1,
                s_out2=s_out2,
                S_last1=S_last1,
                S_last2=S_last2,
                S_cls1=S_cls1,
                S_cls2=S_cls2,
                T_last1=T_last1,
                T_last2=T_last2,
                T_cls1=T_cls1,
                T_atts=T_atts,
                T_atts2=T_atts2,
                loss_task=loss_task,
                task_metrics=task_metrics,
            ),
        )

        loss = loss.float()

    if not is_finite(loss):
        raise RuntimeError(
            f"{spec.name} loss NaN/Inf at epoch={ctx.current_epoch} "
            f"step={ctx.current_step}"
        )

    ctx.scaler.scale(loss).backward()
    ctx.scaler.unscale_(ctx.optimizer)

    # These four methods used to reach the optimizer with no finiteness guard and
    # no gradient diagnostic at all, while ggpkd/rkd/talas had both -- so the
    # method and its baselines were not being trained under the same contract.
    # They now share one: measure the norm, never enforce a ceiling on it, skip
    # the update if it is not finite.
    metrics["grad_norm"] = float(total_grad_norm(ctx.optimizer))
    if not math.isfinite(metrics["grad_norm"]):
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.scaler.update()
        ctx.scheduler.step()
        return loss, {**metrics, "skip": "grad_inf"}

    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    ctx.scheduler.step()

    return loss, metrics
