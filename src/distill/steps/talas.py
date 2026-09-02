"""The talas training step.

Reads from the distiller context: config, device_s, device_t, model_student, criterion, optimizer, scaler, scheduler, train_loader, current_epoch, current_step
"""

import torch
from torch import optim
from torch.amp import autocast
from transformers import get_scheduler

from src.criterions.teacher_anchor_kd import TeacherAnchorKD
from src.distill.numerics import grads_are_finite, is_finite
from src.loss import info_nce

try:
    from pytorch_optimizer import SAM

    SAM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    SAM = None
    SAM_AVAILABLE = False


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

        # Initialize TALAS criterion if needed
        if ctx.criterion is None:
            d_s = ctx.model_student.config.hidden_size
            d_t = teacher_cls.shape[-1]

            # BERT-base has 13 layers: embedding + 12 transformer layers
            num_layers = len(s_out1.hidden_states)

            ctx.criterion = TeacherAnchorKD(
                student_dim=d_s,
                teacher_dim=d_t,
                num_layers=num_layers,
                last_layer_idx=cfg.last_layer_idx,
                start_rkd=cfg.start_rkd,
                w_task=cfg.w_task,
                w_kd=cfg.w_kd,
                w_struct=cfg.w_struct,
                eps_norm=cfg.eps_norm,
            ).to(ctx.device_s)

            # Initialize SAM optimizer with both student and criterion parameters
            if not SAM_AVAILABLE:
                raise RuntimeError(
                    "SAM optimizer not available. Install pytorch_optimizer."
                )

            base_optimizer = optim.AdamW
            ctx.optimizer = SAM(
                [
                    {
                        "params": ctx.model_student.parameters(),
                        "lr": cfg.learning_rate,
                        "weight_decay": 0.01,
                    },
                    {
                        "params": ctx.criterion.parameters(),
                        "lr": cfg.learning_rate,
                        "weight_decay": 0.01,
                    },
                ],
                base_optimizer,
                rho=getattr(cfg, "rho", 0.05),
                adaptive=True,
            )

            # Initialize scheduler
            num_steps = len(ctx.train_loader)
            total_steps = num_steps * cfg.epochs
            min_lr_rate = cfg.min_lr / cfg.learning_rate
            ctx.scheduler = get_scheduler(
                name="cosine_with_min_lr",
                optimizer=ctx.optimizer,
                num_warmup_steps=int(total_steps * cfg.warmup_ratio),
                num_training_steps=total_steps,
                scheduler_specific_kwargs={"min_lr_rate": min_lr_rate},
            )

            print(
                f"Initialized TeacherAnchorKD: {d_s} -> {d_t}, num_layers={num_layers}, last_layer_idx={cfg.last_layer_idx}, start_rkd={cfg.start_rkd}"
            )
            print(f"Initialized SAM optimizer with rho={getattr(cfg, 'rho', 0.05)}")
            print(
                f"Initialized scheduler: {total_steps} steps, warmup={int(total_steps * cfg.warmup_ratio)}"
            )

        # Now safe to call criterion with initialized projection heads
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
    if not grads_are_finite(ctx.optimizer):
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

    # Check gradients again
    if not grads_are_finite(ctx.optimizer):
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
