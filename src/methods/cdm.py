"""CDM: contextual dynamic mapping over aligned token spans."""

import torch
import torch.nn.functional as F
from torch import nn

from config import CDMConfig
from src.criterions.contextual_dynamic_mapping import ContextualDynamicMapping
from src.distill.steps.standard import step
from src.methods.spec import MethodSpec
from src.methods.support import attach_parameters


def build_criterion(ctx, config):
    criterion = ContextualDynamicMapping(
        tok_student=ctx.tok_student,
        tok_teacher=ctx.tok_teacher,
        blending_model_special_token=config.teacher_special_token,
        base_model_special_token=config.student_special_token,
        w_task=config.w_task,
        alpha_dtw=config.alpha_dtw,
        debug_align=config.debug_align,
    )
    d_s = ctx.model_student.config.hidden_size
    d_t = ctx.model_teacher.config.hidden_size
    ctx.proj_s2t = nn.Linear(d_s, d_t, bias=False).to(ctx.device_s)
    attach_parameters(ctx, ctx.proj_s2t.parameters(), config.learning_rate * 2)
    print(f"Initialized CDM projection layer: {d_s} -> {d_t}")
    return criterion


def kd_loss(ctx, t):
    """DTW-aligned token distillation plus a projected pooled-vector MSE."""
    cfg = ctx.config
    keep_s1 = t.batch_s["attention_mask1_stu"].bool() & (
        ~t.batch_s["special_tokens_mask1_stu"].bool()
    )
    keep_t1 = t.batch_t["attention_mask1_tea"].to(ctx.device_s).bool() & (
        ~t.batch_t["special_tokens_mask1_tea"].to(ctx.device_s).bool()
    )

    kd_dtw = ctx.criterion.compute_cdm_loss(
        S_last=t.S_last1,
        T_last=t.T_last1,
        batch_input_ids_stu=t.batch["input_ids1_stu"],
        batch_input_ids_tea=t.batch["input_ids1_tea"],
        keep_mask_stu=keep_s1,
        keep_mask_tea=keep_t1,
        proj_s2t=ctx.proj_s2t,
        device_s=ctx.device_s,
        epoch=ctx.current_epoch,
        step=ctx.current_step,
    )

    S_proj_cls1 = ctx.proj_s2t(t.S_cls1)
    S_proj_cls1_norm = F.normalize(S_proj_cls1, p=2, dim=-1)
    T_cls1_norm = F.normalize(t.T_cls1, p=2, dim=-1)
    kd_cls = F.mse_loss(S_proj_cls1_norm, T_cls1_norm)

    loss = (
        cfg.w_task * t.loss_task
        + cfg.alpha_dtw * kd_dtw * 100
        + cfg.w_cls * kd_cls
    )

    metrics = {
        "loss_total": loss.item(),
        "t.loss_task": t.loss_task.item(),
        "loss_kd_dtw": kd_dtw.item()
        if isinstance(kd_dtw, torch.Tensor)
        else kd_dtw,
        "loss_kd_cls": kd_cls.item(),
    }
    return loss, metrics


SPEC = MethodSpec(
    name="cdm",
    config_cls=CDMConfig,
    step=step,
    kd_loss=kd_loss,
    build_criterion=build_criterion,
)
