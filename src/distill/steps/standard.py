"""The shared training step for cdm, dskd, emo and stella.

Unlike ggpkd/rkd/talas -- which use a cached teacher -- these four run the
teacher model inside the step, so they share a prologue (teacher forward, student
forward, pooling, task loss) and differ only in the KD term.

Reads from the distiller context: config, device_s, device_t, model_student,
model_teacher, criterion, proj_s2t, task_head, tok_student, tok_teacher,
optimizer, scaler, scheduler, current_epoch, current_step, current_stage.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.criterions.stella_distillation import stella_stage1_loss, stella_stage2_loss
from src.loss import info_nce
from src.pooling import last_token_pool


def step(ctx, batch: dict) -> tuple[torch.Tensor, dict]:
    cfg = ctx.config
    method = cfg.distill_method
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
        need_atts = method == "emo"
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
                    att.to(ctx.device_s, non_blocking=True)
                    for att in t_out1.attentions
                )
                T_last2 = None
                T_atts2 = None
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

        # Different models have different forward signatures
        if method == "stella":
            # StellaModel doesn't accept output_attentions or return_dict
            s_out1 = ctx.model_student(
                input_ids=batch_s["input_ids1_stu"],
                attention_mask=batch_s["attention_mask1_stu"],
            )
            s_out2 = ctx.model_student(
                input_ids=batch_s["input_ids2_stu"],
                attention_mask=batch_s["attention_mask2_stu"],
            )
        elif method == "emo":
            # EMO needs attentions
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
            # CDM, DSKD - standard transformers models
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
        if method != "stella":
            S_last1 = s_out1.last_hidden_state
            S_last2 = None if s_out2 is None else s_out2.last_hidden_state
            S_cls1 = S_last1[:, 0, :]
            S_cls2 = None if S_last2 is None else S_last2[:, 0, :]
        else:
            S_cls1 = s_out1["pooled"]
            S_cls2 = s_out2["pooled"]

        if method == "emo":
            loss_task, task_metrics = ctx._compute_task_loss(
                S_cls1, S_cls2, batch_s
            )
        else:
            loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)
            task_metrics = {}

        # ========== Method-specific KD loss ==========
        if method == "cdm":
            keep_s1 = batch_s["attention_mask1_stu"].bool() & (
                ~batch_s["special_tokens_mask1_stu"].bool()
            )
            keep_t1 = batch_t["attention_mask1_tea"].to(ctx.device_s).bool() & (
                ~batch_t["special_tokens_mask1_tea"].to(ctx.device_s).bool()
            )

            kd_dtw = ctx.criterion.compute_cdm_loss(
                S_last=S_last1,
                T_last=T_last1,
                batch_input_ids_stu=batch["input_ids1_stu"],
                batch_input_ids_tea=batch["input_ids1_tea"],
                keep_mask_stu=keep_s1,
                keep_mask_tea=keep_t1,
                proj_s2t=ctx.proj_s2t,
                device_s=ctx.device_s,
                epoch=ctx.current_epoch,
                step=ctx.current_step,
            )

            S_proj_cls1 = ctx.proj_s2t(S_cls1)
            S_proj_cls1_norm = F.normalize(S_proj_cls1, p=2, dim=-1)
            T_cls1_norm = F.normalize(T_cls1, p=2, dim=-1)
            kd_cls = F.mse_loss(S_proj_cls1_norm, T_cls1_norm)

            loss = (
                cfg.w_task * loss_task
                + cfg.alpha_dtw * kd_dtw * 100
                + cfg.w_cls * kd_cls
            )

            metrics = {
                "loss_total": loss.item(),
                "loss_task": loss_task.item(),
                "loss_kd_dtw": kd_dtw.item()
                if isinstance(kd_dtw, torch.Tensor)
                else kd_dtw,
                "loss_kd_cls": kd_cls.item(),
            }

        elif method == "dskd":
            mask_s1 = batch_s["attention_mask1_stu"]
            mask_t1 = batch_t["attention_mask1_tea"].to(ctx.device_s)

            spec_s1 = batch_s.get("special_tokens_mask1_stu", None)
            spec_t1 = batch_t.get("special_tokens_mask1_tea", None)
            if spec_t1 is not None:
                spec_t1 = spec_t1.to(ctx.device_s)

            loss, metrics = ctx.criterion.compute_dskd_loss(
                S_last=S_last1,
                T_last=T_last1,
                S_cls=S_cls1,
                T_cls=T_cls1,
                mask_student=mask_s1,
                mask_teacher=mask_t1,
                task_loss=loss_task,
                special_tokens_mask_student=spec_s1,
                special_tokens_mask_teacher=spec_t1,
                device=ctx.device_s,
            )

        elif method == "emo":

            # EMO's criterion wants objects with `.last_hidden_state` and
            # `.attentions`; the encoder outputs cannot be reused directly because
            # the teacher's tensors have already been moved to the student device.
            teacher_outputs = SimpleNamespace(
                last_hidden_state=T_last1, attentions=T_atts
            )
            student_outputs = SimpleNamespace(
                last_hidden_state=S_last1, attentions=s_out1.attentions
            )

            att_loss_weight = getattr(cfg, "att_loss_weight", 0.1)
            ot_loss_weight = getattr(cfg, "ot_loss_weight", 1.0)

            kd_loss, kd_metrics = ctx.criterion.compute_emo_loss(
                teacher_outputs=teacher_outputs,
                student_outputs=student_outputs,
                input_ids_tea=batch_t["input_ids1_tea"].to(ctx.device_s),
                input_ids_stu=batch_s["input_ids1_stu"],
                attention_mask_tea=batch_t["attention_mask1_tea"].to(ctx.device_s),
                attention_mask_stu=batch_s["attention_mask1_stu"],
                tok_teacher=ctx.tok_teacher,
                tok_student=ctx.tok_student,
                att_loss_weight=att_loss_weight,
                ot_loss_weight=ot_loss_weight,
            )
            if S_last2 is not None and T_last2 is not None:
                teacher_outputs2 = SimpleNamespace(
                    last_hidden_state=T_last2, attentions=T_atts2
                )
                student_outputs2 = SimpleNamespace(
                    last_hidden_state=S_last2, attentions=s_out2.attentions
                )
                kd_loss2, kd_metrics2 = ctx.criterion.compute_emo_loss(
                    teacher_outputs=teacher_outputs2,
                    student_outputs=student_outputs2,
                    input_ids_tea=batch_t["input_ids2_tea"].to(ctx.device_s),
                    input_ids_stu=batch_s["input_ids2_stu"],
                    attention_mask_tea=batch_t["attention_mask2_tea"].to(
                        ctx.device_s
                    ),
                    attention_mask_stu=batch_s["attention_mask2_stu"],
                    tok_teacher=ctx.tok_teacher,
                    tok_student=ctx.tok_student,
                    att_loss_weight=att_loss_weight,
                    ot_loss_weight=ot_loss_weight,
                )
                kd_loss = 0.5 * (kd_loss + kd_loss2)
                kd_metrics = {
                    key: 0.5 * (kd_metrics[key] + kd_metrics2[key])
                    for key in kd_metrics
                }

            w_task = getattr(cfg, "w_task", 0.5)
            alpha_kd = getattr(cfg, "alpha_kd", 0.5)
            loss = w_task * loss_task + alpha_kd * kd_loss

            metrics = {
                "loss_total": loss.item(),
                "loss_task": loss_task.item(),
                **task_metrics,
                **kd_metrics,
            }

        elif method == "stella":
            if ctx.current_stage == 1:
                S_emb = s_out1["fc1"]
                loss, metrics = stella_stage1_loss(
                    S_emb,
                    T_cls1,
                    w_cos=getattr(cfg, "w_cos_stage1", 10.0),
                    w_sim=getattr(cfg, "w_sim_stage1", 200.0),
                    w_tri=getattr(cfg, "w_tri_stage1", 20.0),
                )
            else:
                loss, metrics = stella_stage2_loss(
                    S_cls1,
                    S_cls2,
                    s_out1["fc1"],
                    s_out1["fc2"],
                    s_out1["fc3"],
                    s_out1["fc4"],
                    T_cls1,
                    temperature=cfg.temperature,
                    w_task=cfg.w_task,
                    w_cos=getattr(cfg, "w_cos_stage2", 10.0),
                    w_sim=getattr(cfg, "w_sim_stage2", 200.0),
                    w_tri=getattr(cfg, "w_tri_stage2", 20.0),
                )

        else:
            raise ValueError(f"Unknown distillation method: {method}")

        loss = loss.float()

    ctx.scaler.scale(loss).backward()
    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    ctx.scheduler.step()

    return loss, metrics
