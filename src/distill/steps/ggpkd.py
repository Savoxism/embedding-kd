"""The ggpkd training step.

Reads from the distiller context: config, device_s, model_student, criterion, optimizer, scaler, scheduler, current_epoch, current_step
"""

import torch
from torch.amp import autocast

from src.distill.numerics import (
    MAX_GRAD_NORM,
    assert_module_parameters_finite,
    grads_are_finite,
    is_finite,
)


def step(ctx, batch: dict) -> tuple[torch.Tensor, dict]:
    cfg = ctx.config
    batch_s = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith("_stu") or k in {
            "labels",
            "idx",
            "candidate_idx",
            "candidate_inverse",
            "teacher_probs",
            "geometry_head_probs",
            "geometry_tail_positions",
            "geometry_tail_mass",
        }:
            batch_s[k] = v.to(ctx.device_s, non_blocking=True)

    ctx.optimizer.zero_grad(set_to_none=True)

    with autocast("cuda", enabled=torch.cuda.is_available()):
        s_out1 = ctx.model_student(
            input_ids=batch_s["input_ids1_stu"],
            attention_mask=batch_s["attention_mask1_stu"],
            return_dict=True,
            output_hidden_states=False,
        )

        # A second encoder pass for a SimCSE term used to sit here, gated on
        # `lambda_simcse`. That knob was declared on no config (ggpkd_config.py
        # records its removal), so the gate was constantly False and the pass never
        # ran; it is dropped with the rest of the multi-layer branch it belonged to.

        # Candidates arrive deduplicated and grouped by length, so each chunk
        # pads to its own longest member. `candidate_inverse` expands the
        # encoded rows back to the flat [batch_size * candidate_size] layout
        # the criterion expects; the gather is differentiable, so a candidate
        # shared by several anchors accumulates all of their gradient.

        chunk_embeddings_single = []

        # Encoder budget, counted where the encoding actually happens. Support
        # policies dedup differently -- a draw concentrated on the same columns
        # costs fewer unique texts than a spread one at the same quota -- so
        # "equal candidate quota" is not "equal compute", and an ablation compared
        # at equal quota alone would credit a policy for buying more forward
        # passes. These two counters are what the arms are matched on.
        # getattr rather than a bare += : the step is also driven by contexts that
        # are not the distiller (the train-step contract tests build a minimal one),
        # and a diagnostic counter must not be what decides whether the step runs.
        # Read off `batch`, not `batch_s`: the latter lives on the GPU, and
        # int() on a CUDA tensor blocks until the queue drains -- a per-step sync
        # in the training loop, paid for a counter. The two hold the same values.
        encoded_texts = int(batch["input_ids1_stu"].size(0))
        encoded_tokens = int(batch["attention_mask1_stu"].sum())

        for chunk in batch["candidate_chunks"]:
            chunk_out = ctx.model_student(
                input_ids=chunk["input_ids"].to(
                    ctx.device_s, non_blocking=True
                ),
                attention_mask=chunk["attention_mask"].to(
                    ctx.device_s, non_blocking=True
                ),
                return_dict=True,
                output_hidden_states=False,
            )
            encoded_texts += int(chunk["input_ids"].size(0))
            encoded_tokens += int(chunk["attention_mask"].sum())

            chunk_embeddings_single.append(chunk_out.last_hidden_state[:, 0, :])

        ctx.encoded_texts_total = getattr(ctx, "encoded_texts_total", 0) + encoded_texts
        ctx.encoded_tokens_total = (
            getattr(ctx, "encoded_tokens_total", 0) + encoded_tokens
        )

        S_cls1 = s_out1.last_hidden_state[:, 0, :]
        S_candidates = torch.cat(chunk_embeddings_single, dim=0).index_select(
            0, batch_s["candidate_inverse"]
        )

        loss, metrics = ctx.criterion(
            anchor_embeddings=S_cls1,
            candidate_embeddings=S_candidates,
            teacher_probs=batch_s["teacher_probs"],
            candidate_idx=batch_s.get("candidate_idx"),
            anchor_idx=batch_s.get("idx"),
            geometry_head_probs=batch_s.get("geometry_head_probs"),
            geometry_tail_positions=batch_s.get("geometry_tail_positions"),
            geometry_tail_mass=batch_s.get("geometry_tail_mass"),
        )
        loss = loss.float()

    if not is_finite(loss):
        raise RuntimeError(
            f"GGPKD loss NaN/Inf at epoch={ctx.current_epoch} step={ctx.current_step}"
        )

    ctx.scaler.scale(loss).backward()
    ctx.scaler.unscale_(ctx.optimizer)
    if not grads_are_finite(ctx.optimizer):
        ctx.optimizer.zero_grad(set_to_none=True)
        ctx.scaler.update()
        # Advance the schedule even on a skipped update: it was built for
        # len(train_loader) * epochs steps, so returning early here leaves
        # the LR permanently behind the cosine curve it was sized for.
        ctx.scheduler.step()
        return loss, {**metrics, "skip": "grad_inf"}

    # Every parameter the optimizer will move, not just the student's: a ceiling
    # that skips a param group is not a ceiling.
    clipped = [
        p
        for group in ctx.optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    grad_norm = torch.nn.utils.clip_grad_norm_(clipped, MAX_GRAD_NORM)
    metrics["grad_norm"] = float(grad_norm)
    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    assert_module_parameters_finite(
        ctx.model_student,
        f"GGPKD student after optimizer step "
        f"(epoch={ctx.current_epoch}, step={ctx.current_step})",
    )
    ctx.scheduler.step()

    return loss, metrics
