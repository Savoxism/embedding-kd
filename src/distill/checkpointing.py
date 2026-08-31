"""Checkpoint and final-weight persistence.

Both writers are idempotent per epoch: the distiller can reach them from more
than one place in a run (end-of-epoch, end-of-training, the Stella stage
boundaries), and writing the same epoch twice used to produce duplicate files.

Reads from the distiller context: config, model_student, optimizer, scheduler,
criterion, proj_s2t, task_head, best_loss, and the two `_saved_*_epochs` sets.
"""

import os
import shutil
import tempfile
from pathlib import Path

import torch


def save_checkpoint(ctx, epoch: int, metrics: dict | None = None):
    cfg = ctx.config
    if not cfg.save_dir:
        return
    if epoch in ctx._saved_checkpoint_epochs:
        print(
            f"Checkpoint for epoch {epoch + 1} already saved; skipping duplicate."
        )
        return
    os.makedirs(cfg.save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": ctx.model_student.state_dict(),
        "optimizer_state_dict": ctx.optimizer.state_dict(),
        "scheduler_state_dict": ctx.scheduler.state_dict(),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
    }

    if ctx.proj_s2t is not None:
        checkpoint["proj_s2t_state_dict"] = ctx.proj_s2t.state_dict()

    if ctx.criterion is not None and hasattr(ctx.criterion, "state_dict"):
        checkpoint["criterion_state_dict"] = ctx.criterion.state_dict()
    if ctx.task_head is not None:
        checkpoint["task_head_state_dict"] = ctx.task_head.state_dict()

    if metrics:
        checkpoint["metrics"] = metrics

    path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    print(f"Done save_checkpoint for epoch {epoch + 1}")

    if cfg.save_best and metrics and "loss" in metrics:
        if not hasattr(ctx, "best_loss") or metrics["loss"] < ctx.best_loss:
            ctx.best_loss = metrics["loss"]
            best_path = os.path.join(cfg.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    save_student_weights(ctx, epoch)
    ctx._saved_checkpoint_epochs.add(epoch)


def save_student_weights(ctx, epoch: int):
    weights_dir = getattr(ctx.config, "weights_dir", None)
    if not weights_dir:
        return
    if epoch in ctx._saved_student_weight_epochs:
        print(
            f"Student weights for epoch {epoch + 1} already saved; "
            "skipping duplicate."
        )
        return

    destination_dir = Path(weights_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"student_epoch_{epoch + 1}.pt"
    destination_tmp = destination.with_suffix(".pt.tmp")
    payload = {
        "epoch": epoch + 1,
        "student_model_name": ctx.config.student_model_name,
        "teacher_model_name": ctx.config.teacher_model_name,
        "model_state_dict": ctx.model_student.state_dict(),
    }

    local_dir = Path(ctx.config.save_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    file_descriptor, local_tmp_name = tempfile.mkstemp(
        prefix=f".student_epoch_{epoch + 1}_",
        suffix=".pt",
        dir=local_dir,
    )
    os.close(file_descriptor)
    local_tmp = Path(local_tmp_name)

    try:
        torch.save(payload, local_tmp)
        shutil.copy2(local_tmp, destination_tmp)
        os.replace(destination_tmp, destination)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise OSError(
                f"Saved student weights are missing or empty: {destination}"
            )
    finally:
        local_tmp.unlink(missing_ok=True)
        destination_tmp.unlink(missing_ok=True)

    print(f"Student weights saved: {destination}")
    ctx._saved_student_weight_epochs.add(epoch)
