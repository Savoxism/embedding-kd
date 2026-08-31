"""Stella's two-stage training protocol.

Stella is the only method whose run is not one loop: stage 1 trains the pooled
representation, stage 2 continues with a different objective and its own epoch
budget and scheduler. That made `train()` an `if` with two unrelated bodies, so
the branch lives here and `train()` dispatches to it.

Reads from the distiller context: config, model_student, optimizer, scheduler,
criterion, current_stage, current_epoch, train_loader, and the training and
evaluation methods it drives.
"""

from collections import deque

import torch
import torch.optim as optim
from transformers import get_scheduler

from src.distill.checkpointing import save_checkpoint


def train(ctx) -> None:
    cfg = ctx.config
    print("\n" + "=" * 70)
    print("Starting Stella 2-Stage Training...")
    print("=" * 70)
    print(f"Student: {cfg.student_model_name}")
    print(f"Teacher: {cfg.teacher_model_name}")
    print(f"Stage 1 epochs: {cfg.epochs_stage1}")
    print(f"Stage 2 epochs: {cfg.epochs_stage2}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("STAGE 1: Freeze backbone + fc2,3,4, train fc1 only")
    print("=" * 70)

    for p in ctx.model_student.backbone.parameters():
        p.requires_grad = False
    for head in [
        ctx.model_student.fc2,
        ctx.model_student.fc3,
        ctx.model_student.fc4,
    ]:
        for p in head.parameters():
            p.requires_grad = False

    print("Frozen: backbone, fc2, fc3, fc4")
    print("Trainable: fc1")

    ctx.current_stage = 1
    for epoch in range(cfg.epochs_stage1):
        avg_loss = ctx.train_epoch(epoch)
        ctx.log_experiment_record(
            {"stage": 1, "train": ctx.last_epoch_metrics}
        )

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(ctx, epoch, {"loss": avg_loss})

    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETED!")
    print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("STAGE 2: Unfreeze all, train full model")
    print("=" * 70)

    for p in ctx.model_student.parameters():
        p.requires_grad = True

    print("Unfrozen: all parameters")
    print("Trainable: backbone, fc1, fc2, fc3, fc4")

    ctx.optimizer = optim.AdamW(
        ctx.model_student.parameters(), lr=cfg.learning_rate
    )
    ctx.scheduler = get_scheduler(
        "cosine",
        optimizer=ctx.optimizer,
        num_warmup_steps=int(len(ctx.train_loader) * cfg.warmup_ratio),
        num_training_steps=len(ctx.train_loader) * cfg.epochs_stage2,
    )

    ctx.step_times = []
    ctx.ma_window = deque(maxlen=50)

    ctx.current_stage = 2
    for epoch in range(cfg.epochs_stage2):
        avg_loss = ctx.train_epoch(epoch)
        validation_results = None

        print("\n" + "=" * 60)
        print(f"Evaluation after Stage2 Epoch {epoch + 1}")
        print("=" * 60)

        try:
            validation_results = ctx.evaluate("validation")
        except Exception as e:
            print(f"Warning: Evaluation failed with error: {e}")
            print("Continuing training...")

        print("=" * 60 + "\n")
        ctx.log_experiment_record(
            {
                "stage": 2,
                "train": ctx.last_epoch_metrics,
                "validation": validation_results,
            }
        )

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(ctx, epoch, {"loss": avg_loss})

    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETED!")
    print("=" * 70)

    save_checkpoint(ctx, cfg.epochs_stage2 - 1, {"loss": avg_loss})
    try:
        test_results = ctx.evaluate("test")
        ctx.log_experiment_record({"stage": 2, "test": test_results})
    except Exception as e:
        print(f"Warning: Final test evaluation failed with error: {e}")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
