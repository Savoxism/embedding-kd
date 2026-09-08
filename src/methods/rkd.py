"""RKD-DA: Park et al.'s distance- and angle-wise relational distillation."""

from torch import optim

from config import RKDConfig
from src.criterions.relational_kd import RelationalKnowledgeDistillation
from src.distill.steps.rkd import step
from src.methods.spec import MethodSpec


def build_optimizer(ctx, parameters):
    # The paper's metric-learning setup: Adam with weight decay, not AdamW.
    return optim.Adam(
        parameters,
        lr=ctx.config.learning_rate,
        weight_decay=ctx.config.weight_decay,
    )


def build_scheduler(ctx):
    # Step decay at fixed epochs, as in the original schedule.
    cfg = ctx.config
    milestones = [
        int(epoch) * len(ctx.train_loader) for epoch in cfg.rkd_lr_decay_epochs
    ]
    return optim.lr_scheduler.MultiStepLR(
        ctx.optimizer,
        milestones=milestones,
        gamma=cfg.rkd_lr_decay_gamma,
    )


def build_criterion(ctx, config):
    criterion = RelationalKnowledgeDistillation(
        distance_weight=config.rkd_distance_weight,
        angle_weight=config.rkd_angle_weight,
        task_weight=config.w_task,
        eps=config.eps_norm,
    ).to(ctx.device_s)
    print(
        "RKD-DA criterion initialized: "
        f"distance={config.rkd_distance_weight}, "
        f"angle={config.rkd_angle_weight}, task={config.w_task}"
    )
    return criterion


SPEC = MethodSpec(
    name="rkd",
    config_cls=RKDConfig,
    step=step,
    uses_teacher_cache=True,
    batch_relational=True,
    build_criterion=build_criterion,
    build_optimizer=build_optimizer,
    build_scheduler=build_scheduler,
)
