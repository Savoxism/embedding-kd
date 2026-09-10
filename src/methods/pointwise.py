"""Pointwise embedding KD: the batch study's measurement floor.

Shares the RKD training step -- one student pass, one cached teacher vector, one
criterion call -- because the two differ only in the criterion. It is registered
as a method rather than bolted onto GGPKD because the GGPKD criterion has no way
to express an objective with no relational term at all.
"""

from config.pointwise_config import PointwiseConfig
from src.criterions.pointwise_distillation import PointwiseDistillation
from src.distill.steps.rkd import step
from src.methods.spec import MethodSpec
from src.methods.support import attach_parameters


def build_criterion(ctx, config):
    student_dim = int(ctx.model_student.config.hidden_size)
    teacher_dim = int(ctx.teacher_cls_all.size(-1))
    criterion = PointwiseDistillation(
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        task_weight=config.w_task,
        eps=config.eps_norm,
    ).to(ctx.device_s)
    trainable = [p for p in criterion.parameters() if p.requires_grad]
    if trainable:
        # The projection is trained with the student, at the same learning rate.
        # `attach_parameters` also rebuilds the schedule, which a bare
        # `add_param_group` would leave holding one lr_lambda for two groups.
        attach_parameters(ctx, trainable, config.learning_rate)
    print(
        "Pointwise criterion initialized: "
        f"student_dim={student_dim}, teacher_dim={teacher_dim}, "
        f"projection={'identity' if not trainable else 'linear'}, "
        f"task={config.w_task}"
    )
    return criterion


SPEC = MethodSpec(
    name="pointwise",
    config_cls=PointwiseConfig,
    step=step,
    uses_teacher_cache=True,
    # False on purpose, and it is the property under test: this objective relates
    # no two examples, so it takes a short final batch without changing meaning
    # and its result cannot depend on who shares a batch with whom.
    batch_relational=False,
    build_criterion=build_criterion,
)
