"""TALAS: teacher-anchored layer distillation under a SAM perturbation."""

from pytorch_optimizer import SAM
from torch import optim

from config import TALASConfig
from src.criterions.teacher_anchor_kd import TeacherAnchorKD
from src.distill.steps.talas import step
from src.methods.spec import MethodSpec


def _student_hidden_state_count(model) -> int:
    """How many tensors `output_hidden_states=True` will return.

    A transformer returns the embedding output plus one tensor per layer. This
    used to be read as `len(out.hidden_states)` after the first forward pass,
    which is why the criterion, the optimizer and the scheduler were all built
    inside the training step; the encoder config states the same number before
    any batch exists.
    """
    layers = getattr(model.config, "num_hidden_layers", None)
    if layers is None:
        raise ValueError(
            "TALAS needs the student's layer count; its config has no "
            "`num_hidden_layers`. Set it on the config of a custom student."
        )
    return int(layers) + 1


def build_criterion(ctx, config):
    """Build the criterion, then the SAM optimizer and schedule that wrap it.

    TALAS replaces the optimizer rather than adding a param group to it: SAM
    needs both the student and the criterion's projection heads under the same
    perturbation. `setup_training` has already built the default AdamW at this
    point; it is discarded here, unused and unstepped.
    """
    d_s = ctx.model_student.config.hidden_size
    d_t = int(ctx.teacher_cls_all.shape[-1])
    num_layers = _student_hidden_state_count(ctx.model_student)

    criterion = TeacherAnchorKD(
        student_dim=d_s,
        teacher_dim=d_t,
        num_layers=num_layers,
        last_layer_idx=config.last_layer_idx,
        start_rkd=config.start_rkd,
        w_task=config.w_task,
        w_kd=config.w_kd,
        w_struct=config.w_struct,
        eps_norm=config.eps_norm,
    ).to(ctx.device_s)

    ctx.optimizer = SAM(
        [
            {
                "params": ctx.model_student.parameters(),
                "lr": config.learning_rate,
                "weight_decay": 0.01,
            },
            {
                "params": criterion.parameters(),
                "lr": config.learning_rate,
                "weight_decay": 0.01,
            },
        ],
        optim.AdamW,
        rho=getattr(config, "rho", 0.05),
        adaptive=True,
    )
    ctx.scheduler = ctx.build_scheduler()

    print(
        f"Initialized TeacherAnchorKD: {d_s} -> {d_t}, num_layers={num_layers}, "
        f"last_layer_idx={config.last_layer_idx}, start_rkd={config.start_rkd}"
    )
    print(f"Initialized SAM optimizer with rho={getattr(config, 'rho', 0.05)}")
    return criterion


SPEC = MethodSpec(
    name="talas",
    config_cls=TALASConfig,
    step=step,
    uses_teacher_cache=True,
    build_criterion=build_criterion,
)
