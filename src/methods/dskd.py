"""DSKD: dual-space alignment with trainable projections."""

from config import DSKDConfig
from src.criterions.dual_space_kd import DualSpaceKD
from src.distill.steps.standard import step
from src.methods.spec import MethodSpec
from src.methods.support import attach_parameters


def build_criterion(ctx, config):
    criterion = DualSpaceKD(
        student_dim=ctx.model_student.config.hidden_size,
        teacher_dim=ctx.model_teacher.config.hidden_size,
        w_task=config.w_task,
        alpha_dtw=config.alpha_dtw,
    ).to(ctx.device_s)
    attach_parameters(ctx, criterion.parameters(), config.learning_rate)
    print("DSKD criterion initialized and added to optimizer")
    return criterion


def kd_loss(ctx, t):
    """Dual-space alignment; the criterion returns the combined loss."""
    cfg = ctx.config
    mask_s1 = t.batch_s["attention_mask1_stu"]
    mask_t1 = t.batch_t["attention_mask1_tea"].to(ctx.device_s)

    spec_s1 = t.batch_s.get("special_tokens_mask1_stu", None)
    spec_t1 = t.batch_t.get("special_tokens_mask1_tea", None)
    if spec_t1 is not None:
        spec_t1 = spec_t1.to(ctx.device_s)

    loss, metrics = ctx.criterion.compute_dskd_loss(
        S_last=t.S_last1,
        T_last=t.T_last1,
        S_cls=t.S_cls1,
        T_cls=t.T_cls1,
        mask_student=mask_s1,
        mask_teacher=mask_t1,
        task_loss=t.loss_task,
        special_tokens_mask_student=spec_s1,
        special_tokens_mask_teacher=spec_t1,
        device=ctx.device_s,
    )
    return loss, metrics


SPEC = MethodSpec(
    name="dskd",
    config_cls=DSKDConfig,
    step=step,
    kd_loss=kd_loss,
    build_criterion=build_criterion,
)
