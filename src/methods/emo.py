"""EMO: optimal-transport embedding distillation with attention transfer."""

from types import SimpleNamespace

from torch import nn

from config import EMOConfig
from src.criterions.emo_embedding_distillation import EMODistillation
from src.distill.steps.standard import step
from src.methods.spec import MethodSpec
from src.methods.support import attach_parameters


def build_task_head(ctx, df):
    if "label" not in df.columns:
        return None
    hidden_size = ctx.model_student.config.hidden_size
    num_labels = int(df["label"].nunique())
    if ctx.config.task_type == "single_cls":
        return nn.Linear(hidden_size, num_labels).to(ctx.device_s)
    if ctx.config.task_type == "pair_cls":
        # [u, v, |u - v|, u * v]
        return nn.Linear(hidden_size * 4, num_labels).to(ctx.device_s)
    return None


def build_criterion(ctx, config):
    criterion = EMODistillation(
        d_teacher=ctx.model_teacher.config.hidden_size,
        d_student=ctx.model_student.config.hidden_size,
        k_layers=getattr(config, "k_layers", 1),
        alpha_ot=getattr(config, "alpha_ot", 0.1),
        max_iter=getattr(config, "max_iter_ot", 100),
        teacher_special=getattr(config, "teacher_special_token", "<s>"),
        student_special=getattr(config, "student_special_token", "[CLS]"),
    ).to(ctx.device_s)
    attach_parameters(ctx, criterion.parameters(), config.learning_rate)
    print("EMO criterion initialized and added to optimizer")
    return criterion


def kd_loss(ctx, t):
    """Optimal-transport embedding loss, averaged over both views when present."""
    cfg = ctx.config
    # EMO's criterion wants objects with `.last_hidden_state` and
    # `.attentions`; the encoder outputs cannot be reused directly because
    # the teacher's tensors have already been moved to the student device.
    teacher_outputs = SimpleNamespace(
        last_hidden_state=t.T_last1, attentions=t.T_atts
    )
    student_outputs = SimpleNamespace(
        last_hidden_state=t.S_last1, attentions=t.s_out1.attentions
    )

    att_loss_weight = getattr(cfg, "att_loss_weight", 0.1)
    ot_loss_weight = getattr(cfg, "ot_loss_weight", 1.0)

    kd_loss, kd_metrics = ctx.criterion.compute_emo_loss(
        teacher_outputs=teacher_outputs,
        student_outputs=student_outputs,
        input_ids_tea=t.batch_t["input_ids1_tea"].to(ctx.device_s),
        input_ids_stu=t.batch_s["input_ids1_stu"],
        attention_mask_tea=t.batch_t["attention_mask1_tea"].to(ctx.device_s),
        attention_mask_stu=t.batch_s["attention_mask1_stu"],
        tok_teacher=ctx.tok_teacher,
        tok_student=ctx.tok_student,
        att_loss_weight=att_loss_weight,
        ot_loss_weight=ot_loss_weight,
    )
    if t.S_last2 is not None and t.T_last2 is not None:
        teacher_outputs2 = SimpleNamespace(
            last_hidden_state=t.T_last2, attentions=t.T_atts2
        )
        student_outputs2 = SimpleNamespace(
            last_hidden_state=t.S_last2, attentions=t.s_out2.attentions
        )
        kd_loss2, kd_metrics2 = ctx.criterion.compute_emo_loss(
            teacher_outputs=teacher_outputs2,
            student_outputs=student_outputs2,
            input_ids_tea=t.batch_t["input_ids2_tea"].to(ctx.device_s),
            input_ids_stu=t.batch_s["input_ids2_stu"],
            attention_mask_tea=t.batch_t["attention_mask2_tea"].to(ctx.device_s),
            attention_mask_stu=t.batch_s["attention_mask2_stu"],
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
    loss = w_task * t.loss_task + alpha_kd * kd_loss

    metrics = {
        "loss_total": loss.item(),
        "t.loss_task": t.loss_task.item(),
        **t.task_metrics,
        **kd_metrics,
    }
    return loss, metrics


SPEC = MethodSpec(
    name="emo",
    config_cls=EMOConfig,
    step=step,
    kd_loss=kd_loss,
    needs_attentions=True,
    supervised_task_loss=True,
    build_task_head=build_task_head,
    build_criterion=build_criterion,
)
