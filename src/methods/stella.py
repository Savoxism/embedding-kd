"""Stella: a multi-head student trained in two stages."""

from config import StellaConfig
from src.criterions.stella_distillation import stella_stage1_loss, stella_stage2_loss
from src.distill.stella_trainer import train as train_loop
from src.distill.steps.standard import step
from src.methods.spec import MethodSpec


def build_student(ctx):
    # Not an AutoModel: the student carries four projection heads, and stage 1
    # trains only the first of them.
    from src.models import StellaModel

    cfg = ctx.config
    print(f"Loading Stella student model: {cfg.student_model_name}")
    ctx.current_stage = 1
    return StellaModel(
        cfg.student_model_name,
        output_dim1=getattr(cfg, "output_dim1", 1024),
        pooling=getattr(cfg, "pooling", "cls"),
        output_dim2=getattr(cfg, "output_dim2", 512),
        output_dim3=getattr(cfg, "output_dim3", 256),
        output_dim4=getattr(cfg, "output_dim4", 128),
    )


# Stella's losses are free functions called from the shared step, so there is no
# criterion object to build.


def kd_loss(ctx, t):
    """Stage-1 head alignment, or the stage-2 multi-head objective."""
    cfg = ctx.config
    if ctx.current_stage == 1:
        S_emb = t.s_out1["fc1"]
        loss, metrics = stella_stage1_loss(
            S_emb,
            t.T_cls1,
            w_cos=getattr(cfg, "w_cos_stage1", 10.0),
            w_sim=getattr(cfg, "w_sim_stage1", 200.0),
            w_tri=getattr(cfg, "w_tri_stage1", 20.0),
        )
    else:
        loss, metrics = stella_stage2_loss(
            t.S_cls1,
            t.S_cls2,
            t.s_out1["fc1"],
            t.s_out1["fc2"],
            t.s_out1["fc3"],
            t.s_out1["fc4"],
            t.T_cls1,
            temperature=cfg.temperature,
            w_task=cfg.w_task,
            w_cos=getattr(cfg, "w_cos_stage2", 10.0),
            w_sim=getattr(cfg, "w_sim_stage2", 200.0),
            w_tri=getattr(cfg, "w_tri_stage2", 20.0),
        )
    return loss, metrics


SPEC = MethodSpec(
    name="stella",
    config_cls=StellaConfig,
    step=step,
    kd_loss=kd_loss,
    student_returns_pooled=True,
    build_student=build_student,
    train_loop=train_loop,
)
