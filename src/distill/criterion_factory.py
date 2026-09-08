"""Construction of the per-method criterion, and the optimizer wiring it needs.

Extracted from ``KnowledgeDistiller.__init__``, which was a 130-line if/elif
chain sitting in the middle of a constructor that also set up seeds, devices,
models, data and the optimizer. Nothing here reads distiller state that is not
listed in ``build_criterion``'s docstring, so a criterion can now be built --
and its optimizer wiring checked -- without standing up a training run.

**Why the optimizer wiring lives here.** Three of these criteria own trainable
parameters, and adding a param group *after* the scheduler exists leaves
``LambdaLR`` holding one ``lr_lambda`` for two groups; torch >= 2.6 zips them
with ``strict=True``, so the first ``scheduler.step()`` raises. Every method
that adds a group must therefore rebuild the scheduler in the same breath, and
keeping the two steps apart is what let CDM drift into that bug once already.
"""

from torch import nn

from src.criterions.contextual_dynamic_mapping import ContextualDynamicMapping
from src.criterions.dual_space_kd import DualSpaceKD
from src.criterions.emo_embedding_distillation import EMODistillation
from src.criterions.ggpkd_distillation import GGPKDDistillation
from src.criterions.relational_kd import RelationalKnowledgeDistillation
from src.ggpkd.policy import FIXED_BANDWIDTH_TEMP


def _attach(ctx, parameters, lr):
    """Give the optimizer a new param group and rebuild the schedule over it."""
    ctx.optimizer.add_param_group({"params": parameters, "lr": lr})
    ctx.scheduler = ctx._build_scheduler()


def _build_cdm(ctx, config):
    criterion = ContextualDynamicMapping(
        tok_student=ctx.tok_student,
        tok_teacher=ctx.tok_teacher,
        blending_model_special_token=config.teacher_special_token,
        base_model_special_token=config.student_special_token,
        w_task=config.w_task,
        alpha_dtw=config.alpha_dtw,
        debug_align=config.debug_align,
    )
    d_s = ctx.model_student.config.hidden_size
    d_t = ctx.model_teacher.config.hidden_size
    ctx.proj_s2t = nn.Linear(d_s, d_t, bias=False).to(ctx.device_s)
    _attach(ctx, ctx.proj_s2t.parameters(), config.learning_rate * 2)
    print(f"Initialized CDM projection layer: {d_s} -> {d_t}")
    return criterion


def _build_dskd(ctx, config):
    criterion = DualSpaceKD(
        student_dim=ctx.model_student.config.hidden_size,
        teacher_dim=ctx.model_teacher.config.hidden_size,
        w_task=config.w_task,
        alpha_dtw=config.alpha_dtw,
    ).to(ctx.device_s)
    _attach(ctx, criterion.parameters(), config.learning_rate)
    print("DSKD criterion initialized and added to optimizer")
    return criterion


def _build_emo(ctx, config):
    criterion = EMODistillation(
        d_teacher=ctx.model_teacher.config.hidden_size,
        d_student=ctx.model_student.config.hidden_size,
        k_layers=getattr(config, "k_layers", 1),
        alpha_ot=getattr(config, "alpha_ot", 0.1),
        max_iter=getattr(config, "max_iter_ot", 100),
        teacher_special=getattr(config, "teacher_special_token", "<s>"),
        student_special=getattr(config, "student_special_token", "[CLS]"),
    ).to(ctx.device_s)
    _attach(ctx, criterion.parameters(), config.learning_rate)
    print("EMO criterion initialized and added to optimizer")
    return criterion


def _build_rkd(ctx, config):
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


def _build_ggpkd(ctx, config):
    # `ctx.ggpkd_artifact` is set unconditionally in the data path before this
    # runs, so indexing it directly is right: a `.get()` fallback would hand the
    # criterion `None` and make it blame the graph artifact for what is really a
    # setup-ordering bug.
    artifact = ctx.ggpkd_artifact
    # --direct_temp 0 derives the last free student temperature from the graph
    # itself: the median entropic-affinity bandwidth. The ambient target is the
    # same softmax-of-cosines construction as the transition rows with the
    # sparsification removed, so the graph's own typical bandwidth is the natural
    # scale for it. Written back onto the config so the run manifest and banner
    # record the concrete value, exactly as derived diffusion_quota is.
    if config.direct_temp == 0.0:
        row_temps = artifact.get("row_temps")
        config.direct_temp = (
            float(row_temps.median()) if row_temps is not None else FIXED_BANDWIDTH_TEMP
        )
        print(
            f"Derived direct_temp={config.direct_temp:.4f} "
            "(median graph bandwidth; requested via --direct_temp 0)"
        )
    # `use_ambient=False` is the S4 deletion arm: withholding the bank is what
    # removes scale r=0, because the criterion derives `use_direct` from whether
    # it has teacher embeddings at all.
    criterion = GGPKDDistillation(
        diffusion_scales=config.diffusion_scales,
        teacher_embeddings=ctx.teacher_cls_all if config.use_ambient else None,
        direct_temp=config.direct_temp,
        row_weight=config.row_weight,
        relation_target=config.relation_target,
        row_temps=artifact["row_temps"],
        transition_neighbors=artifact["transition_neighbors"],
        transition_probs=artifact["transition_probs"],
    ).to(ctx.device_s)
    # No param group to add -- the criterion is parameter-free -- but the
    # schedule is rebuilt for symmetry with the methods above, which is how the
    # original code behaved.
    ctx.scheduler = ctx._build_scheduler()
    print(
        "GGPKD criterion initialized: "
        f"batch_local={config.batch_local}, "
        f"ambient={config.use_ambient}, "
        f"relation_target={config.relation_target}, "
        f"row_weight={config.row_weight}"
    )
    return criterion


# `talas` and `stella` are absent on purpose. TALAS builds its criterion lazily
# on the first batch, because it needs the student's layer count; Stella carries
# its losses as free functions. Both leave the criterion None here.
_BUILDERS = {
    "cdm": _build_cdm,
    "dskd": _build_dskd,
    "emo": _build_emo,
    "rkd": _build_rkd,
    "ggpkd": _build_ggpkd,
}


def build_criterion(ctx):
    """Return the criterion for ``ctx.config.distill_method``, or None.

    Reads from the distiller context: config, device_s, model_student,
    model_teacher, tok_student, tok_teacher, optimizer, teacher_cls_all,
    ggpkd_artifact, and `_build_scheduler`. May set `ctx.proj_s2t` and always
    replaces `ctx.scheduler` when it adds a param group.
    """
    builder = _BUILDERS.get(ctx.config.distill_method)
    return None if builder is None else builder(ctx, ctx.config)
