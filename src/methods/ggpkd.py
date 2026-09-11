"""GGPKD: relational distillation over the teacher's kNN transition rows.

The graph, the candidate pool and the deduplicated anchor corpus are this
method's alone. They used to occupy ~120 of `setup_data`'s 271 lines and three
private methods on the distiller, all of them reachable only through
`if distill_method == "ggpkd"`.
"""

import hashlib

import numpy as np
import pandas as pd
import torch

from config import GGPKDConfig
from src.criterions.ggpkd_distillation import GGPKDDistillation
from src.data_utils.ggpkd_dataset import GGPKDCollate, TextPairWithTeacherAndGGPKD
from src.distill.geometry import build_probe_index
from src.distill.steps.ggpkd import step
from src.ggpkd import GGPKDCandidateSampler, build_or_load_ggpkd_artifact
from src.ggpkd.policy import FIXED_BANDWIDTH_TEMP
from src.methods.spec import MethodSpec


def build_fixed_reference_indices(
    corpus_texts: list[str], reference_size: int
) -> np.ndarray:
    """Choose a reproducible, teacher-independent corpus reference set.

    Ranking texts by a namespaced SHA-256 digest behaves like a uniform sample,
    but is stable across training seeds and corpus row reordering. Exact duplicate
    anchors have already been removed by ``prepare_frame``.
    """
    if reference_size < 1:
        raise ValueError("reference_size must be at least 1")
    keyed = []
    for index, text in enumerate(corpus_texts):
        digest = hashlib.sha256(
            b"ggpkd-fixed-reference-v1\0" + str(text).encode("utf-8")
        ).digest()
        keyed.append((digest, index))
    keyed.sort()
    size = min(reference_size, len(keyed))
    return np.asarray([index for _, index in keyed[:size]], dtype=np.int64)


def fixed_reference_fingerprint(
    corpus_texts: list[str], reference_indices: np.ndarray
) -> str:
    """Fingerprint the selected texts, independent of their corpus row numbers."""
    hasher = hashlib.sha256()
    for index in reference_indices:
        encoded = corpus_texts[int(index)].encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()[:12]


def resolve_anchor_column(ctx, df: pd.DataFrame) -> str:
    cfg = ctx.config
    column = cfg.ggpkd_anchor_column
    if column is not None:
        if column not in df.columns:
            raise ValueError(
                f"ggpkd_anchor_column={column!r} is not a column of "
                f"{cfg.train_data_path} (have {list(df.columns)})"
            )
        return column

    if cfg.task_type == "single_cls":
        column = "text"
    elif cfg.task_type == "pair_cls":
        column = "premise"
    else:
        column = "sentence1"
    if column not in df.columns:
        raise ValueError(
            f"GGPKD needs column {column!r} for task_type={cfg.task_type!r}"
        )

    # The teacher graph is built over this column only. If a genuine second view
    # exists it is dropped, and doing that silently would leave the graph
    # describing a different object than the loss thinks it does.
    partner = {"pair_cls": "hypothesis", "pair_reg": "sentence2"}.get(cfg.task_type)
    if partner in df.columns and not df[column].equals(df[partner]):
        print(
            f"WARNING: GGPKD uses only {column!r}; {partner!r} differs from it "
            f"and is not distilled. Set ggpkd_anchor_column explicitly if that "
            f"is not what you want."
        )
    return column


def _dedup_frame(
    ctx, df: pd.DataFrame, anchor_column: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop exact duplicate anchors and report the surviving row positions.

    Two identical texts have cos(s_i, s_j) = 1 for every parameter setting, so
    their logit sits at the ceiling with no gradient while still consuming
    teacher mass and a candidate slot.
    """
    keep_positions = np.arange(len(df), dtype=np.int64)
    texts = df[anchor_column].astype(str)
    duplicated = texts.duplicated(keep="first").to_numpy()
    if not duplicated.any():
        print(f"GGPKD corpus: {len(df)} rows, no duplicate anchors")
        return df.reset_index(drop=True), keep_positions

    keep_positions = np.flatnonzero(~duplicated).astype(np.int64)
    deduped = df.iloc[keep_positions].reset_index(drop=True)
    print(
        f"GGPKD corpus dedup on {anchor_column!r}: "
        f"{len(df)} -> {len(deduped)} rows ({int(duplicated.sum())} exact duplicates removed)"
    )
    return deduped, keep_positions


def source_ids(ctx, df: pd.DataFrame) -> np.ndarray:
    column = ctx.config.ggpkd_source_column
    if column not in df.columns:
        print(
            f"GGPKD: no {column!r} column, hard negatives will not be "
            f"restricted to the same source corpus"
        )
        return np.zeros(len(df), dtype=np.int64)
    codes = pd.factorize(df[column].astype(str))[0].astype(np.int64)
    counts = pd.Series(codes).value_counts().to_dict()
    print(
        f"GGPKD sources: {len(counts)} distinct, sizes={sorted(counts.values(), reverse=True)}"
    )
    return codes


def prepare_frame(ctx, df: pd.DataFrame):
    """Resolve the anchor column and drop duplicate anchors."""
    ctx.ggpkd_anchor_column = resolve_anchor_column(ctx, df)
    return _dedup_frame(ctx, df, ctx.ggpkd_anchor_column)


def build_data(ctx, df: pd.DataFrame, teacher_cls: torch.Tensor):
    """Build the teacher graph, the candidate sampler, and the dataset over them.

    The graph is built after the teacher model has been freed, so the block-wise
    cosine pass in `build_or_load_ggpkd_artifact` has the teacher's VRAM to
    itself. (It only ever reads the cached embeddings, never the model.)
    """
    ctx.ggpkd_artifact = build_or_load_ggpkd_artifact(
        teacher_embeddings=teacher_cls,
        cache_path=ctx.config.ggpkd_cache_path,
        log_dir=ctx.config.ggpkd_log_dir,
        graph_k=ctx.config.graph_k,
        fixed_bandwidth=ctx.config.fixed_bandwidth,
        truncation_tolerance=ctx.config.truncation_tolerance,
        diffusion_scales=ctx.config.diffusion_scales,
        knn_mode=ctx.config.knn_mode,
        holdout_edge_frac=ctx.config.holdout_edge_frac,
        holdout_seed=ctx.config.holdout_seed,
        source_ids=source_ids(ctx, df),
    )
    ctx.ggpkd_sampler = GGPKDCandidateSampler(
        artifact=ctx.ggpkd_artifact,
        diffusion_quota=ctx.config.diffusion_quota,
        hard_neg_k=ctx.config.hard_neg_k,
        random_neg_k=ctx.config.random_neg_k,
        seed=ctx.config.seed,
        support_policy=ctx.config.support_policy,
    )
    anchor_texts = df[ctx.ggpkd_anchor_column].astype(str).tolist()
    reference_indices = None
    if ctx.config.calibration_mode in ("fixed_reference", "fixed_cosine"):
        reference_indices = build_fixed_reference_indices(
            anchor_texts, ctx.config.reference_size
        )
        reference_fingerprint = fixed_reference_fingerprint(
            anchor_texts, reference_indices
        )
        # build_data runs before telemetry writes run.json, so the concrete
        # corpus-defined R becomes part of the reproducibility manifest.
        ctx.config.reference_fingerprint = reference_fingerprint
        print(
            f"GGPKD {ctx.config.calibration_mode} calibration: "
            f"{reference_indices.size} corpus columns, "
            f"fingerprint={reference_fingerprint}"
        )
    # Rebuild the probe on the deduplicated anchor column. The set built
    # in setup_data was sampled from the pre-dedup frame, whose row
    # positions no longer index the teacher cache -- so pairing the two
    # there would silently report the Spearman of mismatched rows.
    probe_index = build_probe_index(len(anchor_texts), size=2048, seed=0)
    ctx.probe_texts = [anchor_texts[int(i)] for i in probe_index]
    ctx.probe_teacher = teacher_cls[torch.from_numpy(np.asarray(probe_index)).long()]
    ctx.train_ds = TextPairWithTeacherAndGGPKD(
        anchor_texts=anchor_texts,
        teacher_cls=teacher_cls,
        sampler=ctx.ggpkd_sampler,
        labels=df["label"].astype(int).tolist() if "label" in df.columns else None,
        batch_local=ctx.config.batch_local,
    )
    # The collate owns the tokenized corpus: anchors and candidates are
    # drawn from the same rows, so every text is tokenized once here
    # instead of ~candidate_size times per epoch in the workers.
    ctx.collate_fn = GGPKDCollate(
        ctx.tok_student,
        ctx.config.task_type,
        ctx.config.max_length,
        corpus_texts=anchor_texts,
        batch_local=ctx.config.batch_local,
        n_scales=len(ctx.config.diffusion_scales),
        reference_indices=reference_indices,
    )
    if ctx.config.batch_local:
        print(
            "GGPKD batch-local baseline: relations among the batch "
            f"only ({ctx.config.batch_size} texts), no candidate draw, no "
            "graph support, no auxiliary rows"
        )
    if ctx.ggpkd_sampler.no_negatives:
        print(
            "GGPKD graph rows draw no negatives: every graph-domain column "
            "has a nonzero transition target"
        )
    # Report the width the anchors actually get, not just the one the
    # config asked for. A quota above the pool fill is silently truncated
    # inside the draw, and reading only the requested number is how a
    # requested width of 500 was mistaken for the real width of 67.
    fill = (ctx.ggpkd_artifact["pool_indices"].numpy() >= 0).sum(axis=1)
    if ctx.ggpkd_sampler.full_pool:
        print(
            "GGPKD candidate set: the anchor's whole transition row "
            f"(row width {ctx.ggpkd_sampler.candidate_size}; real columns "
            f"mean={fill.mean():.1f} min={int(fill.min())} "
            f"max={int(fill.max())}), no sampling, fixed across epochs"
        )
    else:
        short = int((fill < ctx.ggpkd_sampler.diffusion_quota).sum())
        print(
            "GGPKD candidate sampling: "
            f"candidate_size={ctx.ggpkd_sampler.candidate_size} "
            f"(diffusion={ctx.ggpkd_sampler.diffusion_quota}, "
            f"hard={ctx.ggpkd_sampler.hard_neg_k}, "
            f"random={ctx.ggpkd_sampler.random_neg_k}), "
            f"support_policy={ctx.ggpkd_sampler.support_policy}"
        )
        if short:
            print(
                f"  WARNING: {short}/{fill.size} anchors "
                f"({short / fill.size:.1%}) hold fewer than "
                f"{ctx.ggpkd_sampler.diffusion_quota} columns; their draw "
                f"is padded, real mean width is {fill.clip(max=ctx.ggpkd_sampler.diffusion_quota).mean():.1f}"
            )
    return ctx.train_ds, ctx.collate_fn


def build_criterion(ctx, config):
    # `ctx.ggpkd_artifact` is set unconditionally by `build_data` before this
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
    # `use_ambient=False` is the S4 deletion arm. It used to be expressed by
    # withholding the teacher bank, which also removed the only way to compute a
    # `direct` target -- and the controlled support study needs exactly that
    # combination: no ambient scale (so nothing couples the anchors and batch
    # composition cannot reach the loss) with direct targets (so a column drawn
    # off-graph still carries a real teacher opinion). The bank is therefore
    # passed whenever some term reads it, and `use_ambient_scale` alone decides
    # whether scale r=0 is in the objective.
    needs_bank = config.calibration_mode != "none" or config.relation_target in (
        "direct",
        "ambient_only",
    )
    criterion = GGPKDDistillation(
        diffusion_scales=config.diffusion_scales,
        teacher_embeddings=ctx.teacher_cls_all if needs_bank else None,
        use_ambient_scale=config.use_ambient,
        calibration_mode=config.calibration_mode,
        direct_temp=config.direct_temp,
        row_weight=config.row_weight,
        relation_target=config.relation_target,
        row_temps=artifact["row_temps"],
        transition_neighbors=artifact["transition_neighbors"],
        transition_probs=artifact["transition_probs"],
    ).to(ctx.device_s)
    print(
        "GGPKD criterion initialized: "
        f"batch_local={config.batch_local}, "
        f"calibration={config.calibration_mode}, "
        f"relation_target={config.relation_target}, "
        f"row_weight={config.row_weight}"
    )
    return criterion


def on_epoch_start(ctx, epoch: int):
    """Gate the auxiliary row loss for this epoch."""
    cfg = ctx.config
    use_row = cfg.row_weight > 0 and epoch + 1 >= cfg.row_start_epoch
    if ctx.criterion is not None and hasattr(ctx.criterion, "use_row_loss"):
        ctx.criterion.use_row_loss = use_row
    if use_row:
        print(f"L_row is ENABLED for Epoch {epoch + 1}")


SPEC = MethodSpec(
    name="ggpkd",
    config_cls=GGPKDConfig,
    step=step,
    uses_teacher_cache=True,
    batch_relational=True,
    prepare_frame=prepare_frame,
    build_data=build_data,
    build_criterion=build_criterion,
    on_epoch_start=on_epoch_start,
)
