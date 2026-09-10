from src.criterions.ggpkd_distillation import RELATION_TARGETS
from src.data_utils.batch_samplers import BATCH_SAMPLERS
from src.ggpkd.graph_builder import KNN_MODES
from src.ggpkd.policy import SUPPORT_POLICIES, TRUNCATION_TOLERANCE

from .base_config import BaseConfig


class GGPKDConfig(BaseConfig):
    distill_method = "ggpkd"

    # Keep the no-CLI defaults aligned with the pair selected in
    # notebooks/train_colab_topk_no_neg.ipynb.
    student_model_name = "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "float32"

    student_special_token = "##"
    teacher_special_token = "G"

    # ---- Objective -----------------------------------------------------------
    # No student temperature on the diffusion ladder is a free parameter. The
    # criterion rejects scale_temps / broad_scale_temps / row_temp /
    # direct_student_temp and derives all of them:
    #   tau_1(i) = tau_i     the r=1 target IS the transition row, so the student
    #                        reuses the per-row bandwidth stored in the graph;
    #   tau_r = sqrt(r) tau_1   the spread of a diffusion grows as sqrt of its
    #                        time, so scale r is matched at the resolution its own
    #                        target already has. In the fixed-bandwidth baseline
    #                        at 0.05 this is
    #                        (0.0707, 0.200) for r = 2, 4 -- the values that were
    #                        previously written out by hand as (0.07, 0.10);
    #   tau_row(j) = tau_j   row targets are transition rows, so each supervised
    #                        row reuses its stored graph bandwidth;
    #   direct scale         one temperature (direct_temp) on both teacher and
    #                        student side (Hinton et al. 2015 convention).
    # direct_temp is the only student temperature left to choose. In-batch
    # sharing is part of the method definition and is always enabled when corpus
    # indices are available.

    # ---- Direct Scale (r=0) --------------------------------------------------
    # The ambient group is assigned the same total weight as the complete graph
    # group inside the criterion; it is not configurable.
    # The notebook requests 0, which derives this value at startup as the median
    # entropic-affinity bandwidth of the graph. The ambient target is the
    # transition-row construction with sparsification removed, so the graph's
    # typical bandwidth is its natural scale and no fixed temperature has to be
    # retuned across teachers.
    direct_temp = 0.0

    # ---- Ablation switches ---------------------------------------------------
    # Every one of these sits at the method's value. They exist so an ablation
    # arm is one CLI flag away from the full model instead of a branch, and so a
    # run manifest records which arm produced a number.
    #
    # support_policy  which columns the diffusion quota is spent on (S1):
    #                 topk (method) / proportional / uniform / local_topk.
    #                 local_topk is only for the clean no-diffusion control: it
    #                 selects the quota under P^1 while preserving the full
    #                 artifact and therefore the method's graph/ambient balance.
    # relation_target what the selected columns are supervised *against* (S3):
    #                 diffusion (method, composed multi-scale transition rows) or
    #                 direct (the teacher's raw cosine over the same columns).
    # use_ambient     the r=0 scale (S4). False drops it entirely; the criterion
    #                 then sees no teacher bank, which is what "no ambient" means.
    # knn_mode        which retrieval edges survive into the graph (G1):
    #                 mutual (method) / directed / symmetrized.
    # batch_local     the S1 baseline (batch-local relational KD): no graph, no
    #                 candidate draw, no rows -- each anchor is scored against the
    #                 texts that happen to share its minibatch. Implies
    #                 relation_target="ambient_only", which main.py sets for it.
    #                 Deliberately NOT encoder-budget-matched: it encodes 2B texts
    #                 per step against the method's B + unique candidates, and that
    #                 gap is the point, not a flaw. The budget-matched counterpart
    #                 is diffusion_quota=0 with the whole quota spent on uniform
    #                 corpus draws, which needs no flag of its own.
    support_policy = "topk"
    relation_target = "diffusion"
    use_ambient = True
    knn_mode = "mutual"
    batch_local = False

    # ---- Motivation study: batch composition, edge holdout -------------------
    # These three exist for the controlled studies in scripts/exp/ and sit at the
    # method's value, so a normal run is unaffected by their presence.
    #
    # batch_sampler       how a mini-batch is composed (E1, batch intervention).
    #                     `random` is the method and the only setting under which
    #                     the batches are i.i.d.; `teacher_neighbor` fills each
    #                     batch from one teacher neighbourhood, `teacher_diverse`
    #                     spreads it across distant ones. Both are label-free.
    #                     They exist to test whether batch composition changes the
    #                     supervision objective, which it can only do for a loss
    #                     whose support is the batch.
    # holdout_edge_frac   fraction of teacher graph edges withheld from every
    #                     training support (E3). The withheld edges are stored in
    #                     the artifact so a post-hoc evaluation can ask whether the
    #                     student recovered relations it was never supervised on.
    #                     0 is the method: nothing withheld.
    # holdout_seed        which edges. Separate from `seed` on purpose: the split
    #                     must be identical across seeds and across arms, or the
    #                     held-out evaluation is not measuring the same relations.
    batch_sampler = "random"
    holdout_edge_frac = 0.0
    holdout_seed = 12345

    # ---- Teacher Graph -------------------------------------------------------
    graph_k = 200
    # Bandwidth. Each transition row uses tau_i = s_i(1) - s_i(k): the similarity
    # span of its own retrieved neighbourhood, so the k-th neighbour sits exactly one
    # nat below the nearest for every node. This replaced a target-perplexity solve,
    # for two reasons. It keeps the property that ruled out a single fixed
    # temperature -- the row is exactly invariant to s -> a*s + b, because the
    # bandwidth scales with the similarities and a softmax is shift-invariant -- and
    # it costs one subtraction instead of a per-row bisection. The solve it replaced
    # also did not deliver what it promised: it ran on the mutual-filtered neighbour
    # list, where 18.9% of the production corpus had degree at or below perplexity
    # 30, so those rows never reached the target entropy and were solved against
    # their own ceiling instead. There is no target to miss here.
    #
    # graph_k therefore sets both the neighbourhood and its sharpness. That is one
    # knob doing two jobs: a sweep over it cannot separate the two effects, which is
    # the price of not having a second constant.
    fixed_bandwidth = False
    # Sorted, unique, and starting at 1. All three are enforced: the artifact stores
    # its scales sorted, and the temperature ladder is anchored to the r=1 target
    # being the transition row.
    # One-hop transition matching is the paper default. Broader {1,2} and
    # {1,2,4} ladders are reported as the radius ablation rather than being
    # bundled into the canonical method.
    diffusion_scales = (1,)
    # Within the graph group omega_r is proportional to 1/r. The graph group is
    # normalized to total weight 1 and matched by ambient weight 1, so Table 4
    # changes radius without changing the ambient--graph balance.

    # ---- Row Supervision -----------------------------------------------------
    # L_row promotes the teacher-selected pool columns (the diffusion support, not
    # the hard/uniform negatives) to auxiliary rows and matches each one's complete
    # available transition row, weighted uniformly. Batch anchors are excluded:
    # L_rel already matches their transition row as its r=1 target. The row set is a
    # deterministic function of the candidate pool, so this term costs no selection
    # hyperparameter -- row_weight is the only knob L_row has.
    #
    # Four alternatives were implemented and measured on Qwen3-0.6B -> MiniLMv2-H384
    # (seed 42, 5 epochs, row_weight 1.0); all lost and were deleted. The code is in
    # git history at b1f683b and earlier, and the numbers are in
    # docs/experiments/qwen-minilm-tuning.md:
    #
    #   non-backtracking walk selection   74.88, ties uniform closure but costs
    #                                     num_walks + walk_length, and the sampler
    #                                     had to inject visited nodes into the draw
    #   weight by exposed mass m_B(j)     74.76, squares a selection bias that is
    #                                     already mass-proportional
    #   weight by 1/c_B(j)                inert, c_B(j)=1 for ~99% of rows at this
    #                                     corpus and batch size
    #   ambient r=0 term per row          -0.30 out-of-domain, the benchmarks it was
    #                                     meant to calibrate
    #
    # uniform closure:                    74.86 at row_start_epoch 2, 74.82 at 1
    row_weight = 1.0
    # Human-facing, one-based epoch number. At 1 the knob is inert: L_row is on for
    # every epoch, and the curriculum disappears along with the parameter. The
    # warm-up existed because walk selection produced stochastic, noisy auxiliary
    # rows worth withholding for an epoch; the promoted columns are not noisy, and
    # the epoch-1 diagnostics of the start-at-1 run match every later epoch (same
    # ~843 rows at the same 0.44 exposed mass) with a lower final loss_rel.
    row_start_epoch = 1

    # ---- Truncation ----------------------------------------------------------
    # Every capacity in the build is the same operation: keep a subset S of a
    # probability row and renormalize. Discarding mass delta gives exactly
    # TV(p, ptilde) = delta and KL(ptilde || p) = -log(1 - delta), so this single
    # tolerance bounds the perturbation of the targets *in nats* -- the units of
    # the loss. At 1% that is <= 0.01 nats per truncation against a loss around
    # 0.84, and it compounds to at most r * tolerance across the r-step lazy walk.
    #
    # This replaces pool_size, walk_keep_topk and walk_topk outright rather than
    # sitting alongside them: each anchor now keeps exactly as many nodes as the
    # tolerance requires and the arrays are allocated at the width the widest anchor
    # needed. The only remaining sizes are DIFFUSION_ROW_CAP / POOL_ROW_CAP in
    # graph_builder, which are memory guards -- the build reports
    # pool_capped_rows / diffusion_capped_rows if either binds before the tolerance
    # is met, and then the guarantee does not hold.
    truncation_tolerance = TRUNCATION_TOLERANCE

    # ---- Per-Epoch Candidate Sampling ---------------------------------------
    # None is the method: no sampling at all. The candidate set is the anchor's
    # whole truncated transition row -- every column the teacher put mass on and
    # nothing else -- so there is no budget, no draw, and no RNG. The set is a
    # deterministic function of the graph, identical in every epoch.
    #
    # This removed the last tuned quantity in the candidate path. It also removed
    # `support_policy` from the method: at full width, topk / proportional / uniform
    # return the same set, so that flag now only means something for an ablation arm
    # given a budget smaller than the row. An int still sets such a budget.
    #
    # Row width is the pool width; anchors with shorter rows are padded with their
    # own index, which `self_mask` removes from every softmax.
    diffusion_quota = None
    # The method draws no negatives. Every scored column is a column the teacher
    # put diffusion mass on, so candidate_size == diffusion_quota and the whole
    # relational budget is spent on relations the graph actually asserts.
    #
    # This removes the two quotas that were never derived from anything -- 40/26
    # was the best measured pair (75.29 on Qwen3-0.6B -> MiniLMv2-H384, graph v9,
    # seed 42) but the hard:random split was never shown flat, so they stood as
    # two tuned constants in a method whose other knobs are all derived.
    #
    # Two consequences to hold in view, because they cut in opposite directions:
    #
    # * The diffusion group no longer scores a single zero-target column. Its
    #   softmax now runs over the anchor's own support, where every column
    #   carries real teacher mass, so the false-zero gradient that motivated the
    #   ambient scale is gone from the own draw by construction. The audit metric
    #   `amb_mass_on_zero_diff` should now read ~0; if it does not, the draw is
    #   not what this comment claims.
    # * The shared pool loses the ~1,160 uniform-corpus texts per batch that were
    #   its only columns not drawn from someone's graph neighbourhood, dropping
    #   from ~4,445 to ~1,400. Scale r=0 still calibrates similarity levels across
    #   the batch, but over a column set that is now entirely local. That is the
    #   risk this change carries, and STS Spearman plus the pair-classification
    #   thresholds are where it would show up first -- they are precisely the
    #   benchmarks that read absolute cosine levels rather than per-anchor rank.
    #
    # A short support draw is padded with the anchor's own index rather than
    # backfilled with uniform draws; see GGPKDCandidateSampler.sample.
    hard_neg_k = 0
    random_neg_k = 0
    # candidate_size is derived as the sum of the three quotas, so it is now just
    # diffusion_quota. The ablation arms that need negatives -- `uniform_corpus`
    # / `no_graph_support` in Tables 2 and 3, which spends the entire budget on
    # uniform corpus draws -- still set these on the CLI; the machinery stays.
    # Canonical Top-k support is deterministic; the proportional control redraws
    # support per epoch.

    # ---- Corpus Columns ------------------------------------------------------
    # Which column is the graph node, and which defines "same source" for hard
    # negatives. Both were read by the distiller but declared nowhere, so they
    # could not be set through this class at all. None keeps the existing
    # behaviour: the anchor column is picked from task_type.
    ggpkd_anchor_column = None
    ggpkd_source_column = "source"

    # ---- Training Setup ------------------------------------------------------
    batch_size = 64
    epochs = 5
    # 2e-5 undertrains inside the fixed 5-epoch budget: validation avg was still
    # rising monotonically at epoch 5. At 3e-5 the test avg plateaus from epoch 2
    # (74.59 -> 75.19 -> ~flat) with stable grad norms, so the budget is actually
    # spent.
    learning_rate = 3e-5
    min_lr = 3e-6
    num_workers = 4
    # Per-epoch evaluation is off: it existed to answer whether a run converges
    # inside the 5-epoch budget, and that question is answered — at lr 3e-5 the
    # test avg plateaus from epoch 2 (74.59 -> 75.19 -> ~flat). Only the final
    # evaluation runs now; the per-epoch training means, geometry probe and
    # step_metrics.jsonl still record convergence without it. Set 1 to re-enable
    # when a change (new pair, new lr, new objective term) reopens the question.
    eval_every = 0

    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    # cache_teacher removed: nothing read it. Teacher caching is gated purely by
    # whether cache_path already exists on disk (distiller.py).
    cache_path = "cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/teacher_train.pt"
    ggpkd_cache_path = "cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/graph.pt"
    ggpkd_log_dir = "logs/ggpkd/qwen3_0_6b_to_minilmv2_h384"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = (
        "models/ggpkd/qwen3_0_6b_to_minilmv2_h384/base_w1_e1_qauto_lr3e-05_seed42"
    )
    weights_dir = (
        "models/ggpkd_weights/qwen3_0_6b_to_minilmv2_h384/"
        "base_w1_e1_qauto_lr3e-05_seed42"
    )
    final_weights_only = True

    # ---- Multi-Layer Spec ----------------------------------------------------
    # Defining both of these switches the distiller to its multi-layer GGPKD
    # branch. They are off, so that branch never runs, and with it every knob that
    # only that branch reads.
    # kd_teacher_layers = [12, 24, 36, 36]
    # kd_student_layers = [4, 8, 12, 12]

    # ---- Removed: earlier auxiliary objectives -------------------------------
    # The active objective is L_rel + row_weight * L_row inside the GGPKD
    # criterion.
    # lambda_ggpkd,
    # lambda_cosine, lambda_infonce,
    # lambda_simcse, simcse_temp, simcse_start_epoch and lambda_sim used to sit
    # here at 0. Every one of them is read only inside the multi-layer branch
    # above, so on this path they were unreachable: they printed in the run banner
    # and counted against the method's knob budget while doing nothing. Their
    # consumers all read them through getattr with the same defaults these lines
    # carried, so deleting them changes no behaviour. Re-adding one is only
    # meaningful together with the kd_*_layers pair.

    def validate(self):
        """Re-checkable invariants.

        Called from __init__, and again by main.py after the CLI overrides are
        applied -- those are plain attribute assignments on an already-built
        config, so without a second call every flag combination goes unchecked.
        An arm like `--relation_target direct --no_ambient` would then get as far
        as caching the teacher before the criterion refused it.
        """
        if self.row_weight < 0:
            raise ValueError("row_weight must be non-negative")
        if self.direct_temp < 0:
            raise ValueError("direct_temp must be positive, or 0 to derive it")
        if self.row_start_epoch < 1:
            raise ValueError("row_start_epoch must be at least 1")
        if self.diffusion_quota is not None and self.diffusion_quota < 1:
            raise ValueError(
                "diffusion_quota must be None (the whole transition row) or positive"
            )
        if self.support_policy not in SUPPORT_POLICIES:
            raise ValueError(
                f"support_policy must be one of {SUPPORT_POLICIES}, "
                f"got {self.support_policy!r}"
            )
        if self.relation_target not in RELATION_TARGETS:
            raise ValueError(
                f"relation_target must be one of {RELATION_TARGETS}, "
                f"got {self.relation_target!r}"
            )
        if self.knn_mode not in KNN_MODES:
            raise ValueError(
                f"knn_mode must be one of {KNN_MODES}, got {self.knn_mode!r}"
            )
        if self.batch_local:
            if self.relation_target not in ("ambient_only", "direct"):
                raise ValueError(
                    "batch_local forms no graph relations, so its target must be "
                    "read off the teacher bank: relation_target='ambient_only' "
                    "(one ambient temperature over the batch) or 'direct' (the "
                    f"anchor's own tau_i); got {self.relation_target!r}"
                )
            if self.batch_size < 2:
                raise ValueError(
                    "batch_local needs at least two texts per batch to have any "
                    f"relation at all; got batch_size={self.batch_size}"
                )
        if self.relation_target == "ambient_only" and not self.use_ambient:
            # `ambient_only` *is* scale r=0. Removing the scale leaves no term.
            # `direct` is deliberately not caught here any more: it reads the
            # teacher bank, which the criterion now receives independently of
            # whether scale r=0 is in the loss. That combination is the minimal
            # relational objective the controlled support study is built on.
            raise ValueError(
                "relation_target='ambient_only' is the ambient scale itself; it "
                "cannot be combined with use_ambient=False"
            )
        if self.holdout_edge_frac < 0.0 or self.holdout_edge_frac >= 1.0:
            raise ValueError(
                "holdout_edge_frac is the fraction of teacher edges withheld from "
                f"every training support; must be in [0, 1), got {self.holdout_edge_frac}"
            )
        if self.batch_sampler not in BATCH_SAMPLERS:
            raise ValueError(
                f"batch_sampler must be one of {BATCH_SAMPLERS}, "
                f"got {self.batch_sampler!r}"
            )
        if self.support_policy in ("corpus_uniform", "rewired"):
            # Off-graph columns carry diffusion mass exactly zero, so under the
            # method's own target these arms would optimize nothing at all. The
            # teacher's raw cosine is defined for every pair, which is what makes
            # a random-support arm a control rather than a deleted objective.
            if self.relation_target != "direct":
                raise ValueError(
                    f"support_policy={self.support_policy!r} draws columns the "
                    "graph puts no diffusion mass on, so it requires "
                    "relation_target='direct'; got "
                    f"{self.relation_target!r}"
                )
            if self.diffusion_quota is None:
                raise ValueError(
                    f"support_policy={self.support_policy!r} needs an explicit "
                    "--diffusion_quota: there is no transition row to take the "
                    "width from, and the arm is only a control when its support "
                    "size matches the teacher arm it is compared against"
                )
            if self.row_weight > 0:
                # L_row promotes the drawn columns to auxiliary rows. Under a
                # random draw those rows have almost no pool-exposed teacher
                # neighbours, so the term would quietly carry a different amount
                # of supervision in this arm than in the arm it is compared with.
                raise ValueError(
                    f"support_policy={self.support_policy!r} requires "
                    "row_weight=0: L_row's row set is derived from the support "
                    "draw, so leaving it on makes the arms differ in two things"
                )
