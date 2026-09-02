from .base_config import BaseConfig


class GGPKDConfig(BaseConfig):
    distill_method = "ggpkd"

    student_model_name = "google-bert/bert-base-uncased"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-4B"
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
    #                        (0.0707, 0.1000) for r = 2, 4 -- the values that were
    #                        previously written out by hand as (0.07, 0.10);
    #   tau_row(j) = tau_j   row targets are transition rows, so each supervised
    #                        row reuses its stored graph bandwidth;
    #   direct scale         one temperature (direct_temp) on both teacher and
    #                        student side (Hinton et al. 2015 convention).
    # direct_temp is the only student temperature left to choose. In-batch
    # sharing is part of the method definition and is always enabled when corpus
    # indices are available.

    # ---- Direct Scale (r=0) --------------------------------------------------
    # omega_0 is derived as omega_1 inside the criterion; it is not configurable.
    # Measured flat: 0.07 vs 0.10 on Qwen3-0.6B -> MiniLMv2-H384 (seed 42) was a
    # tie (75.25 vs 75.21), so the Hinton-convention default is kept rather than
    # tuned. This is the last student temperature that is still a free choice; 0
    # derives it at startup as the median entropic-affinity bandwidth of the graph
    # (the ambient target is the transition-row construction with the
    # sparsification removed, so the graph's typical bandwidth is its natural
    # scale). If the derived arm measures flat too, 0 becomes the default and no
    # student temperature anywhere in the method is a hyperparameter.
    direct_temp = 0.10  # shared by both sides of the direct scale; 0 = derived

    # Optional head--tail estimator of teacher-weighted cosine distortion. The
    # deterministic stratum is evaluated exactly and one teacher-proportional tail
    # draw supplies an unbiased estimate of the remaining cached-target mass. It is
    # additive to the KL stack, not a replacement, so multi-scale distribution
    # matching and ambient calibration remain intact. Off until its coefficient is
    # selected by the matched-budget ablation.
    unbiased_geometry_weight = 0.0

    # ---- Teacher Graph -------------------------------------------------------
    graph_k = 200
    # Bandwidth selection. Each transition row is solved for its own temperature so
    # that H(P(.|i)) = log(perplexity): one effective number of neighbours for every
    # node, instead of one temperature for every node. Two reasons, both provable:
    # the entropy is strictly monotone in the temperature (dH/dbeta = -beta*Var(s)),
    # so the solution is unique and bisection cannot fail; and the row is exactly
    # invariant to an affine rescaling s -> a*s + b of the teacher's similarities,
    # which is what forced graph_temp to be retuned for each teacher. 30 is the
    # t-SNE default for the same quantity (van der Maaten and Hinton, 2008).
    perplexity = 30
    # Sorted, unique, and starting at 1. All three are enforced: the artifact stores
    # its scales sorted, and the temperature ladder is anchored to the r=1 target
    # being the transition row.
    diffusion_scales = (1, 2, 4)
    # omega_r = 1/r and omega_0 = omega_1 are derived centrally from these scales.

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
    truncation_tolerance = 0.01

    # ---- Per-Epoch Candidate Sampling ---------------------------------------
    # None derives the support size from the graph artifact at startup: the
    # smallest k reaching ROW_COVERAGE_TAU mixture-mass coverage at the median
    # anchor (src/ggpkd/policy.py, where the sweep evidence for the target is
    # recorded). An int (CLI --diffusion_quota) still overrides for ablations.
    diffusion_quota = None
    # 40/26 is the best-known arm (75.29 on Qwen3-0.6B -> MiniLMv2-H384, graph
    # v9, seed 42). The hard:random split has not been shown flat yet -- that one
    # ablation is what stands between these two numbers and a single derived
    # candidate budget.
    hard_neg_k = 40
    random_neg_k = 26
    # candidate_size is derived as the sum of the three quotas. Candidates are
    # always resampled per epoch with mass-proportional stochastic tails; the
    # deterministic head of each draw is DETERMINISTIC_TOPM in policy.py, not a
    # config knob.

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
    cache_path = "cache/ggpkd/qwen3_4b_bert_base_teacher_train.pt"
    ggpkd_cache_path = "cache/ggpkd/qwen3_4b_bert_base_graph.pt"
    ggpkd_log_dir = "logs/ggpkd"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "models/ggpkd/qwen3_4b_to_bert_base"
    final_weights_only = False

    # ---- Multi-Layer Spec ----------------------------------------------------
    # Defining both of these switches the distiller to its multi-layer GGPKD
    # branch. They are off, so that branch never runs, and with it every knob that
    # only that branch reads.
    # kd_teacher_layers = [12, 24, 36, 36]
    # kd_student_layers = [4, 8, 12, 12]

    # ---- Removed: earlier auxiliary objectives -------------------------------
    # The active objective is L_rel + row_weight * L_row +
    # unbiased_geometry_weight * E_hat inside the GGPKD criterion.
    # lambda_ggpkd,
    # lambda_cosine, lambda_infonce,
    # lambda_simcse, simcse_temp, simcse_start_epoch and lambda_sim used to sit
    # here at 0. Every one of them is read only inside the multi-layer branch
    # above, so on this path they were unreachable: they printed in the run banner
    # and counted against the method's knob budget while doing nothing. Their
    # consumers all read them through getattr with the same defaults these lines
    # carried, so deleting them changes no behaviour. Re-adding one is only
    # meaningful together with the kd_*_layers pair.

    def __init__(self, **kwargs):
        # An unknown key is a typo, not a no-op. The previous version skipped it
        # silently, so `GGPKDConfig(walk_lenght=8)` ran the default 4 and looked
        # like the override had been applied.
        unknown = sorted(k for k in kwargs if not hasattr(self, k))
        if unknown:
            raise AttributeError(
                f"GGPKDConfig got unknown option(s): {', '.join(unknown)}"
            )
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.row_weight < 0:
            raise ValueError("row_weight must be non-negative")
        if self.direct_temp < 0:
            raise ValueError("direct_temp must be positive, or 0 to derive it")
        if self.unbiased_geometry_weight < 0:
            raise ValueError("unbiased_geometry_weight must be non-negative")
        if self.row_start_epoch < 1:
            raise ValueError("row_start_epoch must be at least 1")
        if self.diffusion_quota is not None and self.diffusion_quota < 0:
            raise ValueError("diffusion_quota must be None (derived) or non-negative")
