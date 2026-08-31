from .base_config import BaseConfig

# Kept next to the config rather than imported from the criterion so that
# `main.py --row_mode` can validate an override without constructing a module.
ROW_MODES = ("walk", "closure_u", "closure_m", "closure_ht")


class HeatGeoConfig(BaseConfig):
    distill_method = "heatgeo"

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
    direct_temp = 0.10  # shared by both sides of the direct scale

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
    # L_row matches each selected node's complete available transition row rather
    # than a sampled next step, so this is dense row-kernel supervision, not a
    # trajectory likelihood. `row_mode` picks how those nodes are selected:
    #
    #   "walk"       non-backtracking teacher walks discover them; nu is the visit
    #                count. Costs num_walks and walk_length, neither of which
    #                appears in the objective -- the trajectory is never a target.
    #   "closure_u"  every teacher-selected pool column (the diffusion support,
    #                not the hard/uniform negatives) is promoted to a row, weighted
    #                uniformly. Batch anchors are excluded: L_rel already matches
    #                their transition row as its r=1 target. No walk knobs.
    #   "closure_m"  the same rows, weighted by the teacher mass the pool exposes
    #                for each, m_B(j) = P^T_j(Omega_j). No walk knobs.
    #   "closure_ht" the same rows, weighted by 1/c_B(j) (inverse inclusion count).
    #                The bias ladder: selection is already mass-proportional, so
    #                closure_m squares the hub bias (measured -0.10), closure_u
    #                keeps it linear, closure_ht approximately cancels it -- the
    #                epoch-level row measure approaches uniform over the corpus.
    #                No walk knobs.
    #
    # closure_u is the control for closure_m: identical row sets, so the pair
    # isolates the weighting. Under a closure mode num_walks is forced to 0 below,
    # which also stops walk-visited nodes from displacing uniform negatives in the
    # candidate draw -- the one way the walk changes L_rel as well as L_row.
    #
    # Measured on Qwen3-0.6B -> MiniLMv2-H384, row_weight 1.0, seed 42:
    # walk 74.88, closure_u 74.86, closure_m 74.76. closure_u holds the walk's
    # result to within noise while costing zero walk hyperparameters, so it is the
    # default. closure_m loses 0.10: a row with high exposed mass sits deep inside
    # some anchor's neighbourhood, which is exactly where its transition row most
    # nearly duplicates the r=1 target L_rel already gives that anchor, so weighting
    # by mass concentrates nu on the redundant rows. `row_eff_count` in the epoch
    # log measures that concentration directly.
    #
    # "walk" is kept because it is the ablation this replaced, not because it is
    # reachable by default.
    row_mode = "closure_u"
    # Experiment flag, to be resolved into always-on or deleted: give every
    # promoted row an ambient r=0 term over the shared pool, mirroring the anchor
    # stack (weight tied omega_amb = omega_1, temperature direct_temp on both
    # sides -- no new free parameter). Motivated by row_exposed_mass = 0.44: the
    # restricted row target renormalizes less than half of each row's true
    # transition mass, in a method whose every other truncation is held to 1%.
    row_ambient = False
    num_walks = 4
    walk_length = 4
    # 1.0 matches the selected arm: the tuning table is monotone in row_weight
    # (0.5 -> 74.63, 1.0 -> 74.88) and was never probed above 1.0 -- values > 1
    # are an open experiment, not a tuned optimum.
    row_weight = 1.0
    # Human-facing, one-based epoch number. At 1 the knob is inert: L_row is on for
    # every epoch, and the curriculum disappears along with the parameter.
    #
    # The warm-up existed because the walk produced stochastic, noisy auxiliary rows
    # that were worth withholding for an epoch. Closure does not: its rows are the
    # candidate columns L_rel already selected, and the epoch-1 diagnostics of the
    # start-at-1 run show the same 843 rows at the same 0.44 exposed mass as every
    # later epoch, with loss_rel ending *lower* than the start-at-2 run (0.6502 vs
    # 0.6686). Measured at row_weight 1.0, seed 42: closure_u 74.86 starting at 2,
    # 74.82 starting at 1 -- inside single-seed noise, against a walk baseline of
    # 74.88.
    #
    # Set to 2 to restore the curriculum; the walk mode is the arm that wants it.
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
    # pool_capped_rows / walk_capped_rows if either ever binds before the tolerance
    # is met, and then the guarantee does not hold.
    truncation_tolerance = 0.01

    # ---- Per-Epoch Candidate Sampling ---------------------------------------
    diffusion_quota = 14
    hard_neg_k = 26
    random_neg_k = 26
    # candidate_size is derived as 14 + 26 + 26 = 66. Candidates are always
    # resampled per epoch with mass-proportional stochastic tails.
    deterministic_topm = 2

    # ---- Corpus Columns ------------------------------------------------------
    # Which column is the graph node, and which defines "same source" for hard
    # negatives. Both were read by the distiller but declared nowhere, so they
    # could not be set through this class at all. None keeps the existing
    # behaviour: the anchor column is picked from task_type.
    heatgeo_anchor_column = None
    heatgeo_source_column = "source"

    # ---- Training Setup ------------------------------------------------------
    batch_size = 64
    epochs = 5
    learning_rate = 2e-5
    min_lr = 3e-6
    num_workers = 4
    # Disable validation during tuning; final pair thresholds are selected
    # directly on each test task by the final evaluation.
    eval_every = 0

    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    # cache_teacher removed: nothing read it. Teacher caching is gated purely by
    # whether cache_path already exists on disk (distiller.py).
    cache_path = "cache/heatgeo/qwen3_4b_bert_base_teacher_train.pt"
    heatgeo_cache_path = "cache/heatgeo/qwen3_4b_bert_base_graph.pt"
    heatgeo_log_dir = "logs/heatgeo"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "models/heatgeo/qwen3_4b_to_bert_base"
    final_weights_only = False
    use_wandb = True
    wandb_project = "iclr-mdd-heatgeo"
    wandb_run_name = "heatgeo_qwen3_4b_to_bert_base"
    wandb_mode = "online"

    # ---- Multi-Layer Spec ----------------------------------------------------
    # Defining both of these switches the distiller to its multi-layer HeatGeo
    # branch. They are off, so that branch never runs, and with it every knob that
    # only that branch reads.
    # kd_teacher_layers = [12, 24, 36, 36]
    # kd_student_layers = [4, 8, 12, 12]

    # ---- Removed: auxiliary objectives ---------------------------------------
    # The objective is L_rel + row_weight * L_row inside the HeatGeo criterion.
    # lambda_heatgeo,
    # lambda_cosine, lambda_infonce,
    # lambda_simcse, simcse_temp, simcse_start_epoch and lambda_sim used to sit
    # here at 0. Every one of them is read only inside the multi-layer branch
    # above, so on this path they were unreachable: they printed in the run banner
    # and counted against the method's knob budget while doing nothing. Their
    # consumers all read them through getattr with the same defaults these lines
    # carried, so deleting them changes no behaviour. Re-adding one is only
    # meaningful together with the kd_*_layers pair.

    def resolve_row_mode(self) -> None:
        """Validate ``row_mode`` and drop the walk knobs the closure modes do not own.

        Called from ``__init__`` and again by ``main.py`` after the CLI overrides,
        which are plain ``setattr`` and so bypass every check here. Forcing
        ``num_walks`` to 0 rather than merely ignoring it keeps the sampler, the run
        banner and the artifact requirement honest: with ``num_walks > 0`` the
        sampler would still draw walks, still demand the transition arrays for that
        draw, and still let visited nodes displace uniform negatives in the candidate
        set -- which changes L_rel too, in a run whose whole point is that only
        L_row's row selection changed.
        """
        if self.row_mode not in ROW_MODES:
            raise ValueError(
                f"row_mode must be one of {ROW_MODES}, got {self.row_mode!r}"
            )
        if self.row_mode != "walk":
            self.num_walks = 0

    def __init__(self, **kwargs):
        # An unknown key is a typo, not a no-op. The previous version skipped it
        # silently, so `HeatGeoConfig(walk_lenght=8)` ran the default 4 and looked
        # like the override had been applied.
        unknown = sorted(k for k in kwargs if not hasattr(self, k))
        if unknown:
            raise AttributeError(
                f"HeatGeoConfig got unknown option(s): {', '.join(unknown)}"
            )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.resolve_row_mode()
        if self.num_walks < 0:
            raise ValueError("num_walks must be non-negative")
        if self.walk_length < 1:
            raise ValueError("walk_length must be positive")
        if self.row_weight < 0:
            raise ValueError("row_weight must be non-negative")
        if self.row_start_epoch < 1:
            raise ValueError("row_start_epoch must be at least 1")
