from .base_config import BaseConfig

class HeatGeoConfig(BaseConfig):
    distill_method = "heatgeo"

    student_model_name = "google-bert/bert-base-uncased"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-4B"
    teacher_dtype = "float32"

    student_special_token = "##"
    teacher_special_token = "G"

    # ---- Objective -----------------------------------------------------------
    # Temperature ties, enforced by HeatGeoDistillation (it rejects the removed
    # knobs scale_temps / walk_temp / direct_student_temp):
    #   tau_1 = graph_temp   the r=1 target IS the transition row, a softmax of
    #                        teacher cosines at graph_temp -- any other student
    #                        temperature makes zero loss unattainable;
    #   tau_w = graph_temp   walk targets are transition rows, same argument;
    #   direct scale         one temperature (direct_temp) on both teacher and
    #                        student side (Hinton et al. 2015 convention).
    # Only the broader diffusion scales keep free student temperatures.
    student_temp = 0.07               # fallback when broad_scale_temps is unset
    broad_scale_temps = (0.07, 0.10)  # student temps for r = 2, 4; r = 1 uses graph_temp
    share_in_batch = True

    # ---- Direct Scale (r=0) --------------------------------------------------
    direct_weight = 1.0
    direct_temp = 0.10                # shared by both sides of the direct scale

    # ---- Teacher Graph -------------------------------------------------------
    graph_k = 200
    graph_temp = 0.05
    diffusion_scales = (1, 2, 4)
    scale_weights = (1.0, 0.5, 0.25)
    pool_size = 256
    
    # ---- Random Walk Kernel Matching -----------------------------------------
    # L_walk is a KL against the teacher's own transition row, so it shares the
    # scale of L_diff and walk_weight is a genuine trade-off rather than a units
    # conversion. 0 walks disables the term and reproduces the diffusion-only run.
    num_walks = 4
    walk_length = 4
    walk_weight = 0.5
    # walk_temp is gone: it is tied to graph_temp inside the criterion.
    walk_start_epoch = 1
    walk_topk = None
    # Which rows the walk supervises, never what they are matched against: a step may
    # not return to the node it came from (except out of a degree-1 node, where that
    # is the only edge). Same walk budget, more distinct rows, and no new temperature.
    walk_non_backtracking = True
    
    hard_neg_pool = 200
    walk_keep_topk = 2048

    # ---- Per-Epoch Candidate Sampling ---------------------------------------
    candidate_size = 66
    diffusion_quota = 14
    hard_neg_k = 26
    random_neg_k = 26
    resample_candidates_per_epoch = True
    deterministic_topm = 2
    stochastic_candidates = True
    dedup_corpus = True
    diag_topk = 8
    eps_norm = 1e-8

    # ---- Corpus Columns ------------------------------------------------------
    # Which column is the graph node, and which defines "same source" for hard
    # negatives. Both were read by the distiller but declared nowhere, so they
    # could not be set through this class at all. None keeps the existing
    # behaviour: the anchor column is picked from task_type.
    heatgeo_anchor_column = None
    heatgeo_source_column = "source"

    # ---- Training Setup ------------------------------------------------------
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 3e-6
    num_workers = 4
    encode_chunk_size = 256

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
    # The objective is exactly two terms, L = L_diff + walk_weight * L_walk, both
    # inside the HeatGeo criterion. lambda_heatgeo, lambda_cosine, lambda_infonce,
    # lambda_simcse, simcse_temp, simcse_start_epoch and lambda_sim used to sit
    # here at 0. Every one of them is read only inside the multi-layer branch
    # above, so on this path they were unreachable: they printed in the run banner
    # and counted against the method's knob budget while doing nothing. Their
    # consumers all read them through getattr with the same defaults these lines
    # carried, so deleting them changes no behaviour. Re-adding one is only
    # meaningful together with the kd_*_layers pair.

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
