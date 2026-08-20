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
    # The sharpest diffusion target and walk target use the temperature at which
    # each transition row was built. Only broader diffusion scales are free.
    broad_scale_temps = (0.07, 0.10)
    share_in_batch = True

    # ---- Direct Scale (r=0) --------------------------------------------------
    direct_weight = 1.0
    direct_temp = 0.10

    # ---- Teacher Graph -------------------------------------------------------
    graph_k = 200
    # Solve one row temperature so H(P(.|i)) = log(perplexity). Set to None for
    # the fixed-bandwidth graph_temp baseline.
    perplexity = 30
    graph_temp = 0.05
    diffusion_scales = (1, 2, 4)
    scale_weights = (1.0, 0.5, 0.25)
    
    # ---- Random Walk Kernel Matching -----------------------------------------
    # L_walk is a KL against the teacher's own transition row, so it shares the
    # scale of L_diff and walk_weight is a genuine trade-off rather than a units
    # conversion. 0 walks disables the term and reproduces the diffusion-only run.
    num_walks = 4
    walk_length = 4
    walk_weight = 0.5
    # walk_temp is gone: it is tied to graph_temp inside the criterion.
    walk_start_epoch = 1
    walk_non_backtracking = True
    
    hard_neg_pool = 200
    # Keep the smallest support whose discarded mass is at most this tolerance.
    # This replaces pool_size, walk_keep_topk and walk_topk.
    truncation_tolerance = 0.01

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
    # kd_teacher_layers = [12, 24, 36, 36]
    # kd_student_layers = [4, 8, 12, 12]
    # The default path has exactly L_diff + walk_weight * L_walk. Multi-layer-only
    # auxiliary weights remain optional getattr fallbacks in the distiller.

    def __init__(self, **kwargs):
        unknown = sorted(k for k in kwargs if not hasattr(self, k))
        if unknown:
            raise AttributeError(
                f"HeatGeoConfig got unknown option(s): {', '.join(unknown)}"
            )
        for k, v in kwargs.items():
            setattr(self, k, v)
