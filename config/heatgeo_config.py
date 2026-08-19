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
    # One temperature law, enforced by HeatGeoDistillation (it rejects the removed
    # knobs student_temp / broad_scale_temps / direct_temp / direct_weight on top
    # of the provable ties scale_temps / walk_temp / direct_student_temp):
    #   tau_r = graph_temp * r^temp_exponent      (r = 1 reproduces the tie)
    #   tau_0 = graph_temp * r_max^temp_exponent  (direct scale, both sides)
    #   tau_w = graph_temp
    # The direct scale is always on with pre-normalization weight 1: one member
    # of the ladder, not an optional term.
    #
    # temp_exponent is fixed at the heat-kernel value alpha = 1 (tau_r linear in
    # diffusion time; Eq. templaw in the paper) rather than tuned:
    #   1.0 -> temps (0.05, 0.10, 0.20),  tau_0 = 0.20    heat-kernel value
    #   0.5 -> temps (0.05, 0.0707, 0.10), tau_0 = 0.10   historical hand-tuned
    #          ladder, reachable via --temp_exponent for comparison runs
    temp_exponent = 1.0
    #
    # Scale weights follow the same pattern (omega_r = r^-weight_exponent,
    # rejected as a raw sequence by the criterion): the default 0.0 gives equal
    # weight per dyadic octave -- the log-uniform scale measure dt/t used by
    # heat-kernel signatures (HKS, SI-HKS, NetLSD) and unweighted dyadic
    # diffusion wavelets:
    #   0.0 -> (1, 1, 1)        log-uniform, gamma = 1     default
    #   1.0 -> (1, 1/2, 1/4)    historical ladder, reachable via
    #          --weight_exponent for comparison runs
    # The direct scale always carries pre-normalization weight 1 against the
    # normalized diffusion family: a 50/50 split between absolute calibration
    # and neighbourhood ranking, independent of the exponent.
    weight_exponent = 0.0
    share_in_batch = True

    # ---- Teacher Graph -------------------------------------------------------
    graph_k = 200
    graph_temp = 0.05
    diffusion_scales = (1, 2, 4)
    # scale_weights is gone: derived from weight_exponent (see Objective block).
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

    # ---- Training Setup ------------------------------------------------------
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 3e-6
    num_workers = 4
    encode_chunk_size = 256

    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    cache_teacher = True
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
    lambda_heatgeo = 1.0

    # ---- Auxiliary objectives: all off ---------------------------------------
    # The objective is exactly two terms, L = L_diff + walk_weight * L_walk, both
    # inside the HeatGeo criterion. Everything below is a distiller-level term
    # shared with the other methods and is held at 0 for this config.
    lambda_cosine = 0
    lambda_infonce = 0
    lambda_simcse = 0
    simcse_temp = 0
    simcse_start_epoch = 2
    lambda_sim = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
