from .base_config import BaseConfig


class HeatGeoConfig(BaseConfig):
    distill_method = "heatgeo"

    student_model_name = "google-bert/bert-base-uncased"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-4B"
    teacher_dtype = "float32"

    student_special_token = "##"
    teacher_special_token = "G"

    # ---- objective -----------------------------------------------------------
    # One temperature per diffusion scale. This is what keeps the scales from
    # collapsing: with a single shared tau, sum_r w_r KL(p_r||p^S) is algebraically
    # identical to CE(sum_r w_r p_r, p^S) plus a constant, so every scale set with
    # the same mixture gives the same gradient and "multi-scale" reduces to target
    # smoothing. Distinct tau_r also over-determine the candidate similarities,
    # which forces the student to reproduce the teacher's cosine gaps instead of
    # only its ranking -- the part STS and a global cosine threshold depend on.
    # Broader scale -> smoother target -> larger temperature.
    student_temp = 0.07
    scale_temps = (0.05, 0.07, 0.10)

    # Score every anchor against the union of all candidates in the batch. Same
    # encoding cost, batch_size x more negatives per anchor, and the shared columns
    # couple anchors so similarity levels become comparable across the batch.
    share_in_batch = True

    # Off by default. As written (free linear map W_a + scale-invariant cosine) this
    # term is invariant to any invertible transform of the student space, so it can
    # never supply the absolute calibration it was meant to; that is why it pinned
    # at 0.22-0.27 after one epoch. Kept for the ablation row only.
    lambda_anchor = 0.0

    # ---- teacher graph -------------------------------------------------------
    graph_k = 50
    # graph_temp sets how much ranking information the diffusion targets carry. At 0.1
    # the target was within 0.03 nats of uniform on its support, i.e. a binary
    # neighbour/non-neighbour label. Check target_kl_uniform_r* in the build log after
    # changing the teacher or the corpus; below ~0.05 nats, lower this value.
    graph_temp = 0.05
    diffusion_scales = (1, 2, 4)
    scale_weights = (1.0, 0.5, 0.25)

    # Sparse diffusion support kept per anchor, and the same-source nearest-neighbour
    # pool that hard negatives are drawn from. Both are precomputed once; the
    # candidate set itself is not.
    pool_size = 128
    hard_neg_pool = 200

    # ---- per-epoch candidate sampling ---------------------------------------
    # Freezing one candidate set per anchor meant every epoch after the first
    # re-presented comparisons the student had already fit, which is the direct
    # cause of the eval saturating at epoch 1. Quotas must sum to <= candidate_size.
    candidate_size = 64
    diffusion_quota = 32
    hard_neg_k = 16
    random_neg_k = 16
    resample_candidates_per_epoch = True
    # The diffusion quota is split evenly across scales, not drawn from the mixture:
    # the mixture's support is dominated by the broadest walk, which would leave the
    # sharpest scale with one or two scored candidates and no ranking signal.
    # Within each scale, keep its top-m neighbours and sample the rest of its share
    # proportionally to diffusion mass.
    deterministic_topm = 2
    stochastic_candidates = True

    # Exact duplicate texts have cos = 1 by construction, so their logit is pinned at
    # the ceiling with zero gradient while still absorbing teacher mass.
    dedup_corpus = True

    diag_topk = 8

    batch_size = 16
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6
    num_workers = 4

    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    cache_teacher = True
    cache_path = "cache/heatgeo/qwen3_4b_bert_base_teacher_train.pt"
    heatgeo_cache_path = "cache/heatgeo/qwen3_4b_bert_base_graph.pt"
    heatgeo_log_dir = "logs/heatgeo"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "models/heatgeo/qwen3_4b_to_bert_base"
    use_wandb = True
    wandb_project = "iclr-mdd-heatgeo"
    wandb_run_name = "heatgeo_qwen3_4b_to_bert_base"
    wandb_mode = "online"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
