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

    # ---- the direct scale (r=0) ---------------------------------------------
    # The diffusion targets are the graph's mass renormalized over the scored
    # columns, so every column outside the anchor's pool gets target exactly 0 --
    # and a 0 target is not "no opinion", it is a gradient pushing that cosine down
    # forever. With in-batch sharing ~97% of an anchor's columns are zero-target,
    # including the same-source hard negatives whose true teacher cosine is high.
    # That is the objective teaching the student to separate pairs the teacher calls
    # similar, and it is why STS moved only +1.4 while emotion F1 fell 2.9.
    #
    # r=0 targets softmax(cos(t_i,t_j)/direct_temp) over the *full* column set, so
    # every column carries its real teacher mass. The teacher embeddings are already
    # cached, so this is free. This is also what replaced the old pointwise anchor
    # term: comparing student cosines to teacher cosines is not invariant to a linear
    # map of the student space, so that term could never calibrate anything.
    direct_weight = 1.0
    # Over a ~1k-column pool the teacher cosine spread is ~0.5, so 0.05 (the graph
    # value) would put essentially all mass on the single nearest column. 0.10 keeps
    # roughly 80% of the mass on genuinely related columns. Watch kl_direct and
    # target_entropy after changing it.
    direct_temp = 0.10
    direct_student_temp = 0.10

    # ---- teacher graph -------------------------------------------------------
    graph_k = 200
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
    # Hard negatives were a liability while their target was a false zero: same-source,
    # top-200 teacher cosine, and the loss drove them apart anyway. With the direct
    # scale they carry their real teacher mass (~27% of it in a controlled probe, vs
    # ~67% for the diffusion positives), which puts them exactly in the discriminative
    # band, so they are now worth more slots than random negatives -- and in-batch
    # sharing already supplies easy negatives from the other anchors for free.
    hard_neg_k = 24
    random_neg_k = 8
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

    # The previous run peaked at 1687 MB of a 69 GB card -- 2.4%. Batch size is what
    # sets the number of shared columns each anchor is scored against (~batch_size x
    # candidate_size after dedup), and cross-anchor calibration is precisely what STS
    # Spearman and a global cosine threshold measure. 16 -> 32 doubles it for free;
    # go higher if the card allows. learning_rate is raised with it (sqrt scaling)
    # because the step count per epoch halves.
    batch_size = 32
    epochs = 5
    learning_rate = 3e-5
    min_lr = 3e-6
    num_workers = 4

    # Candidates are deduplicated and sorted by length, then encoded in chunks of
    # this size so each chunk pads to its own longest member. Padding the whole
    # batch_size x candidate_size block to its global maximum cost ~3.3x more
    # padded tokens on this corpus (median length 13, longest of ~2000 draws ~70).
    # Smaller chunks pad tighter but launch more kernels; 256 is near the knee.
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
    use_wandb = True
    wandb_project = "iclr-mdd-heatgeo"
    wandb_run_name = "heatgeo_qwen3_4b_to_bert_base"
    wandb_mode = "online"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
