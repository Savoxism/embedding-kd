from .base_config import BaseConfig


class PointwiseConfig(BaseConfig):
    """Pointwise embedding KD, matched to the GGPKD training setup.

    Everything an arm comparison reads is deliberately copied from
    `GGPKDConfig` rather than taken from this method's own literature defaults:
    batch size, epochs, learning rate, schedule and corpus. The batch study asks
    what changes when batch composition changes, and it can only ask that if the
    two arms it compares were trained under the same budget.
    """

    distill_method = "pointwise"

    student_model_name = "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    # The KD term is the whole objective: no task loss, so the arm has exactly one
    # thing in it and its stability across batch samplers means what it says.
    w_task = 0.0
    eps_norm = 1e-12

    batch_size = 64
    epochs = 5
    learning_rate = 3e-5
    min_lr = 3e-6
    num_workers = 4
    eval_every = 0

    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    cache_path = "cache/ggpkd/qwen3_0_6b_to_minilmv2_h384/teacher_train.pt"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "models/pointwise/qwen3_0_6b_to_minilmv2_h384"
    final_weights_only = True
