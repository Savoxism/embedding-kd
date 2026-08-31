from .base_config import BaseConfig


class RKDConfig(BaseConfig):
    """Paper-default RKD-DA settings adapted to the repository's text encoders."""

    distill_method = "rkd"

    student_model_name = "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    # Park et al. use RKD-DA with lambda_D = 1 and lambda_A = 2 for metric
    # learning, without adding the task-specific triplet loss.
    w_task = 0.0
    rkd_distance_weight = 1.0
    rkd_angle_weight = 2.0
    eps_norm = 1e-12

    # Official metric-learning optimization defaults.
    batch_size = 128
    epochs = 80
    learning_rate = 1e-4
    weight_decay = 1e-5
    rkd_lr_decay_epochs = (40, 60)
    rkd_lr_decay_gamma = 0.1

    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    cache_path = "cache/rkd/qwen3_0_6b_to_minilmv2_h384/teacher_train.pt"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "checkpoints/rkd/qwen3_0_6b_to_minilmv2_h384"
    final_weights_only = False
    use_wandb = False

    def __init__(self, **kwargs):
        unknown = sorted(key for key in kwargs if not hasattr(self, key))
        if unknown:
            raise AttributeError(
                f"RKDConfig got unknown option(s): {', '.join(unknown)}"
            )
        for key, value in kwargs.items():
            setattr(self, key, value)
