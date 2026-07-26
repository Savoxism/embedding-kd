from .base_config import BaseConfig


class TMKDConfig(BaseConfig):
    distill_method = "tmkd"

    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    w_task = 1.0
    lambda_tmkd = 1.0
    tmkd_block_size = 512
    tmkd_mode = "full"
    tmkd_deduplicate_identical_pairs = True
    eps_norm = 1e-8

    batch_size = 4
    epochs = 5
    learning_rate = 1e-5
    min_lr = 1e-6
    warmup_ratio = 0.1
    temperature = 0.05

    save_dir = "checkpoints/tmkd"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
