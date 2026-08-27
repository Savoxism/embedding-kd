from .base_config import BaseConfig


class RKDConfig(BaseConfig):

    distill_method = "rkd"

    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    # Park et al. (2019), Sec. 3.3: L = L_task + lambda_RKD * L_RKD. The paper's
    # metric-learning setting is the one that matches this repo (distilling an
    # embedding, no class labels), and it uses lambda_RKD-D = 1 with
    # lambda_RKD-A = 2 -- the RKD-DA row of Table 4 and the authors' released
    # command line (`--dist_ratio 1 --angle_ratio 2`).
    w_task = 1.0
    dist_ratio = 1.0
    angle_ratio = 2.0

    # delta of the Huber loss in Eq. 5.
    huber_delta = 1.0
    eps_norm = 1e-12

    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6

    save_dir = "checkpoints/rkd"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
