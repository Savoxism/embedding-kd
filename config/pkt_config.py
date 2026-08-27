from .base_config import BaseConfig


class PKTConfig(BaseConfig):

    distill_method = "pkt"

    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"

    # The paper transfers knowledge without any supervised term -- its released
    # training loop defaults to `supervised_weight=0` -- so pure PKT is the
    # faithful setting, and it is also the one that matches RIPPLE's objective
    # form. Pass `--w_task 1.0` for the InfoNCE-plus-KD variant the other
    # baselines in this repo use.
    w_task = 0.0
    w_pkt = 1.0

    # Eq. 6. The paper picks the cosine kernel over the Gaussian of Eq. 5
    # precisely to avoid having to tune a bandwidth.
    kernel = "cosine"
    gaussian_sigma = 1.0

    # Eq. 3, 4 and 8 all exclude the self term; the authors' released code does
    # not. See src/criterions/probabilistic_kt.py for why the difference is not
    # cosmetic. `False` reproduces the released code.
    exclude_self = True

    # "sum" is Eq. 8 verbatim, "mean" is the released code, "batchmean" is the
    # per-anchor KL in nats. They differ by factors of N, so w_pkt is not
    # transferable between them.
    reduction = "batchmean"
    eps_kernel = 1e-7

    # Sec. 3 estimates the conditionals from batches of 64-128 samples. This
    # repo's shared protocol uses 32 for every baseline, and holding the batch
    # fixed is what makes the comparison controlled; a batch-size sweep is the
    # experiment that separates the two.
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6

    save_dir = "checkpoints/pkt"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
