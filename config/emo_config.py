from .base_config import BaseConfig


class EMOConfig(BaseConfig):
    
    distill_method = "emo"
    
    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"
    
    student_special_token = "[CLS]"
    teacher_special_token = "<s>"
    
    w_task = 0.5
    alpha_kd = 0.5
    att_loss_weight = 0.1
    ot_loss_weight = 1.0
    
    k_layers = 1
    alpha_ot = 0.1
    max_iter_ot = 100
    
    batch_size = 32
    epochs = 5
    learning_rate = 5e-5
    min_lr = 2e-6
    warmup_ratio = 0.1
    temperature = 0.05
    
    save_dir = "checkpoints/emo"
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
