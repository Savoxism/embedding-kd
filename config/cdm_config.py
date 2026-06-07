from .base_config import BaseConfig


class CDMConfig(BaseConfig):
    
    distill_method = "cdm"
    
    student_model_name = "bert-base-uncased"
    teacher_model_name = "Qwen/Qwen3-Embedding-0.6B"
    teacher_dtype = "bfloat16"
    
    student_special_token = "##"
    teacher_special_token = "G"
    
    w_task = 0.5
    alpha_dtw = 0.5
    w_cls = 1.0
    
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    
    debug_align = False
    
    save_dir = "checkpoints/cdm"
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
