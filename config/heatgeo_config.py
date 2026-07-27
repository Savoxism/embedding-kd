from .base_config import BaseConfig


class HeatGeoConfig(BaseConfig):
    
    distill_method = "heatgeo"
    
    student_model_name = "google-bert/bert-base-uncased"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-4B"
    teacher_dtype = "float32"
    
    student_special_token = "##"
    teacher_special_token = "G"
    
    temperature = 0.1
    student_temp = 0.07
    
    alpha = 1.0
    lambda_spec = 0.0
    lambda_anchor = 0.0
    
    graph_k = 50
    graph_temp = 0.1
    diffusion_scales = (1, 2, 4)
    scale_weights = (1.0, 0.5, 0.25)
    diffusion_topk = 32
    hard_neg_k = 16
    random_neg_k = 16
    candidate_size = 64
    spectral_dim = 16
    use_spectral = True
    
    batch_size = 8
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6
    num_workers = 0
    
    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    cache_teacher = True
    cache_path = "cache/heatgeo/qwen3_4b_bert_base_teacher_train.pt"
    heatgeo_cache_path = "cache/heatgeo/qwen3_4b_bert_base_graph.pt"
    heatgeo_log_dir = "logs/heatgeo"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float32"
    
    save_dir = "models/heatgeo/qwen3_4b_to_bert_base"
    use_wandb = False
    wandb_project = "iclr-mdd-heatgeo"
    wandb_run_name = "heatgeo_qwen3_4b_to_bert_base"
    wandb_mode = "online"

    # SAM Optimizer Settings
    use_sam = True
    rho = 0.05
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
