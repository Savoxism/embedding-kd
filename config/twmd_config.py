from .base_config import BaseConfig

class TWMDConfig(BaseConfig):
    
    distill_method = "twmd"
    
    student_model_name = "bert-base-uncased"
    student_dtype = "float32"
    teacher_model_name = "Qwen/Qwen3-Embedding-4B"
    teacher_dtype = "bfloat16"
    
    student_special_token = "##"
    teacher_special_token = "G"
    
    task_type = "pair_cls"
    
    w_task = 0.001
    lambda_rw_path = 1.0
    lambda_diff = 1.0
    lambda_spec = 0.01
    lambda_anchor = 0.01
    
    temperature = 0.1
    student_temp = 0.07
    tau_rw = 0.05
    
    num_walks = 1
    walk_length = 3
    
    eps_norm = 1e-8
    
    graph_k = 50
    graph_temp = 0.1
    diffusion_scales = [1, 2, 4]
    diffusion_topk = 32
    hard_neg_k = 16
    random_neg_k = 16
    candidate_size = 64
    spectral_dim = 16
    use_spectral = True
    scale_weights = [1.0, 0.1, 0.02]
    
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 1e-6
    num_workers = 2
    
    train_data_path = "data/multi-data/train.csv"
    cache_teacher = True
    cache_path = "./cache/teacher_embeddings.pt"
    heatgeo_cache_path = "./cache/twmd_artifact.pt"
    heatgeo_log_dir = "logs/twmd"
    pooling_method = "last_token"
    normalize_cache = True
    cache_dtype = "float16"
    
    save_dir = "./twmd_checkpoints"
    use_wandb = False
    wandb_project = "iclr-mdd-twmd"
    wandb_run_name = "twmd_qwen3_4b_to_bert_base"
    wandb_mode = "online"

    # SAM Optimizer Settings
    use_sam = True
    rho = 0.05
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
