class BaseConfig:
    
    task_type = "pair_cls"
    max_length = 256
    
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6
    warmup_ratio = 0.06
    
    w_task = 0.5
    alpha_dtw = 0.5
    w_cls = 1.0
    temperature = 0.07
    
    student_model_name = "bert-base-uncased"
    teacher_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_dtype = "float32"
    
    student_special_token = "##"
    teacher_special_token = "_"
    
    train_data_path = "data/merged_3_data_5k_each.csv"
    eval_data_path = None
    num_workers = 2
    
    distill_method = "cdm"
    
    save_dir = "checkpoints"
    save_every = 1
    save_best = True
    
    debug_align = False
    
    seed = 42
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items() 
                if not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
