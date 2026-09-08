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
    
    train_data_path = "data/train_set/merged_3_data_5k_each.csv"
    num_workers = 2
    
    distill_method = "cdm"
    
    save_dir = "checkpoints"
    weights_dir = None
    save_every = 1
    save_best = True
    
    debug_align = False
    # evaluate_test_each_epoch removed: repo-wide grep found no reader anywhere,
    # in any method.
    eval_every = 1
    
    seed = 42

    def __init__(self, **kwargs):
        """Apply overrides onto the class defaults, then check the invariants.

        An unknown key is a typo, not a no-op. Four subclasses used to carry
        their own copy of this loop guarded by `if hasattr(self, k)`, which
        skipped the typo silently -- `GGPKDConfig(walk_lenght=8)` then ran the
        default and looked like the override had been applied. The check lives
        here so no subclass can forget it.
        """
        unknown = sorted(key for key in kwargs if not hasattr(self, key))
        if unknown:
            raise AttributeError(
                f"{type(self).__name__} got unknown option(s): {', '.join(unknown)}"
            )
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.validate()

    def validate(self):
        """Re-checkable invariants; overridden by configs that have any.

        Defined here so callers -- `main.py` after it applies the CLI overrides --
        can call it unconditionally instead of guarding on `hasattr`.
        """

    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def to_dict(self):
        values = {}
        for cls in reversed(type(self).mro()):
            for key, value in vars(cls).items():
                if key.startswith("_") or callable(value):
                    continue
                values[key] = value
        values.update(
            {
                key: value
                for key, value in self.__dict__.items()
                if not key.startswith("_")
            }
        )
        return values
