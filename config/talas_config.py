from .base_config import BaseConfig

DEFAULT_TALAS_PAIR = "qwen3_0_6b_to_minilmv2_h384"

TALAS_PAPER_PAIRS = {
    DEFAULT_TALAS_PAIR: {
        "teacher": "Qwen/Qwen3-Embedding-0.6B",
        "student": "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
        "pooling_method": "last_token",
    },
    "bge_m3_to_minilmv2_h768": {
        "teacher": "BAAI/bge-m3",
        "student": "nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base",
        "pooling_method": "cls",
    },
    "qwen3_4b_to_bert_base": {
        "teacher": "Qwen/Qwen3-Embedding-4B",
        "student": "google-bert/bert-base-uncased",
        "pooling_method": "last_token",
    },
}


def get_talas_paper_pair(pair_key: str) -> dict[str, str]:
    try:
        return dict(TALAS_PAPER_PAIRS[pair_key])
    except KeyError as exc:
        choices = ", ".join(sorted(TALAS_PAPER_PAIRS))
        raise ValueError(
            f"Unknown TALAS paper pair {pair_key!r}; expected one of: {choices}"
        ) from exc


class TALASConfig(BaseConfig):
    distill_method = "talas"

    paper_pair = DEFAULT_TALAS_PAIR
    student_model_name = TALAS_PAPER_PAIRS[paper_pair]["student"]
    teacher_model_name = TALAS_PAPER_PAIRS[paper_pair]["teacher"]
    # Some MiniLM checkpoints are stored in fp16. TALAS uses GradScaler before
    # the ASAM perturbation, which requires fp32 trainable master weights.
    student_dtype = "float32"
    teacher_dtype = "bfloat16"

    student_special_token = "##"
    teacher_special_token = "G"

    last_layer_idx = 2
    start_rkd = 0
    w_task = 0.001
    w_kd = 0.75
    w_struct = 1.0
    eps_norm = 1e-12
    temperature = 0.1
    rho = 0.05

    batch_size = 32
    epochs = 5
    learning_rate = 2e-5
    min_lr = 2e-6

    cache_teacher = True
    cache_path = f"cache/talas/{paper_pair}/teacher_train.pt"
    pooling_method = TALAS_PAPER_PAIRS[paper_pair]["pooling_method"]
    normalize_cache = True
    cache_dtype = "float32"

    save_dir = "checkpoints/talas"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
