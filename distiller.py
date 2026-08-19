import json
import os
import random
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler
from transformers import __version__ as transformers_version

try:
    # import wandb
    wandb = None
    WANDB_AVAILABLE = False
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
try:
    from pytorch_optimizer import SAM

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print(
        "Warning: pytorch_optimizer not installed. SAM optimizer unavailable for TALAS."
    )
from src.cache_teacher import cache_teacher_embeddings, load_cached_embeddings
from src.criterions.contextual_dynamic_mapping import ContextualDynamicMapping
from src.criterions.dual_space_kd import DualSpaceKD
from src.criterions.emo_embedding_distillation import EMODistillation
from src.criterions.heatgeo_distillation import (
    HeatGeoDistillation,
    scale_weights_from_law,
)
from src.criterions.stella_distillation import (
    StellaModel,
    stella_stage1_loss,
    stella_stage2_loss,
)
from src.criterions.teacher_anchor_kd import TeacherAnchorKD
from src.data_utils import DualTokenizerCollate, TextPairRaw
from src.data_utils.dataset_cache import (
    DualTokenizerCollateWithTeacher,
    HeatGeoCollate,
    TextPairWithTeacher,
    TextPairWithTeacherAndHeatGeo,
)

# Use evaluation_automodel for AutoModel (not evaluation_model_define which is for Stella)
from src.evaluation.evaluation_automodel import (
    eval_classification_task,
    eval_cls_tasks,
    eval_pair_task,
    eval_pair_tasks,
    eval_sts_task,
    eval_sts_tasks,
    test_cls_tasks,
    test_pair_tasks,
    test_sts_tasks,
)
from src.heatgeo import HeatGeoCandidateSampler, build_or_load_heatgeo_artifact
from src.loss import info_nce, pair_inbatch_similarity_loss
from src.pooling import last_token_pool


def is_finite(x: torch.Tensor) -> bool:
    return torch.is_tensor(x) and torch.isfinite(x).all().item()


def nonfinite_details(name: str, tensor: torch.Tensor) -> str:
    if not torch.is_tensor(tensor):
        return f"{name}: expected tensor, got {type(tensor).__name__}"
    if tensor.is_floating_point() or tensor.is_complex():
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
    else:
        nan_count = 0
        inf_count = 0
    return (
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, nan_count={nan_count}, inf_count={inf_count}"
    )


def assert_module_parameters_finite(module: nn.Module, module_name: str) -> None:
    finite_status = None
    for parameter in module.parameters():
        current = torch.isfinite(parameter).all()
        finite_status = current if finite_status is None else finite_status & current

    if finite_status is None or bool(finite_status.item()):
        return

    for name, parameter in module.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise RuntimeError(
                f"{module_name} parameters became NaN/Inf: "
                f"{nonfinite_details(name, parameter)}"
            )


def grads_are_finite(optim) -> bool:
    # Accumulated on device and read back once. Testing each gradient in a Python
    # `if` forces a host sync per parameter, which is ~200 stalls per step on a
    # BERT-base student -- for a check that is almost always True.
    finite_status = None
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            current = torch.isfinite(p.grad).all()
            finite_status = current if finite_status is None else finite_status & current
    return finite_status is None or bool(finite_status.item())


class KnowledgeDistiller:
    def __init__(self, config):
        self.config = config
        self.wandb_run = None
        self.global_step = 0
        self.current_epoch = 0
        self.current_step = 0
        self._saved_checkpoint_epochs = set()
        self._saved_student_weight_epochs = set()
        self.setup_seed(config.seed)
        self.setup_devices()
        self.setup_models()
        self.setup_data()
        self.setup_training()
        self.setup_wandb()
        self.proj_s2t = None

        # Initialize criterion based on method
        if config.distill_method == "cdm":
            self.criterion = ContextualDynamicMapping(
                tok_student=self.tok_student,
                tok_teacher=self.tok_teacher,
                blending_model_special_token=config.teacher_special_token,
                base_model_special_token=config.student_special_token,
                w_task=config.w_task,
                alpha_dtw=config.alpha_dtw,
                debug_align=config.debug_align,
            )
        elif config.distill_method == "dskd":
            self.criterion = DualSpaceKD(
                student_dim=self.model_student.config.hidden_size,
                teacher_dim=self.model_teacher.config.hidden_size,
                w_task=config.w_task,
                alpha_dtw=config.alpha_dtw,
            )
            # Move DSKD to device and add to optimizer
            self.criterion.to(self.device_s)
            self.optimizer.add_param_group(
                {"params": self.criterion.parameters(), "lr": config.learning_rate}
            )
            self.scheduler = self._build_scheduler()
            print("DSKD criterion initialized and added to optimizer")
        elif config.distill_method == "emo":
            self.criterion = EMODistillation(
                d_teacher=self.model_teacher.config.hidden_size,
                d_student=self.model_student.config.hidden_size,
                k_layers=getattr(config, "k_layers", 1),
                alpha_ot=getattr(config, "alpha_ot", 0.1),
                max_iter=getattr(config, "max_iter_ot", 100),
                teacher_special=getattr(config, "teacher_special_token", "<s>"),
                student_special=getattr(config, "student_special_token", "[CLS]"),
            )
            # Move EMO to device and add to optimizer
            self.criterion.to(self.device_s)
            self.optimizer.add_param_group(
                {"params": self.criterion.parameters(), "lr": config.learning_rate}
            )
            self.scheduler = self._build_scheduler()
            print("EMO criterion initialized and added to optimizer")
        elif config.distill_method == "heatgeo":
            # Multi-layer HeatGeo logic
            if hasattr(config, "kd_teacher_layers") and hasattr(config, "kd_student_layers"):
                self.kd_student_layers = config.kd_student_layers
                self.kd_teacher_layers = config.kd_teacher_layers
                
                # We expect the last layer pair to use Cosine Loss, and the rest to use HeatGeo.
                self.num_heatgeo_layers = len(self.kd_student_layers) - 1
                self.heatgeo_criterions = nn.ModuleList()
                
                print(f"Multi-layer HeatGeo: Using {self.num_heatgeo_layers} middle layers for HeatGeo, 1 final layer for Cosine.")
                
                for l_idx in range(self.num_heatgeo_layers):
                    # For each middle layer, initialize a HeatGeoDistillation
                    criterion = HeatGeoDistillation(
                        student_dim=self.model_student.config.hidden_size,
                        teacher_dim=self.teacher_cls_all.shape[-1],
                        diffusion_scales=getattr(config, "diffusion_scales", (1, 2, 4)),
                        temp_exponent=getattr(config, "temp_exponent", 1.0),
                        weight_exponent=getattr(config, "weight_exponent", 0.0),
                        eps_norm=getattr(config, "eps_norm", 1e-8),
                        diag_topk=getattr(config, "diag_topk", 8),
                        share_in_batch=getattr(config, "share_in_batch", True),
                        teacher_embeddings=self.teacher_cls_all[:, l_idx, :],
                        walk_weight=getattr(config, "walk_weight", 0.5),
                        walk_topk=getattr(config, "walk_topk", None),
                        # The tie tau_1 = tau_w = graph_temp lives in the criterion;
                        # this is the temperature the transition rows were built at.
                        graph_temp=config.graph_temp,
                        transition_neighbors=getattr(self, "heatgeo_artifact", {}).get("transition_neighbors")
                        if getattr(config, "num_walks", 0) > 0
                        else None,
                        transition_probs=getattr(self, "heatgeo_artifact", {}).get("transition_probs")
                        if getattr(config, "num_walks", 0) > 0
                        else None,
                    ).to(self.device_s)
                    self.heatgeo_criterions.append(criterion)
                
                self.cosine_criterion = nn.CosineEmbeddingLoss().to(self.device_s)
                self.criterion = None # Managed separately
                
                # Setup projection layer if needed before building scheduler
                d_s = self.model_student.config.hidden_size
                d_t = self.teacher_cls_all.shape[-1]
                if d_s != d_t:
                    self.proj_s2t = nn.Linear(d_s, d_t, bias=False).to(self.device_s)
                    self.optimizer.add_param_group(
                        {
                            "params": self.proj_s2t.parameters(),
                            "lr": config.learning_rate * 2,
                        }
                    )
                    print(f"Initialized HeatGeo projection layer: {d_s} -> {d_t}")
                    
                self.scheduler = self._build_scheduler()
            else:
                self.criterion = HeatGeoDistillation(
                    student_dim=self.model_student.config.hidden_size,
                    teacher_dim=self.teacher_cls_all.shape[-1],
                    diffusion_scales=getattr(config, "diffusion_scales", (1, 2, 4)),
                    temp_exponent=getattr(config, "temp_exponent", 1.0),
                    weight_exponent=getattr(config, "weight_exponent", 0.0),
                    eps_norm=getattr(config, "eps_norm", 1e-8),
                    diag_topk=getattr(config, "diag_topk", 8),
                    share_in_batch=getattr(config, "share_in_batch", True),
                    teacher_embeddings=self.teacher_cls_all,
                    walk_weight=getattr(config, "walk_weight", 0.5),
                    walk_topk=getattr(config, "walk_topk", None),
                    # The tie tau_1 = tau_w = graph_temp lives in the criterion;
                    # this is the temperature the transition rows were built at.
                    graph_temp=config.graph_temp,
                    transition_neighbors=getattr(self, "heatgeo_artifact", {}).get("transition_neighbors")
                    if getattr(config, "num_walks", 0) > 0
                    else None,
                    transition_probs=getattr(self, "heatgeo_artifact", {}).get("transition_probs")
                    if getattr(config, "num_walks", 0) > 0
                    else None,
                ).to(self.device_s)
                self.scheduler = self._build_scheduler()
                print("HeatGeo criterion initialized (single layer)")
        else:
            self.criterion = None

        # Projection layer was initialized above if needed

        # Metrics tracking
        self.step_times = []
        self.ma_window = deque(maxlen=50)
        self.warmup_steps = 10

    def setup_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Done setup_seed with seed={seed}")

    def setup_devices(self):
        if torch.cuda.device_count() >= 2:
            self.device_s = torch.device("cuda:0")  # student
            self.device_t = torch.device("cuda:1")  # teacher
            print(
                f"Using 2 GPUs: Student on {self.device_s}, Teacher on {self.device_t}"
            )
        elif torch.cuda.is_available():
            self.device_s = self.device_t = torch.device("cuda:0")
            print("[WARN] Only 1 GPU available -> both on cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device_s = self.device_t = torch.device("mps")
            print("Using Apple Silicon MPS device")
        else:
            self.device_s = self.device_t = torch.device("cpu")
            print("[WARN] No GPU -> CPU training")
        print("Done setup_devices")

    def setup_wandb(self):
        cfg = self.config
        self.use_wandb = bool(getattr(cfg, "use_wandb", False))
        if not self.use_wandb:
            return
        if not WANDB_AVAILABLE:
            print(
                "Warning: wandb is not installed. Install requirements in the project venv to enable W&B logging."
            )
            self.use_wandb = False
            return
        mode = os.environ.get("WANDB_MODE", getattr(cfg, "wandb_mode", "online"))
        project = os.environ.get(
            "WANDB_PROJECT", getattr(cfg, "wandb_project", "iclr-mdd")
        )
        run_name = os.environ.get(
            "WANDB_RUN_NAME", getattr(cfg, "wandb_run_name", None)
        )
        self.wandb_run = wandb.init(
            project=project,
            name=run_name,
            mode=mode,
            config=cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg),
        )
        print(f"W&B logging enabled: project={project}, run={run_name}, mode={mode}")
        if cfg.distill_method == "heatgeo" and hasattr(self, "heatgeo_artifact"):
            graph_stats = self.heatgeo_artifact.get("graph_stats", {})
            if graph_stats:
                wandb.log(
                    {
                        f"heatgeo_graph/{k}": v
                        for k, v in graph_stats.items()
                        if isinstance(v, (int, float))
                    }
                )
            graph_log_path = self.heatgeo_artifact.get("graph_log_path")
            if graph_log_path and os.path.exists(graph_log_path):
                artifact = wandb.Artifact("heatgeo-knn-graph-log", type="dataset")
                artifact.add_file(graph_log_path)
                self.wandb_run.log_artifact(artifact)

    @staticmethod
    def _flatten_metrics(prefix: str, values: dict[str, Any]) -> dict[str, float]:
        flat = {}
        for key, value in values.items():
            name = f"{prefix}/{key}"
            if isinstance(value, (int, float)):
                flat[name] = float(value)
            elif isinstance(value, dict):
                flat.update(KnowledgeDistiller._flatten_metrics(name, value))
        return flat

    def setup_models(self):
        cfg = self.config

        print("Loading tokenizers...")
        tokenizer_kwargs = {"use_fast": True}
        self.tok_student = AutoTokenizer.from_pretrained(
            cfg.student_model_name,
            **tokenizer_kwargs,
        )
        self.tok_teacher = AutoTokenizer.from_pretrained(
            cfg.teacher_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        if cfg.distill_method == "stella":
            print(f"Loading Stella student model: {cfg.student_model_name}")
            self.model_student = StellaModel(
                cfg.student_model_name,
                output_dim1=getattr(cfg, "output_dim1", 1024),
                pooling=getattr(cfg, "pooling", "cls"),
                output_dim2=getattr(cfg, "output_dim2", 512),
                output_dim3=getattr(cfg, "output_dim3", 256),
                output_dim4=getattr(cfg, "output_dim4", 128),
            )
            self.current_stage = 1
        else:
            print(f"Loading student model: {cfg.student_model_name}")
            student_kwargs = {}
            student_dtype_name = getattr(cfg, "student_dtype", None)
            student_dtypes = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            if student_dtype_name is not None:
                if student_dtype_name not in student_dtypes:
                    raise ValueError(
                        f"Unsupported student_dtype={student_dtype_name!r}; "
                        f"expected one of {sorted(student_dtypes)}"
                    )
                try:
                    transformers_major = int(
                        transformers_version.split(".", maxsplit=1)[0]
                    )
                except (TypeError, ValueError):
                    transformers_major = 4
                dtype_argument = "dtype" if transformers_major >= 5 else "torch_dtype"
                student_kwargs[dtype_argument] = student_dtypes[student_dtype_name]
            self.model_student = AutoModel.from_pretrained(
                cfg.student_model_name,
                **student_kwargs,
            )

        print(f"Loading teacher model: {cfg.teacher_model_name}")
        teacher_kwargs = {"trust_remote_code": True}
        if cfg.teacher_dtype == "bfloat16":
            teacher_kwargs["torch_dtype"] = torch.bfloat16
        elif cfg.teacher_dtype == "float16":
            teacher_kwargs["torch_dtype"] = torch.float16

        # EMO method needs attentions, force eager attention implementation
        if cfg.distill_method == "emo":
            teacher_kwargs["attn_implementation"] = "eager"
            print(
                "Using eager attention implementation for EMO (required for output_attentions)"
            )

        self.model_teacher = AutoModel.from_pretrained(
            cfg.teacher_model_name, **teacher_kwargs
        )

        self.model_student.to(self.device_s)
        self.model_teacher.to(self.device_t)

        student_dtype = next(self.model_student.parameters()).dtype
        print(f"Student training dtype: {student_dtype}")
        assert_module_parameters_finite(self.model_student, "Student model after load")

        self.model_teacher.eval()
        for p in self.model_teacher.parameters():
            p.requires_grad_(False)

        print("Models loaded successfully!")
        print("Done setup_models")

    def _resolve_heatgeo_anchor_column(self, df: pd.DataFrame) -> str:
        cfg = self.config
        column = getattr(cfg, "heatgeo_anchor_column", None)
        if column is not None:
            if column not in df.columns:
                raise ValueError(
                    f"heatgeo_anchor_column={column!r} is not a column of "
                    f"{cfg.train_data_path} (have {list(df.columns)})"
                )
            return column

        if cfg.task_type == "single_cls":
            column = "text"
        elif cfg.task_type == "pair_cls":
            column = "premise"
        else:
            column = "sentence1"
        if column not in df.columns:
            raise ValueError(
                f"HeatGeo needs column {column!r} for task_type={cfg.task_type!r}"
            )

        # The teacher graph is built over this column only. If a genuine second view
        # exists it is dropped, and doing that silently would leave the graph
        # describing a different object than the loss thinks it does.
        partner = {"pair_cls": "hypothesis", "pair_reg": "sentence2"}.get(cfg.task_type)
        if partner in df.columns and not df[column].equals(df[partner]):
            print(
                f"WARNING: HeatGeo uses only {column!r}; {partner!r} differs from it "
                f"and is not distilled. Set heatgeo_anchor_column explicitly if that "
                f"is not what you want."
            )
        return column

    def _prepare_heatgeo_frame(
        self, df: pd.DataFrame, anchor_column: str
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Drop exact duplicate anchors and report the surviving row positions.

        Two identical texts have cos(s_i, s_j) = 1 for every parameter setting, so
        their logit sits at the ceiling with no gradient while still consuming
        teacher mass and a candidate slot.
        """
        keep_positions = np.arange(len(df), dtype=np.int64)
        if not getattr(self.config, "dedup_corpus", True):
            return df.reset_index(drop=True), keep_positions

        texts = df[anchor_column].astype(str)
        duplicated = texts.duplicated(keep="first").to_numpy()
        if not duplicated.any():
            print(f"HeatGeo corpus: {len(df)} rows, no duplicate anchors")
            return df.reset_index(drop=True), keep_positions

        keep_positions = np.flatnonzero(~duplicated).astype(np.int64)
        deduped = df.iloc[keep_positions].reset_index(drop=True)
        print(
            f"HeatGeo corpus dedup on {anchor_column!r}: "
            f"{len(df)} -> {len(deduped)} rows ({int(duplicated.sum())} exact duplicates removed)"
        )
        return deduped, keep_positions

    def _heatgeo_source_ids(self, df: pd.DataFrame) -> np.ndarray:
        column = getattr(self.config, "heatgeo_source_column", "source")
        if column not in df.columns:
            print(
                f"HeatGeo: no {column!r} column, hard negatives will not be "
                f"restricted to the same source corpus"
            )
            return np.zeros(len(df), dtype=np.int64)
        codes = pd.factorize(df[column].astype(str))[0].astype(np.int64)
        counts = pd.Series(codes).value_counts().to_dict()
        print(f"HeatGeo sources: {len(counts)} distinct, sizes={sorted(counts.values(), reverse=True)}")
        return codes

    def setup_data(self):
        cfg = self.config

        print(f"Loading training data from: {cfg.train_data_path}")

        df = pd.read_csv(cfg.train_data_path)

        if cfg.task_type == "pair_cls":
            if "premise" not in df.columns or "hypothesis" not in df.columns:
                # Create from text column
                df["premise"] = df["text"] if "text" in df.columns else df.iloc[:, 0]
                df["hypothesis"] = df["text"] if "text" in df.columns else df.iloc[:, 0]

        # HeatGeo is anchor-only: the teacher graph, the candidate pool and the
        # student forward all consume one string per row. Resolve that column once
        # and keep it, instead of each component re-deriving it from the frame.
        self.heatgeo_anchor_column = None
        if cfg.distill_method == "heatgeo":
            self.heatgeo_anchor_column = self._resolve_heatgeo_anchor_column(df)
            df, keep_positions = self._prepare_heatgeo_frame(
                df, self.heatgeo_anchor_column
            )
            self.heatgeo_keep_positions = keep_positions

        self.task_head = None
        if cfg.distill_method == "emo":
            hidden_size = self.model_student.config.hidden_size
            if cfg.task_type == "single_cls" and "label" in df.columns:
                num_labels = int(df["label"].nunique())
                self.task_head = nn.Linear(hidden_size, num_labels).to(self.device_s)
            elif cfg.task_type == "pair_cls" and "label" in df.columns:
                num_labels = int(df["label"].nunique())
                self.task_head = nn.Linear(hidden_size * 4, num_labels).to(
                    self.device_s
                )

        # TALAS and HeatGeo use cached teacher embeddings
        if cfg.distill_method in ("talas", "heatgeo"):
            cache_path = Path(cfg.cache_path)

            # Check if cache exists
            if cache_path.exists():
                print(f"Loading cached teacher embeddings from: {cache_path}")
                teacher_cls_list = load_cached_embeddings(str(cache_path))
                print(f"Loaded {len(teacher_cls_list)} cached embeddings")
            else:
                print("Cache not found. Pre-computing teacher embeddings...")
                os.makedirs(cache_path.parent, exist_ok=True)

                # Create temporary dataset for caching
                temp_ds = TextPairRaw(df, cfg.task_type)
                temp_collate = DualTokenizerCollate(
                    self.tok_student, self.tok_teacher, cfg.task_type, cfg.max_length
                )
                cache_loader = DataLoader(
                    temp_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,  # Don't shuffle for caching
                    collate_fn=temp_collate,
                    pin_memory=True,
                    num_workers=cfg.num_workers,
                    persistent_workers=cfg.num_workers > 0,
                )

                # Cache teacher embeddings
                teacher_cls_list = cache_teacher_embeddings(
                    model_teacher=self.model_teacher,
                    dataloader=cache_loader,
                    device=self.device_t,
                    pooling_method=cfg.pooling_method,
                    normalize=cfg.normalize_cache,
                    dtype=torch.float32
                    if cfg.cache_dtype == "float32"
                    else torch.float16,
                    cache_path=str(cache_path),
                    kd_teacher_layers=getattr(cfg, "kd_teacher_layers", None),
                )
                print(
                    f"Cached {len(teacher_cls_list)} teacher embeddings to {cache_path}"
                )

            # A stale cache computed before dedup still lines up row-for-row with the
            # original frame, so slice it instead of forcing a teacher re-run.
            keep_positions = getattr(self, "heatgeo_keep_positions", None)
            if (
                cfg.distill_method == "heatgeo"
                and keep_positions is not None
                and len(teacher_cls_list) > len(df)
                and len(keep_positions) == len(df)
                and int(keep_positions.max(initial=-1)) < len(teacher_cls_list)
            ):
                print(
                    f"Slicing pre-dedup teacher cache: {len(teacher_cls_list)} -> {len(df)} rows"
                )
                teacher_cls_list = teacher_cls_list[
                    torch.from_numpy(keep_positions).long()
                ]

            if len(teacher_cls_list) != len(df):
                raise ValueError(
                    f"Cached teacher embeddings length mismatch: cache has {len(teacher_cls_list)} "
                    f"rows but training data has {len(df)} rows. Remove or regenerate {cache_path}."
                )

            self.teacher_cls_all = teacher_cls_list

            if cfg.distill_method == "heatgeo":
                # If teacher_cls_list is 3D [N, Num_Layers, Dim], use the final layer for the graph
                teacher_embeddings_for_graph = teacher_cls_list[:, -1, :] if teacher_cls_list.ndim == 3 else teacher_cls_list

                self.heatgeo_artifact = build_or_load_heatgeo_artifact(
                    teacher_embeddings=teacher_embeddings_for_graph,
                    cache_path=cfg.heatgeo_cache_path,
                    log_dir=getattr(cfg, "heatgeo_log_dir", "logs/heatgeo"),
                    graph_k=getattr(cfg, "graph_k", 50),
                    # No fallback default: the criterion ties tau_1 and tau_w to this
                    # value, so the artifact and the loss must read the same source.
                    graph_temp=cfg.graph_temp,
                    diffusion_scales=getattr(cfg, "diffusion_scales", (1, 2, 4)),
                    scale_weights=scale_weights_from_law(
                        getattr(cfg, "diffusion_scales", (1, 2, 4)),
                        getattr(cfg, "weight_exponent", 0.0),
                    ),
                    pool_size=getattr(cfg, "pool_size", 128),
                    hard_neg_pool=getattr(cfg, "hard_neg_pool", 200),
                    source_ids=self._heatgeo_source_ids(df),
                    walk_keep_topk=getattr(cfg, "walk_keep_topk", None),
                )

            # Free teacher model to save GPU memory (teacher not needed after caching)
            del self.model_teacher
            self.model_teacher = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Teacher model freed from GPU memory")

            if cfg.distill_method == "heatgeo":
                self.heatgeo_sampler = HeatGeoCandidateSampler(
                    artifact=self.heatgeo_artifact,
                    candidate_size=getattr(cfg, "candidate_size", 32),
                    diffusion_quota=getattr(cfg, "diffusion_quota", 20),
                    hard_neg_k=getattr(cfg, "hard_neg_k", 8),
                    random_neg_k=getattr(cfg, "random_neg_k", 4),
                    scale_weights=scale_weights_from_law(
                        getattr(cfg, "diffusion_scales", (1, 2, 4)),
                        getattr(cfg, "weight_exponent", 0.0),
                    ),
                    seed=cfg.seed,
                    deterministic_topm=getattr(cfg, "deterministic_topm", 4),
                    stochastic=getattr(cfg, "stochastic_candidates", True),
                    num_walks=getattr(cfg, "num_walks", 0),
                    walk_length=getattr(cfg, "walk_length", 4),
                    walk_topk=getattr(cfg, "walk_topk", None),
                )
                anchor_texts = df[self.heatgeo_anchor_column].astype(str).tolist()
                self.train_ds = TextPairWithTeacherAndHeatGeo(
                    anchor_texts=anchor_texts,
                    teacher_cls=teacher_cls_list,
                    sampler=self.heatgeo_sampler,
                    labels=df["label"].astype(int).tolist()
                    if "label" in df.columns
                    else None,
                )
                # The collate owns the tokenized corpus: anchors and candidates are
                # drawn from the same rows, so every text is tokenized once here
                # instead of ~candidate_size times per epoch in the workers.
                self.collate_fn = HeatGeoCollate(
                    self.tok_student,
                    cfg.task_type,
                    cfg.max_length,
                    corpus_texts=anchor_texts,
                    encode_chunk_size=getattr(cfg, "encode_chunk_size", 256),
                )
                print(
                    "HeatGeo candidate sampling: "
                    f"candidate_size={self.heatgeo_sampler.candidate_size} "
                    f"(diffusion={self.heatgeo_sampler.diffusion_quota}, "
                    f"hard={self.heatgeo_sampler.hard_neg_k}, "
                    f"random={self.heatgeo_sampler.random_neg_k}), "
                    f"stochastic={self.heatgeo_sampler.stochastic}, "
                    f"resample_per_epoch="
                    f"{getattr(cfg, 'resample_candidates_per_epoch', True)}"
                )
            else:
                self.train_ds = TextPairWithTeacher(df, cfg.task_type, teacher_cls_list)
                self.collate_fn = DualTokenizerCollateWithTeacher(
                    self.tok_student, cfg.task_type, cfg.max_length
                )
        else:
            # Standard distillation methods
            self.train_ds = TextPairRaw(df, cfg.task_type)

            self.collate_fn = DualTokenizerCollate(
                self.tok_student,
                self.tok_teacher,
                cfg.task_type,
                cfg.max_length,
            )

        # Workers fork a copy of the dataset when the iterator is created. Persistent
        # workers would keep serving the epoch-0 sampler state forever, so a dataset
        # whose candidates depend on the epoch must re-fork each epoch.
        resamples_per_epoch = hasattr(self.train_ds, "set_epoch")
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0 and not resamples_per_epoch,
            # 13553 rows at batch 16 leaves a remainder of one. Under in-batch sharing
            # that step scores its single anchor against candidate_size columns instead
            # of ~batch_size * candidate_size, which is a structurally different
            # objective taking a real gradient step -- and it is also the step whose
            # diagnostics land in the progress bar.
            drop_last=cfg.distill_method == "heatgeo",
        )

        print(f"Training samples: {len(self.train_ds)}")
        print(f"Training batches: {len(self.train_loader)}")
        print("Done setup_data")

    def _build_scheduler(self):
        cfg = self.config
        total_steps = len(self.train_loader) * cfg.epochs
        min_lr_rate = cfg.min_lr / cfg.learning_rate
        return get_scheduler(
            name="cosine_with_min_lr",
            optimizer=self.optimizer,
            num_warmup_steps=int(total_steps * cfg.warmup_ratio),
            num_training_steps=total_steps,
            scheduler_specific_kwargs={"min_lr_rate": min_lr_rate},
        )

    def setup_training(self):
        cfg = self.config

        # TALAS optimizer/scheduler will be initialized after criterion creation in train_step
        if cfg.distill_method == "talas":
            self.optimizer = None
            self.scheduler = None
            self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
            print(
                "TALAS: Deferring optimizer/scheduler initialization until criterion is created"
            )
        else:
            optimizer_parameters = list(self.model_student.parameters())
            if self.task_head is not None:
                optimizer_parameters.extend(self.task_head.parameters())
            self.optimizer = optim.AdamW(optimizer_parameters, lr=cfg.learning_rate)

            self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

            self.scheduler = self._build_scheduler()

        if cfg.save_dir:
            os.makedirs(cfg.save_dir, exist_ok=True)
            print(f"Checkpoints will be saved to: {cfg.save_dir}")
        print("Done setup_training")

    def sync_all(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

    def _compute_task_loss(
        self,
        student_cls1: torch.Tensor,
        student_cls2: torch.Tensor | None,
        batch_s: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        labels = batch_s.get("labels")
        if cfg.task_type == "single_cls":
            if labels is None or self.task_head is None:
                raise ValueError("single_cls training requires labels and a task head")
            logits = self.task_head(student_cls1)
            loss = F.cross_entropy(logits, labels.long())
            return loss, {
                "task_accuracy": float(
                    (logits.argmax(-1) == labels).float().mean().item()
                )
            }

        if student_cls2 is None:
            raise ValueError(f"{cfg.task_type} training requires a second text")

        if (
            cfg.task_type == "pair_cls"
            and labels is not None
            and self.task_head is not None
        ):
            pair_features = torch.cat(
                [
                    student_cls1,
                    student_cls2,
                    torch.abs(student_cls1 - student_cls2),
                    student_cls1 * student_cls2,
                ],
                dim=-1,
            )
            logits = self.task_head(pair_features)
            loss = F.cross_entropy(logits, labels.long())
            return loss, {
                "task_accuracy": float(
                    (logits.argmax(-1) == labels).float().mean().item()
                )
            }

        if cfg.task_type == "pair_reg" and labels is not None:
            cosine = F.cosine_similarity(student_cls1, student_cls2)
            predictions = (cosine + 1.0) * 2.5
            loss = F.mse_loss(predictions, labels.float())
            return loss, {"task_mse": float(loss.detach().item())}

        loss, _ = info_nce(student_cls1, student_cls2, temperature=cfg.temperature)
        return loss, {}

    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        method = cfg.distill_method

        if method == "heatgeo":
            batch_s = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                if k.endswith("_stu") or k in {
                    "labels",
                    "idx",
                    "candidate_idx",
                    "candidate_inverse",
                    "teacher_probs",
                    "walk_paths",
                }:
                    batch_s[k] = v.to(self.device_s, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    return_dict=True,
                    output_hidden_states=True if hasattr(self, "kd_student_layers") else False,
                )
                
                # Check curriculum for SimCSE
                run_simcse = (getattr(cfg, "lambda_simcse", 0) > 0) and (getattr(self, "current_epoch", 0) >= getattr(cfg, "simcse_start_epoch", 2))
                
                if run_simcse:
                    s_out2 = self.model_student(
                        input_ids=batch_s["input_ids1_stu"],
                        attention_mask=batch_s["attention_mask1_stu"],
                        return_dict=True,
                        output_hidden_states=True if hasattr(self, "kd_student_layers") else False,
                    )

                # Candidates arrive deduplicated and grouped by length, so each chunk
                # pads to its own longest member. `candidate_inverse` expands the
                # encoded rows back to the flat [batch_size * candidate_size] layout
                # the criterion expects; the gather is differentiable, so a candidate
                # shared by several anchors accumulates all of their gradient.
                
                if hasattr(self, "kd_student_layers"):
                    chunk_embeddings_list = [[] for _ in range(self.num_heatgeo_layers)]
                else:
                    chunk_embeddings_single = []
                    
                for chunk in batch["candidate_chunks"]:
                    chunk_out = self.model_student(
                        input_ids=chunk["input_ids"].to(
                            self.device_s, non_blocking=True
                        ),
                        attention_mask=chunk["attention_mask"].to(
                            self.device_s, non_blocking=True
                        ),
                        return_dict=True,
                        output_hidden_states=True if hasattr(self, "kd_student_layers") else False,
                    )
                    
                    if hasattr(self, "kd_student_layers"):
                        for l_idx in range(self.num_heatgeo_layers):
                            s_layer = self.kd_student_layers[l_idx]
                            idx = s_layer if s_layer < len(chunk_out.hidden_states) else len(chunk_out.hidden_states) - 1
                            idx = idx if idx >= -len(chunk_out.hidden_states) else 0
                            chunk_embeddings_list[l_idx].append(chunk_out.hidden_states[idx][:, 0, :])
                    else:
                        chunk_embeddings_single.append(chunk_out.last_hidden_state[:, 0, :])

                if hasattr(self, "kd_student_layers"):
                    total_loss = 0.0
                    for l_idx, criterion in enumerate(self.heatgeo_criterions):
                        s_layer = self.kd_student_layers[l_idx]
                        idx = s_layer if s_layer < len(s_out1.hidden_states) else len(s_out1.hidden_states) - 1
                        idx = idx if idx >= -len(s_out1.hidden_states) else 0
                        S_cls1 = s_out1.hidden_states[idx][:, 0, :]
                        S_candidates = torch.cat(chunk_embeddings_list[l_idx], dim=0).index_select(
                            0, batch_s["candidate_inverse"]
                        )
                        
                        heat_loss, metrics = criterion(
                            anchor_embeddings=S_cls1,
                            candidate_embeddings=S_candidates,
                            teacher_probs=batch_s["teacher_probs"],
                            candidate_idx=batch_s.get("candidate_idx"),
                            anchor_idx=batch_s.get("idx"),
                            walk_paths=batch_s.get("walk_paths"),
                        )
                        total_loss += heat_loss * cfg.lambda_heatgeo
                        
                    # Cosine loss for final layer
                    s_layer_final = self.kd_student_layers[-1]
                    # Get teacher embeddings from the cached tensor using the batch indices (must be on CPU)
                    T_cls1_final = self.teacher_cls_all[batch["idx"], -1, :].to(self.device_s, non_blocking=True)
                    idx = s_layer_final if s_layer_final < len(s_out1.hidden_states) else len(s_out1.hidden_states) - 1
                    idx = idx if idx >= -len(s_out1.hidden_states) else 0
                    S_cls1_final = s_out1.hidden_states[idx][:, 0, :]
                    
                    if getattr(cfg, "lambda_cosine", 0) > 0:
                        if self.proj_s2t is not None:
                            S_cls1_final_proj = self.proj_s2t(S_cls1_final)
                        else:
                            S_cls1_final_proj = S_cls1_final
                        target = torch.ones(S_cls1_final_proj.size(0), device=self.device_s)
                        cosine_loss = self.cosine_criterion(S_cls1_final_proj, T_cls1_final, target)
                        total_loss += cosine_loss * cfg.lambda_cosine
                        metrics["loss_cosine"] = cosine_loss.item()
                        
                    if run_simcse:
                        idx = s_layer_final if s_layer_final < len(s_out2.hidden_states) else len(s_out2.hidden_states) - 1
                        idx = idx if idx >= -len(s_out2.hidden_states) else 0
                        S_cls2_final = s_out2.hidden_states[idx][:, 0, :]
                        
                        if self.proj_s2t is not None:
                            S_cls2_final_proj = self.proj_s2t(S_cls2_final)
                            # SimCSE needs S_cls1_final_proj as well
                            S_cls1_final_proj = self.proj_s2t(S_cls1_final)
                        else:
                            S_cls2_final_proj = S_cls2_final
                            S_cls1_final_proj = S_cls1_final
                            
                        # SimCSE InfoNCE
                        S1 = F.normalize(S_cls1_final_proj, p=2, dim=-1)
                        S2 = F.normalize(S_cls2_final_proj, p=2, dim=-1)
                        
                        sim_matrix = torch.matmul(S1, S2.t()) / getattr(cfg, "simcse_temp", 0.05)
                        labels = torch.arange(S1.size(0), device=S1.device)
                        simcse_loss = F.cross_entropy(sim_matrix, labels)
                        
                        total_loss += simcse_loss * cfg.lambda_simcse
                        metrics["loss_simcse"] = simcse_loss.item()
                        
                    if getattr(cfg, "lambda_infonce", 0) > 0:
                        if self.proj_s2t is not None:
                            S_cls1_final_proj = self.proj_s2t(S_cls1_final)
                        else:
                            S_cls1_final_proj = S_cls1_final
                        infonce_loss, _ = info_nce(S_cls1_final_proj, T_cls1_final, temperature=getattr(cfg, "student_temp", 0.07))
                        total_loss += infonce_loss * cfg.lambda_infonce
                        metrics["loss_infonce"] = infonce_loss.item()
                        
                    if getattr(cfg, "lambda_sim", 0) > 0:
                        # Similarity loss doesn't need projection since it compares inner (BxB) similarity matrices!
                        sim_loss = pair_inbatch_similarity_loss(S_cls1_final, T_cls1_final)
                        total_loss += sim_loss * cfg.lambda_sim
                        metrics["loss_sim"] = sim_loss.item()
                    
                    loss = total_loss.float()
                else:
                    S_cls1 = s_out1.last_hidden_state[:, 0, :]
                    S_candidates = torch.cat(chunk_embeddings_single, dim=0).index_select(
                        0, batch_s["candidate_inverse"]
                    )

                    loss, metrics = self.criterion(
                        anchor_embeddings=S_cls1,
                        candidate_embeddings=S_candidates,
                        teacher_probs=batch_s["teacher_probs"],
                        candidate_idx=batch_s.get("candidate_idx"),
                        anchor_idx=batch_s.get("idx"),
                        walk_paths=batch_s.get("walk_paths"),
                    )
                    loss = loss.float()

            if not is_finite(loss):
                raise RuntimeError(
                    f"HeatGeo loss NaN/Inf at epoch={self.current_epoch} step={self.current_step}"
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if not grads_are_finite(self.optimizer):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                # Advance the schedule even on a skipped update: it was built for
                # len(train_loader) * epochs steps, so returning early here leaves
                # the LR permanently behind the cosine curve it was sized for.
                self.scheduler.step()
                return loss, {**metrics, "skip": "grad_inf"}

            self.scaler.step(self.optimizer)
            self.scaler.update()
            assert_module_parameters_finite(
                self.model_student,
                f"HeatGeo student after optimizer step "
                f"(epoch={self.current_epoch}, step={self.current_step})",
            )
            self.scheduler.step()

            return loss, metrics

        if method == "talas":
            batch_s = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                if k.endswith("_stu") or k == "labels" or k == "teacher_cls":
                    batch_s[k] = v.to(self.device_s, non_blocking=True)

            # ========== FIRST PASS ==========
            with autocast("cuda", enabled=torch.cuda.is_available()):
                teacher_cls = batch_s["teacher_cls"]

                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    output_hidden_states=False,
                    return_dict=True,
                )

                S_last1 = s_out1.last_hidden_state
                S_last2 = s_out2.last_hidden_state
                S_cls1 = S_last1[:, 0, :]
                S_cls2 = S_last2[:, 0, :]

                loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)

                # Initialize TALAS criterion if needed
                if self.criterion is None:
                    d_s = self.model_student.config.hidden_size
                    d_t = teacher_cls.shape[-1]

                    # BERT-base has 13 layers: embedding + 12 transformer layers
                    num_layers = len(s_out1.hidden_states)

                    self.criterion = TeacherAnchorKD(
                        student_dim=d_s,
                        teacher_dim=d_t,
                        num_layers=num_layers,
                        last_layer_idx=cfg.last_layer_idx,
                        start_rkd=cfg.start_rkd,
                        w_task=cfg.w_task,
                        w_kd=cfg.w_kd,
                        w_struct=cfg.w_struct,
                        eps_norm=cfg.eps_norm,
                    ).to(self.device_s)

                    # Initialize SAM optimizer with both student and criterion parameters
                    if not SAM_AVAILABLE:
                        raise RuntimeError(
                            "SAM optimizer not available. Install pytorch_optimizer."
                        )

                    base_optimizer = optim.AdamW
                    self.optimizer = SAM(
                        [
                            {
                                "params": self.model_student.parameters(),
                                "lr": cfg.learning_rate,
                                "weight_decay": 0.01,
                            },
                            {
                                "params": self.criterion.parameters(),
                                "lr": cfg.learning_rate * 5,
                            },
                        ],
                        base_optimizer,
                        rho=getattr(cfg, "rho", 0.05),
                        adaptive=True,
                    )

                    # Initialize scheduler
                    num_steps = len(self.train_loader)
                    total_steps = num_steps * cfg.epochs
                    min_lr_rate = cfg.min_lr / cfg.learning_rate
                    self.scheduler = get_scheduler(
                        name="cosine_with_min_lr",
                        optimizer=self.optimizer,
                        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
                        num_training_steps=total_steps,
                        scheduler_specific_kwargs={"min_lr_rate": min_lr_rate},
                    )

                    print(
                        f"Initialized TeacherAnchorKD: {d_s} -> {d_t}, num_layers={num_layers}, last_layer_idx={cfg.last_layer_idx}, start_rkd={cfg.start_rkd}"
                    )
                    print(
                        f"Initialized SAM optimizer with rho={getattr(cfg, 'rho', 0.05)}"
                    )
                    print(
                        f"Initialized scheduler: {total_steps} steps, warmup={int(total_steps * cfg.warmup_ratio)}"
                    )

                # Now safe to call criterion with initialized projection heads
                student_outputs = {
                    "hidden_states": s_out1.hidden_states,
                    "last_hidden_state": S_last1,
                }

                loss, metrics = self.criterion(
                    student_outputs=student_outputs,
                    teacher_cls=teacher_cls,
                    task_loss=loss_task,
                )

                loss = loss.float()

            # Backward pass 1 (this will init gradients for first_step)
            self.scaler.scale(loss).backward()

            # Check gradients
            self.scaler.unscale_(self.optimizer)
            if not grads_are_finite(self.optimizer):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                return loss, {**metrics, "skip": "grad_inf_p1"}

            # SAM first step
            self.optimizer.first_step(zero_grad=True)

            # ========== SECOND PASS ==========
            with autocast("cuda", enabled=torch.cuda.is_available()):
                s_out1_2 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                s_out2_2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    output_hidden_states=False,
                    return_dict=True,
                )

                S_last1_2 = s_out1_2.last_hidden_state
                S_last2_2 = s_out2_2.last_hidden_state
                S_cls1_2 = S_last1_2[:, 0, :]
                S_cls2_2 = S_last2_2[:, 0, :]

                loss_task_2, _ = info_nce(
                    S_cls1_2, S_cls2_2, temperature=cfg.temperature
                )

                student_outputs_2 = {
                    "hidden_states": s_out1_2.hidden_states,
                    "last_hidden_state": S_last1_2,
                }

                loss_2, _ = self.criterion(
                    student_outputs=student_outputs_2,
                    teacher_cls=teacher_cls,
                    task_loss=loss_task_2,
                )

                loss_2 = loss_2.float()

            # Check loss_2 is finite
            if not is_finite(loss_2):
                raise RuntimeError("loss_2 NaN/Inf")

            # Check loss_2 finite before backward
            if not is_finite(loss_2):
                raise RuntimeError(
                    f"loss_2 NaN/Inf at epoch={self.current_epoch} step={self.current_step}"
                )

            # Backward pass 2 - IMPORTANT: Do NOT scale (plain backward)
            loss_2.backward()

            # Check gradients again
            if not grads_are_finite(self.optimizer):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                return loss, {**metrics, "skip": "grad_inf_p2"}

            # SAM second step
            self.optimizer.second_step(zero_grad=True)
            self.scaler.update()
            self.scheduler.step()

            # Clean up
            del s_out1, s_out2, s_out1_2, s_out2_2
            del student_outputs, student_outputs_2

            return loss, metrics

        # Standard distillation methods with teacher inference
        batch_s, batch_t = {}, {}
        for k, v in batch.items():
            if not torch.is_tensor(v):
                continue
            if k.endswith("_stu") or k == "labels":
                batch_s[k] = v.to(self.device_s, non_blocking=True)
            if k.endswith("_tea"):
                batch_t[k] = v.to(self.device_t, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            need_atts = method == "emo"
            with torch.inference_mode():
                t_out1 = self.model_teacher(
                    input_ids=batch_t["input_ids1_tea"],
                    attention_mask=batch_t["attention_mask1_tea"],
                    output_attentions=need_atts,
                    return_dict=True,
                )
                T_last1 = t_out1.last_hidden_state
                T_cls1 = last_token_pool(T_last1, batch_t["attention_mask1_tea"])

                T_last1 = T_last1.to(self.device_s, non_blocking=True)
                T_cls1 = T_cls1.to(self.device_s, non_blocking=True)

                if need_atts:
                    T_atts = tuple(
                        att.to(self.device_s, non_blocking=True)
                        for att in t_out1.attentions
                    )
                    T_last2 = None
                    T_atts2 = None
                    if "input_ids2_tea" in batch_t:
                        t_out2 = self.model_teacher(
                            input_ids=batch_t["input_ids2_tea"],
                            attention_mask=batch_t["attention_mask2_tea"],
                            output_attentions=True,
                            return_dict=True,
                        )
                        T_last2 = t_out2.last_hidden_state.to(
                            self.device_s, non_blocking=True
                        )
                        T_atts2 = tuple(
                            attention.to(self.device_s, non_blocking=True)
                            for attention in t_out2.attentions
                        )

            # Different models have different forward signatures
            if method == "stella":
                # StellaModel doesn't accept output_attentions or return_dict
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                )
            elif method == "emo":
                # EMO needs attentions
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_attentions=True,
                    return_dict=True,
                )
                s_out2 = None
                if "input_ids2_stu" in batch_s:
                    s_out2 = self.model_student(
                        input_ids=batch_s["input_ids2_stu"],
                        attention_mask=batch_s["attention_mask2_stu"],
                        output_attentions=True,
                        return_dict=True,
                    )
            else:
                # CDM, DSKD - standard transformers models
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    return_dict=True,
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    return_dict=True,
                )
            if method != "stella":
                S_last1 = s_out1.last_hidden_state
                S_last2 = None if s_out2 is None else s_out2.last_hidden_state
                S_cls1 = S_last1[:, 0, :]
                S_cls2 = None if S_last2 is None else S_last2[:, 0, :]
            else:
                S_cls1 = s_out1["pooled"]
                S_cls2 = s_out2["pooled"]

            if method == "emo":
                loss_task, task_metrics = self._compute_task_loss(
                    S_cls1, S_cls2, batch_s
                )
            else:
                loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)
                task_metrics = {}

            # ========== Method-specific KD loss ==========
            if method == "cdm":
                if self.proj_s2t is None:
                    d_s, d_t = S_cls1.size(-1), T_cls1.size(-1)
                    self.proj_s2t = nn.Linear(d_s, d_t, bias=False).to(self.device_s)
                    self.optimizer.add_param_group(
                        {
                            "params": self.proj_s2t.parameters(),
                            "lr": cfg.learning_rate * 2,
                        }
                    )
                    print(f"Initialized projection layer: {d_s} -> {d_t}")

                keep_s1 = batch_s["attention_mask1_stu"].bool() & (
                    ~batch_s["special_tokens_mask1_stu"].bool()
                )
                keep_t1 = batch_t["attention_mask1_tea"].to(self.device_s).bool() & (
                    ~batch_t["special_tokens_mask1_tea"].to(self.device_s).bool()
                )

                kd_dtw = self.criterion.compute_cdm_loss(
                    S_last=S_last1,
                    T_last=T_last1,
                    batch_input_ids_stu=batch["input_ids1_stu"],
                    batch_input_ids_tea=batch["input_ids1_tea"],
                    keep_mask_stu=keep_s1,
                    keep_mask_tea=keep_t1,
                    proj_s2t=self.proj_s2t,
                    device_s=self.device_s,
                    epoch=self.current_epoch,
                    step=self.current_step,
                )

                S_proj_cls1 = self.proj_s2t(S_cls1)
                S_proj_cls1_norm = F.normalize(S_proj_cls1, p=2, dim=-1)
                T_cls1_norm = F.normalize(T_cls1, p=2, dim=-1)
                kd_cls = F.mse_loss(S_proj_cls1_norm, T_cls1_norm)

                loss = (
                    cfg.w_task * loss_task
                    + cfg.alpha_dtw * kd_dtw * 100
                    + cfg.w_cls * kd_cls
                )

                metrics = {
                    "loss_total": loss.item(),
                    "loss_task": loss_task.item(),
                    "loss_kd_dtw": kd_dtw.item()
                    if isinstance(kd_dtw, torch.Tensor)
                    else kd_dtw,
                    "loss_kd_cls": kd_cls.item(),
                }

            elif method == "dskd":
                mask_s1 = batch_s["attention_mask1_stu"]
                mask_t1 = batch_t["attention_mask1_tea"].to(self.device_s)

                spec_s1 = batch_s.get("special_tokens_mask1_stu", None)
                spec_t1 = batch_t.get("special_tokens_mask1_tea", None)
                if spec_t1 is not None:
                    spec_t1 = spec_t1.to(self.device_s)

                loss, metrics = self.criterion.compute_dskd_loss(
                    S_last=S_last1,
                    T_last=T_last1,
                    S_cls=S_cls1,
                    T_cls=T_cls1,
                    mask_student=mask_s1,
                    mask_teacher=mask_t1,
                    task_loss=loss_task,
                    special_tokens_mask_student=spec_s1,
                    special_tokens_mask_teacher=spec_t1,
                    device=self.device_s,
                )

            elif method == "emo":

                class TeacherOutput:
                    def __init__(self, last_hidden_state, attentions):
                        self.last_hidden_state = last_hidden_state
                        self.attentions = attentions

                class StudentOutput:
                    def __init__(self, last_hidden_state, attentions):
                        self.last_hidden_state = last_hidden_state
                        self.attentions = attentions

                teacher_outputs = TeacherOutput(T_last1, T_atts)
                student_outputs = StudentOutput(S_last1, s_out1.attentions)

                att_loss_weight = getattr(cfg, "att_loss_weight", 0.1)
                ot_loss_weight = getattr(cfg, "ot_loss_weight", 1.0)

                kd_loss, kd_metrics = self.criterion.compute_emo_loss(
                    teacher_outputs=teacher_outputs,
                    student_outputs=student_outputs,
                    input_ids_tea=batch_t["input_ids1_tea"].to(self.device_s),
                    input_ids_stu=batch_s["input_ids1_stu"],
                    attention_mask_tea=batch_t["attention_mask1_tea"].to(self.device_s),
                    attention_mask_stu=batch_s["attention_mask1_stu"],
                    tok_teacher=self.tok_teacher,
                    tok_student=self.tok_student,
                    att_loss_weight=att_loss_weight,
                    ot_loss_weight=ot_loss_weight,
                )
                if S_last2 is not None and T_last2 is not None:
                    teacher_outputs2 = TeacherOutput(T_last2, T_atts2)
                    student_outputs2 = StudentOutput(S_last2, s_out2.attentions)
                    kd_loss2, kd_metrics2 = self.criterion.compute_emo_loss(
                        teacher_outputs=teacher_outputs2,
                        student_outputs=student_outputs2,
                        input_ids_tea=batch_t["input_ids2_tea"].to(self.device_s),
                        input_ids_stu=batch_s["input_ids2_stu"],
                        attention_mask_tea=batch_t["attention_mask2_tea"].to(
                            self.device_s
                        ),
                        attention_mask_stu=batch_s["attention_mask2_stu"],
                        tok_teacher=self.tok_teacher,
                        tok_student=self.tok_student,
                        att_loss_weight=att_loss_weight,
                        ot_loss_weight=ot_loss_weight,
                    )
                    kd_loss = 0.5 * (kd_loss + kd_loss2)
                    kd_metrics = {
                        key: 0.5 * (kd_metrics[key] + kd_metrics2[key])
                        for key in kd_metrics
                    }

                w_task = getattr(cfg, "w_task", 0.5)
                alpha_kd = getattr(cfg, "alpha_kd", 0.5)
                loss = w_task * loss_task + alpha_kd * kd_loss

                metrics = {
                    "loss_total": loss.item(),
                    "loss_task": loss_task.item(),
                    **task_metrics,
                    **kd_metrics,
                }

            elif method == "stella":
                if self.current_stage == 1:
                    S_emb = s_out1["fc1"]
                    loss, metrics = stella_stage1_loss(
                        S_emb,
                        T_cls1,
                        w_cos=getattr(cfg, "w_cos_stage1", 10.0),
                        w_sim=getattr(cfg, "w_sim_stage1", 200.0),
                        w_tri=getattr(cfg, "w_tri_stage1", 20.0),
                    )
                else:
                    loss, metrics = stella_stage2_loss(
                        S_cls1,
                        S_cls2,
                        s_out1["fc1"],
                        s_out1["fc2"],
                        s_out1["fc3"],
                        s_out1["fc4"],
                        T_cls1,
                        temperature=cfg.temperature,
                        w_task=cfg.w_task,
                        w_cos=getattr(cfg, "w_cos_stage2", 10.0),
                        w_sim=getattr(cfg, "w_sim_stage2", 200.0),
                        w_tri=getattr(cfg, "w_tri_stage2", 20.0),
                    )

            else:
                raise ValueError(f"Unknown distillation method: {method}")

            loss = loss.float()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss, metrics

    def train_epoch(self, epoch: int):
        self.model_student.train()
        self.current_epoch = epoch

        # Redraw the candidate sets. Without this the student sees the identical
        # anchor/candidate comparisons every epoch, fits them in the first one, and
        # spends the rest of the run overfitting a frozen 32-way problem.
        if hasattr(self.train_ds, "set_epoch"):
            sampling_epoch = (
                epoch if getattr(self.config, "resample_candidates_per_epoch", True) else 0
            )
            self.train_ds.set_epoch(sampling_epoch)

        total_loss = 0.0
        n_items = 0
        metric_totals = {}
        epoch_step_times = []
        peak_memory_mb = 0.0
        # Per-step diagnostics, buffered here and written once at the end of the epoch.
        # Epoch means alone cannot show *when* inside an epoch a curve flattened, and
        # the HeatGeo objective saturated inside epoch 1 on the previous run -- five
        # points per curve is a summary, not a diagnosis. Buffering keeps this to one
        # file write per epoch rather than one per step.
        step_records: list[dict] = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for step, batch in enumerate(pbar):
            self.current_step = step

            self.sync_all()
            t0 = time.perf_counter()

            loss, metrics = self.train_step(batch)

            self.sync_all()
            dt = time.perf_counter() - t0
            epoch_step_times.append(dt)
            self.global_step += 1
            if getattr(self, "use_wandb", False) and WANDB_AVAILABLE:
                log_payload = {
                    "train/epoch": epoch + 1,
                    "train/global_step": self.global_step,
                    "train/step_seconds": dt,
                }
                log_payload.update(self._flatten_metrics("train", metrics))
                wandb.log(log_payload, step=self.global_step)

            bs = batch["input_ids1_stu"].size(0)
            loss_value = loss.item()
            total_loss += loss_value * bs
            n_items += bs
            avg_loss = total_loss / max(1, n_items)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_totals[key] = metric_totals.get(key, 0.0) + float(value) * bs

            step_record = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "step": step,
                "batch_size": int(bs),
                "loss": float(loss_value),
                "step_seconds": float(dt),
                # train_step() has already called scheduler.step(), so this is the rate
                # the *next* step will use.
                "lr_next": float(self.optimizer.param_groups[0]["lr"]),
            }
            step_record.update(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
            )
            step_records.append(step_record)

            mem_info = {}
            for dev_id in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(dev_id) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(dev_id) / 1024**2
                peak_memory_mb = max(peak_memory_mb, mem_alloc)
                mem_info[f"gpu{dev_id}"] = f"{mem_alloc:.0f}/{mem_reserved:.0f}MB"

            if step >= self.warmup_steps:
                self.step_times.append(dt)
                self.ma_window.append(dt)
                avg_step = sum(self.step_times) / len(self.step_times)
                ma_step = sum(self.ma_window) / len(self.ma_window)

                postfix = {
                    "avg_loss": f"{avg_loss:.4f}",
                    "ms/step": f"{avg_step * 1000:.1f}",
                    "ms/step(ma)": f"{ma_step * 1000:.1f}",
                    "it/s": f"{1.0 / ma_step:.2f}",
                    **mem_info,
                }

                for k, v in metrics.items():
                    if k != "loss_total":
                        # Format only if v is numeric (not string like 'skip': 'grad_inf_p1')
                        postfix[k] = (
                            f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                        )

                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", **mem_info})

        if len(self.step_times) > 0:
            epoch_avg = sum(self.step_times) / len(self.step_times)
            print(
                f"[Epoch {epoch + 1}] Avg step time = {epoch_avg * 1000:.2f} ms "
                f"({1.0 / epoch_avg:.2f} it/s)"
            )

        print(f"Done train_epoch {epoch + 1}")
        self.log_step_records(step_records)
        epoch_means = {key: value / max(1, n_items) for key, value in metric_totals.items()}

        # The progress bar shows the *last* step's diagnostics, and the last batch is
        # usually a short remainder -- with 13553 items and batch 16 it holds a single
        # anchor, which makes target_entropy/js_floor swing wildly for reasons that
        # have nothing to do with training. Print the example-weighted epoch means.
        if epoch_means:
            headline = [
                "loss_diff", "js_floor", "loss_excess", "target_entropy",
                "student_entropy", "student_entropy_ratio", "student_top1",
                "target_top1", "candidates_per_anchor",
            ]
            shown = [k for k in headline if k in epoch_means]
            shown += sorted(k for k in epoch_means if k.startswith("kl_scale"))
            body = "  ".join(f"{k}={epoch_means[k]:.4f}" for k in shown)
            print(f"[Epoch {epoch + 1}] mean over {n_items} examples: {body}")

        self.last_epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "mean_step_seconds": (
                sum(epoch_step_times) / len(epoch_step_times)
                if epoch_step_times
                else 0.0
            ),
            "peak_memory_mb": peak_memory_mb,
            **epoch_means,
        }
        return avg_loss

    def save_checkpoint(self, epoch: int, metrics: dict | None = None):
        cfg = self.config
        if not cfg.save_dir:
            return
        if epoch in self._saved_checkpoint_epochs:
            print(
                f"Checkpoint for epoch {epoch + 1} already saved; skipping duplicate."
            )
            return
        os.makedirs(cfg.save_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model_student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
        }

        if self.proj_s2t is not None:
            checkpoint["proj_s2t_state_dict"] = self.proj_s2t.state_dict()

        if self.criterion is not None and hasattr(self.criterion, "state_dict"):
            checkpoint["criterion_state_dict"] = self.criterion.state_dict()
        if self.task_head is not None:
            checkpoint["task_head_state_dict"] = self.task_head.state_dict()

        if metrics:
            checkpoint["metrics"] = metrics

        path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        print(f"Done save_checkpoint for epoch {epoch + 1}")

        if cfg.save_best and metrics and "loss" in metrics:
            if not hasattr(self, "best_loss") or metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                best_path = os.path.join(cfg.save_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                print(f"Best model saved: {best_path}")

        self.save_student_weights(epoch)
        self._saved_checkpoint_epochs.add(epoch)

    def save_student_weights(self, epoch: int):
        weights_dir = getattr(self.config, "weights_dir", None)
        if not weights_dir:
            return
        if epoch in self._saved_student_weight_epochs:
            print(
                f"Student weights for epoch {epoch + 1} already saved; "
                "skipping duplicate."
            )
            return

        destination_dir = Path(weights_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"student_epoch_{epoch + 1}.pt"
        destination_tmp = destination.with_suffix(".pt.tmp")
        payload = {
            "epoch": epoch + 1,
            "student_model_name": self.config.student_model_name,
            "teacher_model_name": self.config.teacher_model_name,
            "model_state_dict": self.model_student.state_dict(),
        }

        local_dir = Path(self.config.save_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, local_tmp_name = tempfile.mkstemp(
            prefix=f".student_epoch_{epoch + 1}_",
            suffix=".pt",
            dir=local_dir,
        )
        os.close(file_descriptor)
        local_tmp = Path(local_tmp_name)

        try:
            torch.save(payload, local_tmp)
            shutil.copy2(local_tmp, destination_tmp)
            os.replace(destination_tmp, destination)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise OSError(
                    f"Saved student weights are missing or empty: {destination}"
                )
        finally:
            local_tmp.unlink(missing_ok=True)
            destination_tmp.unlink(missing_ok=True)

        print(f"Student weights saved: {destination}")
        self._saved_student_weight_epochs.add(epoch)

    @staticmethod
    def _benchmark_name(path: str, split: str) -> str:
        name = Path(path).stem
        suffix = f"_{split}"
        return name.removesuffix(suffix)

    @staticmethod
    def _metric_details(values: dict[str, Any]) -> str:
        labels = {
            "accuracy": "Acc",
            "f1": "F1",
            "precision": "P",
            "recall": "R",
            "average_precision": "AP",
            "spearman": "Spearman",
        }
        details = []
        for key, label in labels.items():
            value = values.get(key)
            if isinstance(value, (int, float)):
                details.append(f"{label}={100.0 * float(value):.2f}")
        return " ".join(details)

    def print_evaluation_table(
        self,
        split: str,
        results: dict[str, Any],
    ) -> None:
        primary_metrics = {
            "classification": "f1",
            "pair": "average_precision",
            "sts": "spearman",
        }
        rows = []
        for family in ("classification", "pair", "sts"):
            family_scores = []
            for path, raw_values in results.get(family, {}).items():
                values = (
                    {"spearman": raw_values} if family == "sts" else dict(raw_values)
                )
                metric_name = primary_metrics[family]
                score = float(values[metric_name])
                family_scores.append(score)
                rows.append(
                    (
                        family,
                        self._benchmark_name(path, split),
                        metric_name,
                        f"{100.0 * score:.2f}",
                        self._metric_details(values),
                    )
                )
            if family_scores:
                rows.append(
                    (
                        family,
                        "MEAN",
                        primary_metrics[family],
                        f"{100.0 * sum(family_scores) / len(family_scores):.2f}",
                        "",
                    )
                )

        title = (
            f"VALIDATION - EPOCH {self.current_epoch + 1}"
            if split == "validation"
            else "FINAL TEST"
        )
        headers = ("Family", "Benchmark", "Primary metric", "Score", "Details")
        widths = [
            max([len(headers[index]), *(len(row[index]) for row in rows)])
            for index in range(len(headers))
        ]
        separator = "-+-".join("-" * width for width in widths)

        print("\n" + "=" * len(separator))
        print(title)
        print("=" * len(separator))
        print(
            " | ".join(
                headers[index].ljust(widths[index]) for index in range(len(headers))
            )
        )
        print(separator)
        for row in rows:
            print(
                " | ".join(row[index].ljust(widths[index]) for index in range(len(row)))
            )
        print("=" * len(separator) + "\n")

    def evaluate(self, split: str = "validation"):
        if split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")
        if split == "validation":
            classification_tasks = eval_cls_tasks
            pair_tasks = eval_pair_tasks
            sts_tasks = eval_sts_tasks
            thresholds = None
        else:
            classification_tasks = test_cls_tasks
            pair_tasks = test_pair_tasks
            sts_tasks = test_sts_tasks
            thresholds = getattr(self, "pair_validation_thresholds", None)
            if thresholds is None:
                raise RuntimeError(
                    "Pair test evaluation requires thresholds selected on validation data"
                )

        classification = eval_classification_task(
            self.model_student, classification_tasks, self.tok_student
        )
        pair, selected_thresholds = eval_pair_task(
            self.model_student,
            pair_tasks,
            self.tok_student,
            thresholds=thresholds,
        )
        sts = eval_sts_task(self.model_student, sts_tasks, self.tok_student)
        if split == "validation":
            self.pair_validation_thresholds = selected_thresholds
        results = {
            "classification": classification,
            "pair": pair,
            "sts": sts,
        }
        self.print_evaluation_table(split, results)
        return results

    def log_step_records(self, records: list[dict]):
        """Append one JSONL line per training step to `step_metrics.jsonl`.

        Written separately from metrics.jsonl, which stays one record per epoch: the
        two have different row counts and different consumers, and mixing them would
        force every reader of the epoch table to filter.
        """
        if not records or not self.config.save_dir:
            return
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, "step_metrics.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(record, default=float, sort_keys=True) + "\n"
                for record in records
            )

    def log_experiment_record(self, record: dict[str, Any]):
        if not self.config.save_dir:
            return
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, "metrics.jsonl")
        payload = {
            "method": self.config.distill_method,
            "seed": self.config.seed,
            **record,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=float, sort_keys=True) + "\n")

    def train(self):
        cfg = self.config

        if cfg.distill_method == "stella":
            print("\n" + "=" * 70)
            print("Starting Stella 2-Stage Training...")
            print("=" * 70)
            print(f"Student: {cfg.student_model_name}")
            print(f"Teacher: {cfg.teacher_model_name}")
            print(f"Stage 1 epochs: {cfg.epochs_stage1}")
            print(f"Stage 2 epochs: {cfg.epochs_stage2}")
            print(f"Batch size: {cfg.batch_size}")
            print(f"Learning rate: {cfg.learning_rate}")
            print("=" * 70 + "\n")

            print("\n" + "=" * 70)
            print("STAGE 1: Freeze backbone + fc2,3,4, train fc1 only")
            print("=" * 70)

            for p in self.model_student.backbone.parameters():
                p.requires_grad = False
            for head in [
                self.model_student.fc2,
                self.model_student.fc3,
                self.model_student.fc4,
            ]:
                for p in head.parameters():
                    p.requires_grad = False

            print("Frozen: backbone, fc2, fc3, fc4")
            print("Trainable: fc1")

            self.current_stage = 1
            for epoch in range(cfg.epochs_stage1):
                avg_loss = self.train_epoch(epoch)
                self.log_experiment_record(
                    {"stage": 1, "train": self.last_epoch_metrics}
                )

                if (epoch + 1) % cfg.save_every == 0:
                    self.save_checkpoint(epoch, {"loss": avg_loss})

            print("\n" + "=" * 70)
            print("STAGE 1 COMPLETED!")
            print("=" * 70 + "\n")

            print("\n" + "=" * 70)
            print("STAGE 2: Unfreeze all, train full model")
            print("=" * 70)

            for p in self.model_student.parameters():
                p.requires_grad = True

            print("Unfrozen: all parameters")
            print("Trainable: backbone, fc1, fc2, fc3, fc4")

            self.optimizer = optim.AdamW(
                self.model_student.parameters(), lr=cfg.learning_rate
            )
            self.scheduler = get_scheduler(
                "cosine",
                optimizer=self.optimizer,
                num_warmup_steps=int(len(self.train_loader) * cfg.warmup_ratio),
                num_training_steps=len(self.train_loader) * cfg.epochs_stage2,
            )

            self.step_times = []
            self.ma_window = deque(maxlen=50)

            self.current_stage = 2
            for epoch in range(cfg.epochs_stage2):
                avg_loss = self.train_epoch(epoch)
                validation_results = None

                print("\n" + "=" * 60)
                print(f"Evaluation after Stage2 Epoch {epoch + 1}")
                print("=" * 60)

                try:
                    # from src.evaluation.evaluation_model_define import (
                    #     eval_classification_task,
                    #     eval_pair_task,
                    #     eval_sts_task,
                    #     test_cls_tasks,
                    #     test_pair_tasks,
                    #     test_sts_tasks
                    # )
                    validation_results = self.evaluate("validation")
                except Exception as e:
                    print(f"Warning: Evaluation failed with error: {e}")
                    print("Continuing training...")

                print("=" * 60 + "\n")
                self.log_experiment_record(
                    {
                        "stage": 2,
                        "train": self.last_epoch_metrics,
                        "validation": validation_results,
                    }
                )

                if (epoch + 1) % cfg.save_every == 0:
                    self.save_checkpoint(epoch, {"loss": avg_loss})

            print("\n" + "=" * 70)
            print("STAGE 2 COMPLETED!")
            print("=" * 70)

            self.save_checkpoint(cfg.epochs_stage2 - 1, {"loss": avg_loss})
            try:
                test_results = self.evaluate("test")
                if (
                    getattr(self, "use_wandb", False)
                    and WANDB_AVAILABLE
                    and test_results is not None
                ):
                    wandb.log(
                        self._flatten_metrics("test", test_results),
                        step=self.global_step,
                    )
                self.log_experiment_record({"stage": 2, "test": test_results})
            except Exception as e:
                print(f"Warning: Final test evaluation failed with error: {e}")

            print("\n" + "=" * 70)
            print("Training completed successfully!")
            print("=" * 70)

        else:
            print("\n" + "=" * 60)
            print("Starting training...")
            print("=" * 60)
            print(f"Method: {cfg.distill_method}")
            print(f"Student: {cfg.student_model_name}")
            print(f"Teacher: {cfg.teacher_model_name}")
            print(f"Epochs: {cfg.epochs}")
            print(f"Batch size: {cfg.batch_size}")
            print(f"Learning rate: {cfg.learning_rate}")
            print("=" * 60 + "\n")

            for epoch in range(cfg.epochs):
                self.current_epoch = epoch
                
                # ---- Walk Curriculum ----
                use_walk = (
                    getattr(cfg, "num_walks", 0) > 0
                    and epoch >= getattr(cfg, "walk_start_epoch", 1)
                )
                if hasattr(self, "heatgeo_criterions"):
                    for crit in self.heatgeo_criterions:
                        crit.use_walk_loss = use_walk
                elif self.criterion is not None and hasattr(self.criterion, "use_walk_loss"):
                    self.criterion.use_walk_loss = use_walk
                if use_walk:
                    print(f"Walk kernel-matching loss is ENABLED for Epoch {epoch + 1}")
                # ------------------------------------

                avg_loss = self.train_epoch(epoch)
                validation_results = None

                print("\n" + "=" * 60)
                print(f"Evaluation after Epoch {epoch + 1}")
                print("=" * 60)

                if (epoch + 1) % cfg.eval_every == 0:
                    try:
                        validation_results = self.evaluate("validation")
                        if (
                            getattr(self, "use_wandb", False)
                            and WANDB_AVAILABLE
                            and validation_results is not None
                        ):
                            wandb.log(
                                self._flatten_metrics("validation", validation_results),
                                step=self.global_step,
                            )
                    except Exception as e:
                        print(f"Warning: Validation failed with error: {e}")
                        print("Continuing training...")

                print("=" * 60 + "\n")
                self.log_experiment_record(
                    {
                        "train": self.last_epoch_metrics,
                        "validation": validation_results,
                    }
                )

                final_weights_only = bool(
                    getattr(cfg, "final_weights_only", False)
                )
                should_save_final_weights = (
                    final_weights_only and epoch + 1 == cfg.epochs
                )
                should_save_checkpoint = (
                    not final_weights_only and (epoch + 1) % cfg.save_every == 0
                )
                if should_save_final_weights or should_save_checkpoint:
                    try:
                        if should_save_final_weights:
                            if not getattr(cfg, "weights_dir", None):
                                raise ValueError(
                                    "final_weights_only requires weights_dir"
                                )
                            self.save_student_weights(epoch)
                        else:
                            self.save_checkpoint(epoch, {"loss": avg_loss})
                    except Exception as e:
                        if getattr(cfg, "weights_dir", None):
                            raise RuntimeError(
                                f"Required epoch {epoch + 1} weights could not be saved"
                            ) from e
                        print(f"Warning: Saving checkpoint failed with error: {e}")
                        print("Continuing training...")

            print("\n" + "=" * 60)
            print("Training completed!")
            print("=" * 60)
            print("Done train()")

            if getattr(cfg, "final_weights_only", False):
                if not getattr(cfg, "weights_dir", None):
                    raise ValueError("final_weights_only requires weights_dir")
                self.save_student_weights(cfg.epochs - 1)
            else:
                self.save_checkpoint(cfg.epochs - 1, {"loss": avg_loss})
            try:
                test_results = self.evaluate("test")
                if (
                    getattr(self, "use_wandb", False)
                    and WANDB_AVAILABLE
                    and test_results is not None
                ):
                    wandb.log(
                        self._flatten_metrics("test", test_results),
                        step=self.global_step,
                    )
                self.log_experiment_record({"test": test_results})
            except Exception as e:
                print(f"Warning: Final test evaluation failed with error: {e}")
