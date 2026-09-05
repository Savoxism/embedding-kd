import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler
from transformers import __version__ as transformers_version

from src.cache_teacher import (
    cache_teacher_embeddings,
    load_cached_embeddings,
    validate_cached_embeddings,
)
from src.criterions.contextual_dynamic_mapping import ContextualDynamicMapping
from src.criterions.dual_space_kd import DualSpaceKD
from src.criterions.emo_embedding_distillation import EMODistillation
from src.criterions.ggpkd_distillation import GGPKDDistillation
from src.criterions.relational_kd import RelationalKnowledgeDistillation
from src.criterions.stella_distillation import (
    StellaModel,
)
from src.data_utils import DualTokenizerCollate, TextPairRaw
from src.data_utils.dataset_cache import (
    DualTokenizerCollateWithTeacher,
    GGPKDCollate,
    TextPairWithTeacher,
    TextPairWithTeacherAndGGPKD,
)
from src.distill.benchmarks import add_domain_averages, print_evaluation_table
from src.distill.checkpointing import save_checkpoint, save_student_weights
from src.distill.geometry import build_probe_index, build_probe_set, probe_geometry
from src.distill.numerics import (
    assert_module_parameters_finite,
)
from src.distill.stella_trainer import train as stella_train
from src.distill.steps.ggpkd import step as ggpkd_step
from src.distill.steps.rkd import step as rkd_step
from src.distill.steps.standard import step as standard_step
from src.distill.steps.talas import step as talas_step
from src.distill.telemetry import append_epoch_record, new_run_id, write_run_manifest
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
from src.ggpkd import GGPKDCandidateSampler, build_or_load_ggpkd_artifact
from src.ggpkd.policy import (
    FIXED_BANDWIDTH_TEMP,
    ROW_COVERAGE_TAU,
    derive_diffusion_quota,
)
from src.loss import info_nce


class KnowledgeDistiller:
    def __init__(self, config):
        self.config = config
        self.global_step = 0
        self.current_epoch = 0
        self.current_step = 0
        self._saved_checkpoint_epochs = set()
        self._saved_student_weight_epochs = set()
        # Identifies every file this run writes. metrics.jsonl / step_metrics.jsonl
        # / epochs.jsonl are all opened in append mode, so without it a save_dir
        # reused across runs interleaves them with no way to separate the rows.
        self.run_id = new_run_id()
        # Cumulative student forward-pass budget, incremented by the ggpkd step.
        # Cumulative rather than per-step because that is the quantity the
        # ablations are matched on, and a mean over steps loses it.
        self.encoded_texts_total = 0
        self.encoded_tokens_total = 0
        self.probe_texts: list[str] = []
        # Cached teacher vectors for exactly those probe texts, in the same order.
        # Without them the probe reports the student's space in isolation; with
        # them it also reports `teacher_student_spearman`, which is the one
        # geometry number that is comparable across arms without a benchmark.
        self.probe_teacher: torch.Tensor | None = None
        self.setup_seed(config.seed)
        self.setup_devices()
        self.setup_models()
        self.setup_data()
        self.proj_s2t = None
        self.setup_training()

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
            # Built here, not lazily on the first batch. `add_param_group` after the
            # scheduler exists leaves LambdaLR with one lr_lambda for two param
            # groups; torch >= 2.6 zips them with strict=True, so the first
            # `scheduler.step()` raises. Every other method that adds a group does
            # so here and rebuilds the scheduler -- CDM was the one that did not.
            d_s = self.model_student.config.hidden_size
            d_t = self.model_teacher.config.hidden_size
            self.proj_s2t = nn.Linear(d_s, d_t, bias=False).to(self.device_s)
            self.optimizer.add_param_group(
                {
                    "params": self.proj_s2t.parameters(),
                    "lr": config.learning_rate * 2,
                }
            )
            self.scheduler = self._build_scheduler()
            print(f"Initialized CDM projection layer: {d_s} -> {d_t}")
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
        elif config.distill_method == "rkd":
            self.criterion = RelationalKnowledgeDistillation(
                distance_weight=config.rkd_distance_weight,
                angle_weight=config.rkd_angle_weight,
                task_weight=config.w_task,
                eps=config.eps_norm,
            ).to(self.device_s)
            print(
                "RKD-DA criterion initialized: "
                f"distance={config.rkd_distance_weight}, "
                f"angle={config.rkd_angle_weight}, task={config.w_task}"
            )
        elif config.distill_method == "ggpkd":
            # `self.ggpkd_artifact` is set unconditionally in the data path before
            # this runs, so indexing it directly is right: a `.get()` fallback would
            # hand the criterion `None` and make it blame the graph artifact for what
            # is really a setup-ordering bug.
            artifact = self.ggpkd_artifact
            # --direct_temp 0 derives the last free student temperature from the
            # graph itself: the median entropic-affinity bandwidth. The ambient
            # target is the same softmax-of-cosines construction as the transition
            # rows with the sparsification removed, so the graph's own typical
            # bandwidth is the natural scale for it. Written back onto the config so
            # the run manifest and banner record the concrete value, exactly as
            # derived diffusion_quota is.
            if config.direct_temp == 0.0:
                row_temps = artifact.get("row_temps")
                config.direct_temp = (
                    float(row_temps.median())
                    if row_temps is not None
                    else FIXED_BANDWIDTH_TEMP
                )
                print(
                    f"Derived direct_temp={config.direct_temp:.4f} "
                    "(median graph bandwidth; requested via --direct_temp 0)"
                )
            # `use_ambient=False` is the S4 deletion arm: withholding the bank is
            # what removes scale r=0, because the criterion derives `use_direct`
            # from whether it has teacher embeddings at all.
            self.criterion = GGPKDDistillation(
                diffusion_scales=config.diffusion_scales,
                teacher_embeddings=(
                    self.teacher_cls_all if config.use_ambient else None
                ),
                direct_temp=config.direct_temp,
                row_weight=config.row_weight,
                relation_target=config.relation_target,
                row_temps=artifact["row_temps"],
                transition_neighbors=artifact["transition_neighbors"],
                transition_probs=artifact["transition_probs"],
            ).to(self.device_s)
            self.scheduler = self._build_scheduler()
            print(
                "GGPKD criterion initialized: "
                f"batch_local={config.batch_local}, "
                f"ambient={config.use_ambient}, "
                f"relation_target={config.relation_target}, "
                f"row_weight={config.row_weight}"
            )
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

    def setup_models(self):
        cfg = self.config

        print("Loading tokenizers...")
        tokenizer_kwargs = {"use_fast": True}
        cached_teacher_methods = {"talas", "ggpkd", "rkd"}
        self._teacher_cache_ready = (
            cfg.distill_method in cached_teacher_methods
            and Path(cfg.cache_path).is_file()
        )
        self.tok_student = AutoTokenizer.from_pretrained(
            cfg.student_model_name,
            **tokenizer_kwargs,
        )
        if self._teacher_cache_ready:
            self.tok_teacher = None
            print(
                f"{cfg.distill_method.upper()} teacher cache found; "
                "skipping teacher tokenizer/model loading"
            )
        else:
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

        if self._teacher_cache_ready:
            self.model_teacher = None
        else:
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
                    "Using eager attention implementation for EMO "
                    "(required for output_attentions)"
                )

            self.model_teacher = AutoModel.from_pretrained(
                cfg.teacher_model_name, **teacher_kwargs
            )

        self.model_student.to(self.device_s)
        if self.model_teacher is not None:
            self.model_teacher.to(self.device_t)

        student_dtype = next(self.model_student.parameters()).dtype
        print(f"Student training dtype: {student_dtype}")
        assert_module_parameters_finite(self.model_student, "Student model after load")

        if self.model_teacher is not None:
            self.model_teacher.eval()
            for p in self.model_teacher.parameters():
                p.requires_grad_(False)

        print("Models loaded successfully!")
        print("Done setup_models")

    def _resolve_ggpkd_anchor_column(self, df: pd.DataFrame) -> str:
        cfg = self.config
        column = cfg.ggpkd_anchor_column
        if column is not None:
            if column not in df.columns:
                raise ValueError(
                    f"ggpkd_anchor_column={column!r} is not a column of "
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
                f"GGPKD needs column {column!r} for task_type={cfg.task_type!r}"
            )

        # The teacher graph is built over this column only. If a genuine second view
        # exists it is dropped, and doing that silently would leave the graph
        # describing a different object than the loss thinks it does.
        partner = {"pair_cls": "hypothesis", "pair_reg": "sentence2"}.get(cfg.task_type)
        if partner in df.columns and not df[column].equals(df[partner]):
            print(
                f"WARNING: GGPKD uses only {column!r}; {partner!r} differs from it "
                f"and is not distilled. Set ggpkd_anchor_column explicitly if that "
                f"is not what you want."
            )
        return column

    def _prepare_ggpkd_frame(
        self, df: pd.DataFrame, anchor_column: str
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Drop exact duplicate anchors and report the surviving row positions.

        Two identical texts have cos(s_i, s_j) = 1 for every parameter setting, so
        their logit sits at the ceiling with no gradient while still consuming
        teacher mass and a candidate slot.
        """
        keep_positions = np.arange(len(df), dtype=np.int64)
        texts = df[anchor_column].astype(str)
        duplicated = texts.duplicated(keep="first").to_numpy()
        if not duplicated.any():
            print(f"GGPKD corpus: {len(df)} rows, no duplicate anchors")
            return df.reset_index(drop=True), keep_positions

        keep_positions = np.flatnonzero(~duplicated).astype(np.int64)
        deduped = df.iloc[keep_positions].reset_index(drop=True)
        print(
            f"GGPKD corpus dedup on {anchor_column!r}: "
            f"{len(df)} -> {len(deduped)} rows ({int(duplicated.sum())} exact duplicates removed)"
        )
        return deduped, keep_positions

    def _ggpkd_source_ids(self, df: pd.DataFrame) -> np.ndarray:
        column = self.config.ggpkd_source_column
        if column not in df.columns:
            print(
                f"GGPKD: no {column!r} column, hard negatives will not be "
                f"restricted to the same source corpus"
            )
            return np.zeros(len(df), dtype=np.int64)
        codes = pd.factorize(df[column].astype(str))[0].astype(np.int64)
        counts = pd.Series(codes).value_counts().to_dict()
        print(
            f"GGPKD sources: {len(counts)} distinct, sizes={sorted(counts.values(), reverse=True)}"
        )
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

        # GGPKD is anchor-only: the teacher graph, the candidate pool and the
        # student forward all consume one string per row. Resolve that column once
        # and keep it, instead of each component re-deriving it from the frame.
        # Fixed probe set for the geometry diagnostics. Sampled from the training
        # corpus with a fixed seed so the numbers are comparable across epochs and
        # across runs; a probe set that moves measures nothing.
        probe_column = next(
            (c for c in ("text", "anchor", "sentence1", "premise") if c in df.columns),
            None,
        )
        if probe_column is not None:
            self.probe_texts = build_probe_set(df, probe_column, size=2048, seed=0)

        self.ggpkd_anchor_column = None
        if cfg.distill_method == "ggpkd":
            self.ggpkd_anchor_column = self._resolve_ggpkd_anchor_column(df)
            df, keep_positions = self._prepare_ggpkd_frame(
                df, self.ggpkd_anchor_column
            )
            self.ggpkd_keep_positions = keep_positions

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

        # TALAS, GGPKD and RKD use cached teacher embeddings.
        if cfg.distill_method in ("talas", "ggpkd", "rkd"):
            cache_path = Path(cfg.cache_path)

            # Check if cache exists
            if cache_path.exists():
                print(f"Loading cached teacher embeddings from: {cache_path}")
                teacher_cls_list = load_cached_embeddings(str(cache_path))
                print(f"Loaded {len(teacher_cls_list)} cached embeddings")
            else:
                print("Cache not found. Pre-computing teacher embeddings...")
                os.makedirs(cache_path.parent, exist_ok=True)

                # The first text of each sample -- exactly what the old caching
                # collate fed the teacher. Taken from TextPairRaw rather than
                # re-derived from the frame so the two cannot drift apart when a
                # task type resolves its columns differently.
                cache_texts = [
                    sample[0] for sample in TextPairRaw(df, cfg.task_type).samples
                ]

                teacher_cls_list = cache_teacher_embeddings(
                    model_teacher=self.model_teacher,
                    texts=cache_texts,
                    tokenizer=self.tok_teacher,
                    max_length=cfg.max_length,
                    device=self.device_t,
                    pooling_method=cfg.pooling_method,
                    normalize=cfg.normalize_cache,
                    dtype=torch.float32
                    if cfg.cache_dtype == "float32"
                    else torch.float16,
                    cache_path=str(cache_path),
                    teacher_model_name=cfg.teacher_model_name,
                )
                print(
                    f"Cached {len(teacher_cls_list)} teacher embeddings to {cache_path}"
                )

            # A stale cache computed before dedup still lines up row-for-row with the
            # original frame, so slice it instead of forcing a teacher re-run.
            keep_positions = getattr(self, "ggpkd_keep_positions", None)
            if (
                cfg.distill_method == "ggpkd"
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

            # Every cached-teacher method now stores one pooled vector per row; the
            # multi-layer producer that could emit [N, L, D] is gone, so a 3-D cache
            # is a stale artifact and should be rejected loudly rather than silently
            # reduced to its last layer.
            validate_cached_embeddings(
                teacher_cls_list,
                len(df),
                cache_path=str(cache_path),
                require_single_layer=True,
            )

            self.teacher_cls_all = teacher_cls_list

            if cfg.distill_method == "ggpkd":
                self.ggpkd_artifact = build_or_load_ggpkd_artifact(
                    teacher_embeddings=teacher_cls_list,
                    cache_path=cfg.ggpkd_cache_path,
                    log_dir=cfg.ggpkd_log_dir,
                    graph_k=cfg.graph_k,
                    perplexity=cfg.perplexity,
                    truncation_tolerance=cfg.truncation_tolerance,
                    diffusion_scales=cfg.diffusion_scales,
                    knn_mode=cfg.knn_mode,
                    source_ids=self._ggpkd_source_ids(df),
                )

            # Free teacher model to save GPU memory (teacher not needed after caching)
            del self.model_teacher
            self.model_teacher = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Teacher model freed from GPU memory")

            if cfg.distill_method == "ggpkd":
                if cfg.diffusion_quota is None:
                    # Written back onto the config so the run manifest and the
                    # banner below record the concrete value this run trained on.
                    cfg.diffusion_quota = derive_diffusion_quota(
                        self.ggpkd_artifact["pool_probs"].numpy(),
                        self.ggpkd_artifact["metadata"]["diffusion_scales"],
                    )
                    print(
                        f"Derived diffusion_quota={cfg.diffusion_quota} "
                        f"(coverage tau={ROW_COVERAGE_TAU} at the median anchor)"
                    )
                self.ggpkd_sampler = GGPKDCandidateSampler(
                    artifact=self.ggpkd_artifact,
                    diffusion_quota=cfg.diffusion_quota,
                    hard_neg_k=cfg.hard_neg_k,
                    random_neg_k=cfg.random_neg_k,
                    seed=cfg.seed,
                    support_policy=cfg.support_policy,
                )
                anchor_texts = df[self.ggpkd_anchor_column].astype(str).tolist()
                # Rebuild the probe on the deduplicated anchor column. The set built
                # in setup_data was sampled from the pre-dedup frame, whose row
                # positions no longer index the teacher cache -- so pairing the two
                # there would silently report the Spearman of mismatched rows.
                probe_index = build_probe_index(len(anchor_texts), size=2048, seed=0)
                self.probe_texts = [anchor_texts[int(i)] for i in probe_index]
                self.probe_teacher = teacher_cls_list[
                    torch.from_numpy(np.asarray(probe_index)).long()
                ]
                self.train_ds = TextPairWithTeacherAndGGPKD(
                    anchor_texts=anchor_texts,
                    teacher_cls=teacher_cls_list,
                    sampler=self.ggpkd_sampler,
                    labels=df["label"].astype(int).tolist()
                    if "label" in df.columns
                    else None,
                    batch_local=cfg.batch_local,
                )
                # The collate owns the tokenized corpus: anchors and candidates are
                # drawn from the same rows, so every text is tokenized once here
                # instead of ~candidate_size times per epoch in the workers.
                self.collate_fn = GGPKDCollate(
                    self.tok_student,
                    cfg.task_type,
                    cfg.max_length,
                    corpus_texts=anchor_texts,
                    batch_local=cfg.batch_local,
                    n_scales=len(cfg.diffusion_scales),
                )
                if cfg.batch_local:
                    print(
                        "GGPKD batch-local baseline: relations among the batch "
                        f"only ({cfg.batch_size} texts), no candidate draw, no "
                        "graph support, no auxiliary rows"
                    )
                print(
                    "GGPKD candidate sampling: "
                    f"candidate_size={self.ggpkd_sampler.candidate_size} "
                    f"(diffusion={self.ggpkd_sampler.diffusion_quota}, "
                    f"hard={self.ggpkd_sampler.hard_neg_k}, "
                    f"random={self.ggpkd_sampler.random_neg_k}), "
                    f"support_policy={self.ggpkd_sampler.support_policy}, "
                    "resample_per_epoch=True"
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
            # GGPKD and RKD define their relational support from the batch. A short
            # remainder would therefore optimize a structurally different objective;
            # RKD additionally needs at least two examples to form a relation.
            drop_last=cfg.distill_method in ("ggpkd", "rkd"),
        )

        print(f"Training samples: {len(self.train_ds)}")
        print(f"Training batches: {len(self.train_loader)}")
        print("Done setup_data")

    def _build_scheduler(self):
        cfg = self.config
        total_steps = len(self.train_loader) * cfg.epochs
        if cfg.distill_method == "rkd":
            milestones = [
                int(epoch) * len(self.train_loader) for epoch in cfg.rkd_lr_decay_epochs
            ]
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=cfg.rkd_lr_decay_gamma,
            )
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
        elif cfg.distill_method == "rkd":
            self.optimizer = optim.Adam(
                self.model_student.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
            self.scheduler = self._build_scheduler()
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

    def probe_geometry_now(self) -> dict[str, float] | None:
        """Geometry of the student's space, or None if it cannot be measured.

        Never fatal: a diagnostic that can end a training run is worse than no
        diagnostic. Failures are reported once and the run continues.
        """
        if not self.probe_texts or self.tok_student is None:
            return None
        try:
            return probe_geometry(
                self.model_student,
                self.tok_student,
                self.probe_texts,
                teacher_embeddings=self.probe_teacher,
                max_length=min(128, int(getattr(self.config, "max_length", 128))),
                seed=0,
            )
        except Exception as error:
            if not getattr(self, "_warned_geometry", False):
                self._warned_geometry = True
                print(f"Warning: geometry probe unavailable ({error})")
            return None

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

        if method == "ggpkd":
            return ggpkd_step(self, batch)

        if method == "rkd":
            return rkd_step(self, batch)

        if method == "talas":
            return talas_step(self, batch)

        # Standard distillation methods with teacher inference
        return standard_step(self, batch)

    def train_epoch(self, epoch: int):
        self.model_student.train()
        self.current_epoch = epoch

        # Redraw the candidate sets. Without this the student sees the identical
        # anchor/candidate comparisons every epoch, fits them in the first one, and
        # spends the rest of the run overfitting a frozen 32-way problem.
        if hasattr(self.train_ds, "set_epoch"):
            self.train_ds.set_epoch(epoch)

        total_loss = 0.0
        n_items = 0
        metric_totals = {}
        epoch_step_times = []
        peak_memory_mb = 0.0
        # Per-step diagnostics, buffered here and written once at the end of the epoch.
        # Epoch means alone cannot show *when* inside an epoch a curve flattened, and
        # the GGPKD objective saturated inside epoch 1 on the previous run -- five
        # points per curve is a summary, not a diagnosis. Buffering keeps this to one
        # file write per epoch rather than one per step.
        step_records: list[dict] = []

        interactive_progress = sys.stderr.isatty()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            disable=not interactive_progress,
        )
        total_steps = len(self.train_loader)
        log_interval = max(1, total_steps // 10)

        for step, batch in enumerate(pbar):
            self.current_step = step

            self.sync_all()
            t0 = time.perf_counter()

            loss, metrics = self.train_step(batch)

            self.sync_all()
            dt = time.perf_counter() - t0
            epoch_step_times.append(dt)
            self.global_step += 1
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

            postfix = {"avg_loss": f"{avg_loss:.4f}", **mem_info}
            if step >= self.warmup_steps:
                self.step_times.append(dt)
                self.ma_window.append(dt)
                avg_step = sum(self.step_times) / len(self.step_times)
                ma_step = sum(self.ma_window) / len(self.ma_window)

                postfix.update(
                    {
                        "ms/step": f"{avg_step * 1000:.1f}",
                        "it/s": f"{1.0 / ma_step:.2f}",
                    }
                )

            concise_metrics = (
                ("rel", "loss_rel", True),
                (
                    "row",
                    "loss_row_weighted",
                        getattr(self.config, "row_weight", 0.0) > 0,
                    ),
                ("grad", "grad_norm", True),
            )
            for label, key, enabled in concise_metrics:
                value = metrics.get(key)
                if enabled and isinstance(value, (int, float)):
                    postfix[label] = f"{value:.4f}"

            if interactive_progress:
                pbar.set_postfix(postfix)
            elif (step + 1) % log_interval == 0 or step + 1 == total_steps:
                details = "  ".join(f"{key}={value}" for key, value in postfix.items())
                print(
                    f"[Epoch {epoch + 1}/{self.config.epochs}] "
                    f"step {step + 1}/{total_steps}  {details}",
                    flush=True,
                )

        if len(self.step_times) > 0:
            epoch_avg = sum(self.step_times) / len(self.step_times)
            print(
                f"[Epoch {epoch + 1}] Avg step time = {epoch_avg * 1000:.2f} ms "
                f"({1.0 / epoch_avg:.2f} it/s)"
            )

        print(f"Done train_epoch {epoch + 1}")
        self.log_step_records(step_records)
        epoch_means = {
            key: value / max(1, n_items) for key, value in metric_totals.items()
        }

        # The progress bar shows the *last* step's diagnostics, and the last batch is
        # usually a short remainder -- with 13553 items and batch 16 it holds a single
        # anchor, which makes target_entropy/js_floor swing wildly for reasons that
        # have nothing to do with training. Print the example-weighted epoch means.
        if epoch_means:
            headline = [
                "loss_rel",
                "loss_amb",
                "loss_nbr",
                "loss_diff",
                "loss_row_weighted",
                "row_count",
                "row_exposed_mass",
                "row_valid_ratio",
                "js_floor",
                "loss_excess",
                "target_entropy",
                "student_entropy",
                "student_entropy_ratio",
                "student_top1",
                "target_top1",
                "candidates_per_anchor",
            ]
            shown = [k for k in headline if k in epoch_means]
            semantic_kls = ("kl_amb", "kl_nbr")
            shown += [k for k in semantic_kls if k in epoch_means]
            shown += sorted(k for k in epoch_means if k.startswith("kl_diff_r"))
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
            "encoded_texts_cum": self.encoded_texts_total,
            "encoded_tokens_cum": self.encoded_tokens_total,
            **epoch_means,
        }
        return avg_loss

    def evaluate(self, split: str = "test", final: bool = False):
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
            # Report oracle-threshold pair metrics: select the threshold directly
            # on each test task, then evaluate that same test task with it.
            thresholds = None

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
        results = add_domain_averages(
            {
                "classification": classification,
                "pair": pair,
                "sts": sts,
            }
        )
        print_evaluation_table(self.current_epoch, split, results, final=final)
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
                json.dumps(
                    {"run_id": self.run_id, **record}, default=float, sort_keys=True
                )
                + "\n"
                for record in records
            )

    def log_experiment_record(self, record: dict[str, Any]):
        if not self.config.save_dir:
            return
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, "metrics.jsonl")
        payload = {
            "run_id": self.run_id,
            "method": self.config.distill_method,
            "seed": self.config.seed,
            **record,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=float, sort_keys=True) + "\n")

    def train(self):
        cfg = self.config
        write_run_manifest(
            cfg.save_dir,
            self.run_id,
            cfg,
            artifact=getattr(self, "ggpkd_artifact", None),
        )

        if cfg.distill_method == "stella":
            stella_train(self)
            return

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

        epoch_results = None
        for epoch in range(cfg.epochs):
            self.current_epoch = epoch

            use_row = (
                cfg.distill_method == "ggpkd"
                and cfg.row_weight > 0
                and epoch + 1 >= cfg.row_start_epoch
            )
            if self.criterion is not None and hasattr(self.criterion, "use_row_loss"):
                self.criterion.use_row_loss = use_row
            if use_row:
                print(f"L_row is ENABLED for Epoch {epoch + 1}")

            avg_loss = self.train_epoch(epoch)
            # The paper protocol performs no epoch selection. Per-epoch records
            # contain training and geometry diagnostics; test benchmarks are run
            # once on the final model after the fixed training budget.
            epoch_results = None

            eval_every = int(getattr(cfg, "eval_every", 0))
            if eval_every > 0 and (epoch + 1) % eval_every == 0:
                print("\n" + "=" * 60)
                print(f"Evaluation after Epoch {epoch + 1}")
                print("=" * 60)
                try:
                    epoch_results = self.evaluate("test")
                except Exception as e:
                    print(f"Warning: Evaluation failed with error: {e}")
                    print("Continuing training...")
                print("=" * 60 + "\n")
            self.log_experiment_record(
                {
                    "train": self.last_epoch_metrics,
                    "test": epoch_results,
                }
            )
            append_epoch_record(
                cfg.save_dir,
                self.run_id,
                {
                    "epoch": epoch + 1,
                    "train": self.last_epoch_metrics,
                    "test": epoch_results,
                    "geometry": self.probe_geometry_now(),
                },
            )

            final_weights_only = bool(getattr(cfg, "final_weights_only", False))
            should_save_final_weights = final_weights_only and epoch + 1 == cfg.epochs
            should_save_checkpoint = (
                not final_weights_only and (epoch + 1) % cfg.save_every == 0
            )
            if should_save_final_weights or should_save_checkpoint:
                try:
                    if should_save_final_weights:
                        if not getattr(cfg, "weights_dir", None):
                            raise ValueError("final_weights_only requires weights_dir")
                        save_student_weights(self, epoch)
                    else:
                        save_checkpoint(self, epoch, {"loss": avg_loss})
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
            save_student_weights(self, cfg.epochs - 1)
        else:
            save_checkpoint(self, cfg.epochs - 1, {"loss": avg_loss})
        if epoch_results is not None:
            # The last epoch already ran on the test split; re-running it would
            # print the same numbers a second time.
            print(f"Final test scores are the Epoch {cfg.epochs} table above.")
        else:
            try:
                test_results = self.evaluate("test", final=True)
                self.log_experiment_record({"test": test_results})
            except Exception as e:
                print(f"Warning: Final test evaluation failed with error: {e}")
