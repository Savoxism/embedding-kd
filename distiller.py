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
from torch import optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler
from transformers import __version__ as transformers_version

from src.cache_teacher import (
    cache_teacher_embeddings,
    check_cache_provenance,
    load_cached_embeddings,
    validate_cached_embeddings,
)
from src.data_utils import DualTokenizerCollate, TextPairRaw
from src.data_utils.dataset_cache import (
    DualTokenizerCollateWithTeacher,
    TextPairWithTeacher,
)
from src.distill.benchmarks import add_domain_averages, print_evaluation_table
from src.distill.checkpointing import save_checkpoint, save_student_weights
from src.distill.geometry import build_probe_set, probe_geometry
from src.distill.numerics import (
    assert_module_parameters_finite,
)
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
from src.loss import info_nce
from src.methods import get_method


class KnowledgeDistiller:
    def __init__(self, config):
        self.config = config
        # Everything method-specific is reached through this: the flags the
        # shared pipeline reads, and the hooks that replace a piece of it.
        self.method = get_method(config.distill_method)
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

        self.criterion = (
            None
            if self.method.build_criterion is None
            else self.method.build_criterion(self, config)
        )

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
        self._teacher_cache_ready = (
            self.method.uses_teacher_cache and Path(cfg.cache_path).is_file()
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
        if self.method.build_student is not None:
            self.model_student = self.method.build_student(self)
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
            if self.method.needs_attentions:
                # The student's attention maps are read too, not just the
                # teacher's. Only the teacher was pinned to eager before, so a
                # student on a fused kernel returned an empty attention tuple and
                # the criterion indexed off the end of it.
                student_kwargs["attn_implementation"] = "eager"
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

            if self.method.needs_attentions:
                # The fused attention kernels return no attention maps.
                teacher_kwargs["attn_implementation"] = "eager"
                print(
                    f"Using eager attention for {self.method.name} "
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

        # A method may drop rows (GGPKD deduplicates its anchors). The surviving
        # positions are kept because a teacher cache written before the drop
        # still lines up with the original frame and can be sliced instead of
        # recomputed.
        self.keep_positions = None
        if self.method.prepare_frame is not None:
            df, self.keep_positions = self.method.prepare_frame(self, df)

        self.task_head = None
        if self.method.build_task_head is not None:
            self.task_head = self.method.build_task_head(self, df)

        if self.method.uses_teacher_cache:
            cache_path = Path(cfg.cache_path)

            # Check if cache exists
            if cache_path.exists():
                print(f"Loading cached teacher embeddings from: {cache_path}")
                # The guard lives in cache_teacher.py but was unreachable: that
                # module only checks provenance on its own cache-hit branch, and
                # this branch means the cache already existed, so that branch
                # never ran. Nothing else distinguishes two teachers of the same
                # width over the same corpus -- validate_cached_embeddings checks
                # shape, dtype and finiteness -- so a mismatched cache_path was a
                # silent wrong-teacher run.
                check_cache_provenance(
                    str(cache_path),
                    {
                        "teacher_model_name": cfg.teacher_model_name,
                        "pooling_method": cfg.pooling_method,
                        "normalize": bool(cfg.normalize_cache),
                    },
                )
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
            keep_positions = self.keep_positions
            if (
                keep_positions is not None
                and len(teacher_cls_list) > len(df)
                and len(keep_positions) == len(df)
                and int(keep_positions.max(initial=-1)) < len(teacher_cls_list)
            ):
                print(
                    f"Slicing pre-drop teacher cache: {len(teacher_cls_list)} -> {len(df)} rows"
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

            # Free teacher model to save GPU memory (teacher not needed after caching)
            del self.model_teacher
            self.model_teacher = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Teacher model freed from GPU memory")

            if self.method.build_data is not None:
                self.train_ds, self.collate_fn = self.method.build_data(
                    self, df, teacher_cls_list
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
        # A teacher-informed batch sampler is the batch-intervention arm. It owns
        # the grouping, so it replaces `shuffle` and `batch_size` outright --
        # DataLoader rejects passing either alongside a batch_sampler -- and it is
        # reseeded per epoch, which makes it a second reason the workers cannot
        # persist across epochs.
        self.batch_sampler = self.build_batch_sampler()
        loader_kwargs = {}
        if self.batch_sampler is not None:
            loader_kwargs["batch_sampler"] = self.batch_sampler
            resamples_per_epoch = True
        else:
            loader_kwargs["batch_size"] = cfg.batch_size
            loader_kwargs["shuffle"] = True
            # A batch-relational method defines its support from the batch, so a
            # short remainder would optimize a structurally different objective;
            # RKD additionally needs at least two examples to form a relation.
            loader_kwargs["drop_last"] = self.method.batch_relational
        self.train_loader = DataLoader(
            self.train_ds,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0 and not resamples_per_epoch,
            **loader_kwargs,
        )

        print(f"Training samples: {len(self.train_ds)}")
        print(f"Training batches: {len(self.train_loader)}")
        print("Done setup_data")

    def build_batch_sampler(self):
        """The teacher-informed batch sampler, or None for the i.i.d. default.

        Reads the teacher's own retrieval order out of the GGPKD graph artifact,
        so the grouping is defined by the same neighbourhoods the method distils
        from and no second notion of "close" enters the study. Without an
        artifact -- any method that is not GGPKD -- a teacher-informed request is
        refused rather than silently downgraded to random: the arm label would
        then name an intervention that did not happen.
        """
        mode = getattr(self.config, "batch_sampler", "random")
        if mode == "random":
            return None

        from src.data_utils.batch_samplers import TeacherBatchSampler

        artifact = getattr(self, "ggpkd_artifact", None)
        if artifact is None:
            # The batch-intervention study needs this on methods that build no
            # graph of their own -- the pointwise arm is the whole measurement
            # floor, and it has to be grouped the same three ways as the others.
            # Reading a prebuilt artifact off disk is what makes that possible;
            # the runner builds one per corpus before any arm starts, so all arms
            # of one experiment are grouped by the same neighbourhoods.
            path = getattr(self.config, "ggpkd_cache_path", None)
            if not path or not os.path.exists(path):
                raise ValueError(
                    f"batch_sampler={mode!r} composes batches from teacher "
                    "neighbourhoods and needs a graph artifact: "
                    f"{self.config.distill_method!r} builds none and "
                    f"ggpkd_cache_path={path!r} does not exist"
                )
            print(f"Batch composition reads the prebuilt graph: {path}")
            artifact = torch.load(path, map_location="cpu", weights_only=False)
        neighbors = artifact["transition_neighbors"].numpy()
        sampler = TeacherBatchSampler(
            mode=mode,
            neighbors=neighbors,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            drop_last=self.method.batch_relational,
        )
        print(
            f"Batch composition: {mode} "
            f"({len(sampler)} batches of {self.config.batch_size}, "
            "regrouped every epoch)"
        )
        return sampler

    def build_scheduler(self):
        """The method's schedule over the current optimizer.

        Public because a criterion that adds a param group -- or replaces the
        optimizer, as TALAS does -- has to rebuild the schedule in the same
        breath, and those builders live in `src/methods/`.
        """
        if self.method.build_scheduler is not None:
            return self.method.build_scheduler(self)
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

        parameters = list(self.model_student.parameters())
        if self.task_head is not None:
            parameters.extend(self.task_head.parameters())
        # A criterion with parameters of its own is not built yet; it either adds
        # a param group here or replaces the optimizer outright, and rebuilds the
        # schedule in the same breath (see src/methods/support.py).
        self.optimizer = (
            optim.AdamW(parameters, lr=cfg.learning_rate)
            if self.method.build_optimizer is None
            else self.method.build_optimizer(self, parameters)
        )
        self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
        self.scheduler = self.build_scheduler()

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

    def compute_task_loss(
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
        return self.method.step(self, batch)

    def train_epoch(self, epoch: int):
        self.model_student.train()
        self.current_epoch = epoch

        # Redraw the candidate sets. Without this the student sees the identical
        # anchor/candidate comparisons every epoch, fits them in the first one, and
        # spends the rest of the run overfitting a frozen 32-way problem.
        if hasattr(self.train_ds, "set_epoch"):
            self.train_ds.set_epoch(epoch)
        # The teacher-informed grouping is redrawn for the same reason: a fixed
        # partition would let the student see one frozen set of batch-mates.
        if getattr(self, "batch_sampler", None) is not None:
            self.batch_sampler.set_epoch(epoch)

        total_loss = 0.0
        n_items = 0
        avg_loss = 0.0
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
        device_count = torch.cuda.device_count()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            disable=not interactive_progress,
        )
        total_steps = len(self.train_loader)
        log_interval = max(1, total_steps // 10)

        # Timed with CUDA events rather than by draining the queue on both sides
        # of the step. `sync_all()` before and after every step forced the host to
        # wait for the whole device queue twice per step, so the CPU could never
        # run ahead of the GPU -- a measurement that changed the thing it measured.
        # Events are recorded on the stream and read below, after `loss.item()`
        # has already drained it, so the timing costs no synchronization of its own.
        use_events = torch.cuda.is_available()
        start_event = torch.cuda.Event(enable_timing=True) if use_events else None
        end_event = torch.cuda.Event(enable_timing=True) if use_events else None

        for step, batch in enumerate(pbar):
            self.current_step = step

            t0 = time.perf_counter()
            if use_events:
                start_event.record()

            loss, metrics = self.train_step(batch)

            if use_events:
                end_event.record()
            self.global_step += 1
            bs = batch["input_ids1_stu"].size(0)
            # Drains the stream, so both events have completed by the time the
            # elapsed time is read.
            loss_value = loss.item()
            if use_events:
                end_event.synchronize()
                dt = start_event.elapsed_time(end_event) / 200.0
            else:
                dt = time.perf_counter() - t0
            epoch_step_times.append(dt)
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

            # Accumulated on every step -- these feed the end-of-epoch step-time
            # summary, so they must not be gated on whether this step prints.
            if step >= self.warmup_steps:
                self.step_times.append(dt)
                self.ma_window.append(dt)
            peak_memory_mb = max(
                peak_memory_mb,
                max(
                    (
                        torch.cuda.memory_allocated(dev_id) / 1024**2
                        for dev_id in range(device_count)
                    ),
                    default=0.0,
                ),
            )

            # Everything below is display only. A non-interactive run reaches it
            # about ten times an epoch, so formatting it on every step was work
            # thrown away.
            reporting = (
                interactive_progress
                or (step + 1) % log_interval == 0
                or step + 1 == total_steps
            )
            if not reporting:
                continue

            postfix = {"avg_loss": f"{avg_loss:.4f}"}
            for dev_id in range(device_count):
                mem_alloc = torch.cuda.memory_allocated(dev_id) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(dev_id) / 1024**2
                postfix[f"gpu{dev_id}"] = f"{mem_alloc:.0f}/{mem_reserved:.0f}MB"
            if self.step_times:
                avg_step = sum(self.step_times) / len(self.step_times)
                ma_step = sum(self.ma_window) / len(self.ma_window)
                postfix.update(
                    {
                        "ms/step": f"{avg_step * 200:.1f}",
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
                f"[Epoch {epoch + 1}] Avg step time = {epoch_avg * 200:.2f} ms "
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
        # The selected thresholds are returned but not kept: both splits pass
        # `thresholds=None`, so each split selects on itself and there is no
        # threshold to carry across. The distiller used to stash the validation
        # ones on `self`, where nothing ever read them.
        pair, _ = eval_pair_task(
            self.model_student,
            pair_tasks,
            self.tok_student,
            thresholds=thresholds,
        )
        sts = eval_sts_task(self.model_student, sts_tasks, self.tok_student)
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

        if self.method.train_loop is not None:
            self.method.train_loop(self)
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

            if self.method.on_epoch_start is not None:
                self.method.on_epoch_start(self, epoch)

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
