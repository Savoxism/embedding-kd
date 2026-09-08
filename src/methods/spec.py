"""What a distillation method *is*, as data rather than as scattered branches.

Before this, a method's identity lived as its own name repeated in if-chains and
membership tuples across `distiller.py`, `main.py` and `criterion_factory.py`:
19 reads of `config.distill_method` in the distiller alone, and the same property
spelled twice in two different shapes -- `cached_teacher_methods = {...}` in
`setup_models` and `distill_method in ("talas", "ggpkd", "rkd")` in `setup_data`.
Adding a method meant finding all 13 sites; forgetting one failed silently.

A `MethodSpec` states the same facts once. The flags answer questions the shared
pipeline asks ("does this method read a teacher cache?"); the hooks are the
points where a method needs its own code, and `None` means "use the shared
default", so a spec only carries what actually differs.

Every hook takes the distiller as `ctx` -- the same convention the step functions
already use -- and the docstring of each field names what it may touch.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MethodSpec:
    """One distillation method's declaration.

    Flags are read by the shared pipeline; hooks replace a piece of it.
    """

    name: str
    """The `--method` value and `config.distill_method`."""

    config_cls: type
    """Config class instantiated for this method."""

    step: Callable[[Any, dict], tuple]
    """`step(ctx, batch) -> (loss, metrics)`. One training step."""

    uses_teacher_cache: bool = False
    """Teacher embeddings are precomputed to `config.cache_path`.

    Drives three things that used to be spelled separately: skipping the teacher
    tokenizer/model load when the cache is on disk, taking the cached-teacher
    data path, and accepting `--prepare_cache_only`.
    """

    batch_relational: bool = False
    """The objective relates the examples inside a batch to each other.

    Such a method cannot take a short final batch -- the remainder optimizes a
    structurally different objective -- so the loader drops it.
    """

    build_student: Callable | None = None
    """`(ctx) -> nn.Module`. Non-`AutoModel` students only."""

    prepare_frame: Callable | None = None
    """`(ctx, df) -> (df, keep_positions | None)`, before anything reads the corpus.

    `keep_positions` is the surviving row positions when the method drops rows;
    the shared path uses it to slice a teacher cache that predates the drop.
    """

    build_task_head: Callable | None = None
    """`(ctx, df) -> nn.Module | None`. A supervised head trained with the student."""

    build_data: Callable | None = None
    """`(ctx, df, teacher_cls) -> (dataset, collate_fn)`.

    `None` takes the shared path: cached-teacher or teacher-in-the-loop according
    to `uses_teacher_cache`.
    """

    needs_attentions: bool = False
    """The objective reads attention maps, from both encoders.

    Also forces the teacher onto an eager attention implementation, since the
    fused kernels return no attention maps.
    """

    student_returns_pooled: bool = False
    """The student returns a dict with its own pooled vector, not `last_hidden_state`."""

    supervised_task_loss: bool = False
    """The task term is the labelled loss (`ctx.compute_task_loss`), not in-batch InfoNCE."""

    kd_loss: Callable | None = None
    """`(ctx, tensors) -> (loss, metrics)`. The KD term of the shared step.

    Required by every method whose `step` is `src.distill.steps.standard`; the
    shared step computes both encoders' outputs and the task loss, and this is
    the only part that differs between those methods.
    """

    build_criterion: Callable | None = None
    """`(ctx, config) -> criterion | None`, after the optimizer exists.

    May add a param group or replace the optimizer outright; see
    `src.methods.support.attach_parameters`.
    """

    build_optimizer: Callable | None = None
    """`(ctx, parameters) -> Optimizer`. `None` means AdamW at `config.learning_rate`."""

    build_scheduler: Callable | None = None
    """`(ctx) -> LRScheduler`. `None` means warmup + cosine-with-min-lr."""

    on_epoch_start: Callable | None = None
    """`(ctx, epoch) -> None`, before each epoch of the shared training loop."""

    train_loop: Callable | None = None
    """`(ctx) -> None`. Replaces the shared epoch loop entirely."""
