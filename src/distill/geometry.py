"""Embedding-space diagnostics for the student.

The training loss says how well the student matches the teacher's targets on the
columns it was shown. It says nothing about the *space* the student produces, and
that space is the deliverable: STS reads cosine gaps, and pair classification
thresholds a single global cosine. Those two are exactly where a run can regress
while every training curve looks healthy -- the per-row ambient arm cost 0.30 on
out-of-domain at unchanged in-domain, and nothing in the loss saw it coming.

Everything here runs on one fixed probe set, once per epoch, under `no_grad`.
Fixing the probe set matters: the numbers are only comparable across epochs and
across runs if the texts do not move.

References for the two that are not obvious:
    alignment / uniformity -- Wang and Isola, "Understanding Contrastive
    Representation Learning through Alignment and Uniformity on the Hypersphere"
    (ICML 2020). Alignment is E||f(x) - f(x+)||^2 over positive pairs; uniformity
    is log E exp(-2||f(x) - f(y)||^2) over random pairs. Lower alignment means
    positives land together; lower (more negative) uniformity means the sphere is
    covered rather than collapsed. They trade off, which is why both are reported.
"""

from typing import Any

import torch
import torch.nn.functional as F

# Temperature of the dense teacher distribution p^T that weights the distortion
# below. Deliberately a constant of the *probe* rather than a copy of the run's
# direct_temp: E_hat is compared across ablation arms, and a quantity whose
# weighting moves with the arm being measured cannot order those arms. 0.05 is
# the fixed-bandwidth graph temperature, so the weighting is the same shape as
# the teacher rows the method distils from.
PROBE_DISTORTION_TEMP = 0.05


@torch.no_grad()
def _encode(model, tokenizer, texts: list[str], max_length: int, batch_size: int):
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        result = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        # Same convention the evaluation path uses, so the geometry describes the
        # vectors the benchmarks actually score.
        if isinstance(result, dict) and "pooled" in result:
            pooled = result["pooled"]
        else:
            hidden = (
                result.last_hidden_state
                if hasattr(result, "last_hidden_state")
                else result["last_hidden_state"]
            )
            pooled = hidden[:, 0, :]
        out.append(pooled.float().cpu())
    return torch.cat(out, dim=0)


def _effective_rank(embeddings: torch.Tensor) -> float:
    """exp(H) of the normalized singular-value spectrum (Roy and Vetterli, 2007).

    A d-dimensional encoder whose outputs live on a 3-dimensional subspace scores
    ~3, not d. Reported alongside anisotropy because the two fail differently: a
    space can be full-rank and still have every pair at cosine 0.9.
    """
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.double())
    total = singular.sum()
    if total <= 0:
        return 0.0
    p = singular / total
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


@torch.no_grad()
def probe_geometry(
    model,
    tokenizer,
    texts: list[str],
    *,
    pairs: list[tuple[str, str]] | None = None,
    teacher_embeddings: torch.Tensor | None = None,
    max_length: int = 128,
    batch_size: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Geometry of the student's space on a fixed probe set.

    Args:
        texts: the probe set; must be identical across epochs and runs.
        pairs: optional positive pairs, for `alignment`. Without them the
            alignment term is skipped rather than faked from unrelated texts.
        teacher_embeddings: cached teacher vectors for the same `texts`, in the
            same order. Enables `teacher_student_spearman`: how much of the
            teacher's ordering of pairwise similarity the student reproduces, in
            one number and independent of any benchmark.
    """
    was_training = model.training
    model.eval()
    try:
        embeddings = _encode(model, tokenizer, texts, max_length, batch_size)
        normalized = F.normalize(embeddings, p=2, dim=-1)

        generator = torch.Generator().manual_seed(seed)
        n = normalized.size(0)
        # A fixed random pair sample rather than the full n^2 matrix: at a few
        # thousand probes the estimate is tight and the cost stays flat.
        n_pairs = min(20000, n * (n - 1) // 2)
        left = torch.randint(0, n, (n_pairs,), generator=generator)
        right = torch.randint(0, n, (n_pairs,), generator=generator)
        keep = left != right
        left, right = left[keep], right[keep]
        cosines = (normalized[left] * normalized[right]).sum(dim=-1)

        stats: dict[str, float] = {
            # Mean cosine of unrelated texts. At 0 the space is isotropic; near 1
            # everything points the same way and no threshold can separate
            # anything.
            "anisotropy": float(cosines.mean()),
            "cos_std": float(cosines.std()),
            "cos_p50": float(cosines.quantile(0.50)),
            "cos_p90": float(cosines.quantile(0.90)),
            "cos_p99": float(cosines.quantile(0.99)),
            "effective_rank": _effective_rank(embeddings),
            "embedding_norm_mean": float(embeddings.norm(dim=-1).mean()),
            "probe_size": float(n),
            # Uniformity on the hypersphere (Wang and Isola 2020), lower is better.
            "uniformity": float(
                torch.log(
                    torch.exp(-2.0 * (normalized[left] - normalized[right]).pow(2).sum(-1))
                    .mean()
                    .clamp_min(1e-30)
                )
            ),
        }

        if pairs:
            flat = [text for pair in pairs for text in pair]
            paired = F.normalize(
                _encode(model, tokenizer, flat, max_length, batch_size), p=2, dim=-1
            )
            first, second = paired[0::2], paired[1::2]
            stats["alignment"] = float((first - second).pow(2).sum(dim=-1).mean())
            stats["positive_cos_mean"] = float((first * second).sum(dim=-1).mean())
            # The gap that STS actually reads: positives above the background.
            stats["separation"] = stats["positive_cos_mean"] - stats["anisotropy"]

        if teacher_embeddings is not None and len(teacher_embeddings) == n:
            teacher = F.normalize(teacher_embeddings.float().cpu(), p=2, dim=-1)
            teacher_cos = (teacher[left] * teacher[right]).sum(dim=-1)
            stats["teacher_student_spearman"] = _spearman(teacher_cos, cosines)
            stats["teacher_anisotropy"] = float(teacher_cos.mean())
            stats.update(_distortion(teacher, normalized))
        return stats
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def _distortion(teacher: torch.Tensor, student: torch.Tensor) -> dict[str, float]:
    """Teacher-weighted global distortion E_hat, and its unweighted companion.

        E_hat = mean_i sum_j p^T_i(j) (cos_T(i,j) - cos_S(i,j))^2,
        p^T_i  = softmax_j( cos_T(i,j) / PROBE_DISTORTION_TEMP ),  j != i

    This is the middle link of the paper's causal chain -- support coverage ->
    geometry preservation -> downstream -- and the only one of the three that a
    benchmark score cannot stand in for. The weighting is what makes it the right
    quantity: an unweighted cosine error is dominated by the ~99% of pairs the
    teacher considers unrelated and where being wrong costs nothing, so it moves
    with anisotropy rather than with preserved geometry. Weighting by the
    teacher's own distribution asks the question the method is trying to answer --
    is the student right *where the teacher has an opinion*.
    `cosine_rmse` is reported beside it precisely so the two can be seen to
    disagree; if a policy improves the unweighted number while E_hat worsens, it
    flattened the space rather than preserving it.

    Both are computed densely over the probe. At 2k probes that is a 2k x 2k
    float32 block, ~16 MB -- cheaper than the pair sampling it sits next to.
    """
    n = teacher.size(0)
    if n < 2:
        return {}
    teacher_cos = teacher @ teacher.t()
    student_cos = student @ student.t()
    self_pairs = torch.eye(n, dtype=torch.bool, device=teacher_cos.device)

    weights = F.softmax(
        (teacher_cos / PROBE_DISTORTION_TEMP).masked_fill(self_pairs, float("-inf")),
        dim=-1,
    )
    squared_gap = (teacher_cos - student_cos).pow(2).masked_fill(self_pairs, 0.0)
    off_diagonal = squared_gap.masked_select(~self_pairs)
    return {
        "teacher_weighted_distortion": float((weights * squared_gap).sum(-1).mean()),
        "cosine_rmse": float(off_diagonal.mean().sqrt()),
    }


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation without a scipy round-trip (no ties expected on cosines)."""
    rank_a = a.argsort().argsort().double()
    rank_b = b.argsort().argsort().double()
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denominator = rank_a.norm() * rank_b.norm()
    if denominator <= 0:
        return 0.0
    return float((rank_a * rank_b).sum() / denominator)


def build_probe_index(n_rows: int, size: int = 2048, seed: int = 0):
    """Row positions of the probe set, so callers can index aligned arrays.

    `build_probe_set` returns strings, which is all the student needs. Anything
    that has to line a *second* array up with the probe -- cached teacher vectors,
    for `teacher_student_spearman` -- needs the positions, and re-deriving them at
    the call site is how two "identical" probe sets drift apart.
    """
    import numpy as np

    if n_rows <= size:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    index = rng.choice(n_rows, size=size, replace=False)
    index.sort()
    return index


def build_probe_set(
    frame: Any, text_column: str, size: int = 2048, seed: int = 0
) -> list[str]:
    """A deterministic sample of the training corpus to probe with.

    Deterministic by construction: the same corpus and seed give the same texts,
    which is what makes the numbers comparable between runs.
    """
    texts = frame[text_column].astype(str).tolist()
    index = build_probe_index(len(texts), size=size, seed=seed)
    return [texts[int(i)] for i in index]
