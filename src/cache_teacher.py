import json
import os

import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoModel

from .pooling import last_token_pool, mean_pooling


def _provenance_path(cache_path: str) -> str:
    return f"{cache_path}.provenance.json"


def _provenance(teacher_model_name: str | None, pooling_method: str, normalize: bool):
    return {
        "teacher_model_name": teacher_model_name,
        "pooling_method": pooling_method,
        "normalize": bool(normalize),
    }


def check_cache_provenance(cache_path: str, expected: dict) -> None:
    """Refuse a cache that was written for a different teacher or pooling.

    The cache-hit gate is nothing but ``os.path.exists``, and
    ``validate_cached_embeddings`` checks shape, dtype and finiteness -- none of
    which distinguishes one 1024-dimensional teacher from another over the same
    corpus. That made a mismatched ``cache_path`` a silent wrong-teacher run
    rather than an error. The graph artifact already guards itself this way with
    ``teacher_fingerprint``; this is the same idea for the embedding cache.

    A cache written before this guard has no sidecar. That is warned about, not
    rejected, so existing caches stay usable.
    """
    path = _provenance_path(cache_path)
    if not os.path.exists(path):
        print(
            f"Warning: teacher cache {cache_path} predates provenance recording; "
            f"cannot verify it was built with {expected['teacher_model_name']}."
        )
        return
    with open(path) as handle:
        found = json.load(handle)
    mismatched = {
        k: (found.get(k), v) for k, v in expected.items() if found.get(k) != v
    }
    if mismatched:
        detail = ", ".join(
            f"{k}: cache has {old!r}, run wants {new!r}"
            for k, (old, new) in sorted(mismatched.items())
        )
        raise ValueError(
            f"Teacher cache {cache_path} was built for a different configuration "
            f"({detail}). Point --cache_path elsewhere or delete the stale cache."
        )


def _length_sorted_batches(
    lengths: list[int], max_tokens: int, max_rows: int
) -> list[list[int]]:
    """Group example indices by length, under a padded-token budget.

    Two things at once. Sorting means a batch pads to its own longest member
    rather than to the longest in an arbitrary slice of the corpus -- 2.4x fewer
    padded tokens at this corpus's length distribution. Budgeting by *padded
    tokens* rather than by row count then keeps memory bounded while letting the
    short batches, which are most of them, run far wider than a fixed row count
    would allow.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for index in order:
        candidate_width = max(width, lengths[index])
        if current and (
            candidate_width * (len(current) + 1) > max_tokens
            or len(current) >= max_rows
        ):
            batches.append(current)
            current, width = [index], lengths[index]
        else:
            current.append(index)
            width = candidate_width
    if current:
        batches.append(current)
    return batches


def cache_teacher_embeddings(
    model_teacher: AutoModel,
    texts: list[str],
    tokenizer,
    device: torch.device,
    pooling_method: str = "last_token",
    cache_path: str | None = None,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = True,
    normalize: bool = False,
    teacher_model_name: str | None = None,
    max_length: int = 256,
    max_tokens_per_batch: int = 8192,
    max_rows_per_batch: int = 256,
) -> torch.Tensor:
    """One pooled teacher vector per text, in the order the texts were given.

    This used to consume a `DataLoader` over `DualTokenizerCollate`, which
    tokenized four things per row -- student and teacher, first and second text --
    when only the teacher's first text is ever read here. On the ggpkd corpus
    the second text is a copy of the first, so three of the four were pure waste,
    and none of the student ones were consumed at all.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached teacher embeddings from: {cache_path}")
        check_cache_provenance(
            cache_path, _provenance(teacher_model_name, pooling_method, normalize)
        )
        cached_data = torch.load(cache_path, map_location="cpu")
        print(f"Done loading cached embeddings: {cached_data.shape}")
        return cached_data

    print("Pre-computing teacher embeddings...")
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)

    texts = [str(text) for text in texts]
    encoded = tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]
    lengths = [len(row) for row in encoded]
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    batches = _length_sorted_batches(lengths, max_tokens_per_batch, max_rows_per_batch)

    teacher_cls_all: torch.Tensor | None = None
    with torch.inference_mode():
        for index_group in tqdm(batches, desc="Caching teacher embeddings"):
            width = max(lengths[i] for i in index_group)
            input_ids = torch.full((len(index_group), width), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(index_group), width), dtype=torch.long)
            for row, i in enumerate(index_group):
                input_ids[row, : lengths[i]] = torch.tensor(encoded[i])
                attention_mask[row, : lengths[i]] = 1
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            with autocast(
                "cuda",
                enabled=use_amp and torch.cuda.is_available(),
            ):
                t_out1 = model_teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=False,
                )

                T_last1 = t_out1.last_hidden_state  # [B, L, d_t]
                if pooling_method == "last_token":
                    # Reads the last *unmasked* position, so the batch's padded
                    # width does not enter the result. (With sorted batches every
                    # row is often the same length, which sends it down its
                    # left-padding branch instead; on an unpadded batch the two
                    # branches pick the same position.)
                    T_cls1 = last_token_pool(T_last1, attention_mask)
                elif pooling_method == "mean":
                    T_cls1 = mean_pooling(T_last1, attention_mask)
                elif pooling_method == "cls":
                    T_cls1 = T_last1[:, 0, :]
                else:
                    raise ValueError(f"Unknown pooling method: {pooling_method}")

                if normalize:
                    T_cls1 = F.normalize(T_cls1, p=2, dim=-1)
                T_cls1 = T_cls1.to(dtype)

            # Scatter back to corpus order. Unlike the evaluation path, order is
            # load-bearing here: row i of this tensor is the teacher embedding of
            # corpus item i, and the graph, the candidate pools and the criterion
            # all index it that way.
            block = T_cls1.cpu()
            if teacher_cls_all is None:
                teacher_cls_all = torch.empty(
                    (len(texts), block.shape[-1]), dtype=block.dtype
                )
            teacher_cls_all[torch.tensor(index_group, dtype=torch.long)] = block

    if teacher_cls_all is None:
        raise ValueError("No texts to cache teacher embeddings for")
    if cache_path:
        os.makedirs(
            os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
            exist_ok=True,
        )
        torch.save(teacher_cls_all, cache_path)
        with open(_provenance_path(cache_path), "w") as handle:
            json.dump(
                _provenance(teacher_model_name, pooling_method, normalize),
                handle,
                indent=1,
                sort_keys=True,
            )
        print(f"Saved cached teacher embeddings to: {cache_path}")

    print(f"Done caching teacher embeddings: {teacher_cls_all.shape}")
    return teacher_cls_all


def load_cached_embeddings(cache_path: str) -> torch.Tensor:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    print(f"Loading cached embeddings from: {cache_path}")
    embeddings = torch.load(cache_path, map_location="cpu")
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings


def validate_cached_embeddings(
    embeddings: torch.Tensor,
    expected_rows: int,
    *,
    cache_path: str = "<memory>",
    require_single_layer: bool = False,
) -> torch.Tensor:
    """Reject partial, stale, or numerically invalid teacher-cache tensors."""
    if not torch.is_tensor(embeddings):
        raise TypeError(
            f"Teacher cache {cache_path} must contain a tensor, got "
            f"{type(embeddings).__name__}"
        )
    allowed_dims = (2,) if require_single_layer else (2, 3)
    if embeddings.ndim not in allowed_dims:
        raise ValueError(
            f"Teacher cache {cache_path} has shape {tuple(embeddings.shape)}; "
            f"expected {'[N, D]' if require_single_layer else '[N, D] or [N, L, D]'}"
        )
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Teacher cache {cache_path} row mismatch: cache has "
            f"{embeddings.shape[0]} rows, data has {expected_rows}"
        )
    if embeddings.shape[-1] <= 0:
        raise ValueError(f"Teacher cache {cache_path} has an empty embedding dimension")
    if not embeddings.is_floating_point():
        raise TypeError(
            f"Teacher cache {cache_path} must be floating point, got {embeddings.dtype}"
        )
    if not bool(torch.isfinite(embeddings).all().item()):
        raise ValueError(f"Teacher cache {cache_path} contains NaN or Inf")
    return embeddings


def clear_cache_and_free_memory():
    import gc

    torch.cuda.empty_cache()
    gc.collect()
    print("Done clearing GPU cache and freeing memory")
