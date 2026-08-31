"""The evaluation batching must be a speed change, not a numerical one.

Length-sorting reorders examples and widens the batch, so the thing worth pinning
is that neither moves a metric: the reference here is the old behaviour --
original file order, per-batch padding, batch size 64 -- rebuilt inline so the
comparison survives the deletion of the code that used to do it.
"""

from types import SimpleNamespace

import torch
from torch import nn

import src.evaluation.evaluation_automodel as ev

STS_PATH = "data/val_set/stsb_validation.csv"
PAIR_PATH = "data/val_set/wic_validation.csv"


class _CharTokenizer:
    """Character-level stand-in; no download, deterministic, fast."""

    pad_token_id = 0
    name_or_path = "char-test"
    vocab_size = 64

    def __call__(
        self, texts, truncation=True, padding=False, max_length=128, return_tensors=None
    ):
        ids = [[(ord(c) % 60) + 2 for c in text[:max_length]] or [2] for text in texts]
        if not padding and return_tensors is None:
            return {"input_ids": ids}
        width = max(len(row) for row in ids)
        return {
            "input_ids": torch.tensor([r + [0] * (width - len(r)) for r in ids]),
            "attention_mask": torch.tensor(
                [[1] * len(r) + [0] * (width - len(r)) for r in ids]
            ),
        }


class _MaskedEncoder(nn.Module):
    """Honours the attention mask, so padding width cannot leak into the output."""

    def __init__(self, dim: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(64, dim)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


def _reference_batches(path, columns, label_column, label_dtype, tokenizer,
                       max_len, batch_size=64):
    """The previous behaviour: file order, per-batch padding, batch size 64."""
    import pandas as pd

    frame = pd.read_csv(ev.BASE_DIR / path)
    labels = torch.tensor(frame[label_column].tolist(), dtype=label_dtype)
    batches = []
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        batch = {"labels": labels[start:stop]}
        for position, column in enumerate(columns):
            encoded = tokenizer(
                frame[column].astype(str).tolist()[start:stop],
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            batch[f"input_ids{position + 1}"] = encoded["input_ids"]
            batch[f"attention_mask{position + 1}"] = encoded["attention_mask"]
        batches.append(batch)
    return batches


def test_sorted_batching_leaves_sts_spearman_unchanged():
    torch.manual_seed(0)
    model, tokenizer = _MaskedEncoder(), _CharTokenizer()
    ev.clear_eval_cache()

    reference = ev.eval_sts(
        model,
        _reference_batches(
            STS_PATH, ("sentence1", "sentence2"), "score", torch.float, tokenizer, 128
        ),
    )
    sorted_result = ev.eval_sts(
        model,
        ev._sorted_batches(
            STS_PATH, ("sentence1", "sentence2"), "score", torch.float, tokenizer, 128
        ),
    )
    assert reference == sorted_result


def test_sorted_batching_leaves_pair_metrics_unchanged():
    torch.manual_seed(0)
    model, tokenizer = _MaskedEncoder(), _CharTokenizer()
    ev.clear_eval_cache()

    reference = ev.eval_pair(
        model,
        _reference_batches(
            PAIR_PATH, ("sentence1", "sentence2"), "label", torch.float, tokenizer, 128
        ),
    )
    sorted_result = ev.eval_pair(
        model,
        ev._sorted_batches(
            PAIR_PATH, ("sentence1", "sentence2"), "label", torch.float, tokenizer, 128
        ),
    )
    for key in ("average_precision", "accuracy", "f1", "precision", "recall"):
        assert reference[key] == sorted_result[key], key


def test_sorting_cuts_padding_and_keeps_every_example():
    tokenizer = _CharTokenizer()
    ev.clear_eval_cache()

    reference = _reference_batches(
        STS_PATH, ("sentence1", "sentence2"), "score", torch.float, tokenizer, 128
    )
    batches = ev._sorted_batches(
        STS_PATH, ("sentence1", "sentence2"), "score", torch.float, tokenizer, 128
    )

    def slots(rows):
        return sum(b["input_ids1"].numel() + b["input_ids2"].numel() for b in rows)

    def rows(collection):
        return sum(b["labels"].numel() for b in collection)

    assert rows(batches) == rows(reference), "no example may be dropped"
    assert slots(batches) < slots(reference), "sorting must reduce padded work"
    # The labels are a permutation of the originals, not a different set.
    assert torch.equal(
        torch.cat([b["labels"] for b in batches]).sort().values,
        torch.cat([b["labels"] for b in reference]).sort().values,
    )


def test_batches_are_cached_and_tokenized_once():
    tokenizer = _CharTokenizer()
    ev.clear_eval_cache()
    calls = {"n": 0}
    original = tokenizer.__call__

    class _Counting(_CharTokenizer):
        def __call__(self, *args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

    counting = _Counting()
    first = ev._sorted_batches(
        STS_PATH, ("sentence1", "sentence2"), "score", torch.float, counting, 128
    )
    after_first = calls["n"]
    second = ev._sorted_batches(
        STS_PATH, ("sentence1", "sentence2"), "score", torch.float, counting, 128
    )

    assert first is second, "the second epoch must reuse the cached batches"
    assert calls["n"] == after_first, "no re-tokenization on a cache hit"
    # One batched call per text column, not one per mini-batch.
    assert after_first == 2


def test_evaluating_restores_the_previous_mode():
    model = _MaskedEncoder()

    model.train()
    with ev._evaluating(model):
        assert not model.training
    assert model.training, "a training run must go back to train mode"

    model.eval()
    with ev._evaluating(model):
        assert not model.training
    assert not model.training, "the final test eval must stay in eval mode"
