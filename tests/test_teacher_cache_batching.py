"""The teacher cache must stay in corpus order, whatever order it computes in.

Row `i` of the cached tensor is the teacher embedding of corpus item `i`: the
kNN graph, the diffusion pools, the candidate sampler and the criterion's teacher
bank all index it that way. Length-sorted batching computes those rows out of
order, so the scatter back is the load-bearing part and these tests pin it.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.cache_teacher import (
    _length_sorted_batches,
    cache_teacher_embeddings,
    check_cache_provenance,
)


class _CharTokenizer:
    pad_token_id = 0

    def __call__(self, texts, truncation=True, max_length=128, **kwargs):
        return {
            "input_ids": [
                [(ord(c) % 60) + 2 for c in text[:max_length]] or [2] for text in texts
            ]
        }


class _IdentityTeacher(nn.Module):
    """Its pooled output is a deterministic function of the text alone.

    That is what makes an order bug visible: if a row lands in the wrong slot,
    the value no longer matches the text that produced it.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim
        self.register_buffer("_unused", torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, return_dict=True,
                output_hidden_states=False):
        # Encode the token ids into the hidden state so pooling recovers them.
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.dim)
        hidden[:, :, 0] = input_ids.float()
        hidden[:, :, 1] = torch.arange(seq).float().unsqueeze(0)
        return SimpleNamespace(last_hidden_state=hidden)


def _texts():
    # Deliberately varied lengths so sorting really reorders them.
    return [
        "a", "bb" * 9, "ccc", "d" * 30, "ee", "ffffff", "g" * 20, "hh",
        "iii" * 4, "j", "kkkk", "l" * 25,
    ]


def test_embeddings_come_back_in_corpus_order():
    texts = _texts()
    model, tokenizer = _IdentityTeacher(), _CharTokenizer()

    out = cache_teacher_embeddings(
        model_teacher=model,
        texts=texts,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        pooling_method="last_token",
        use_amp=False,
        max_tokens_per_batch=64,
        max_rows_per_batch=4,
    )

    assert out.shape[0] == len(texts)
    # last_token_pool takes the final real position, so channel 0 must hold the
    # last token id of each text and channel 1 its index -- both recoverable from
    # the text alone, and both wrong if a row landed in the wrong slot.
    encoded = tokenizer(texts)["input_ids"]
    for position, ids in enumerate(encoded):
        assert out[position, 0] == pytest.approx(float(ids[-1])), (
            f"row {position} holds another text's embedding"
        )
        assert out[position, 1] == pytest.approx(float(len(ids) - 1))


def test_batching_order_does_not_change_the_result():
    """Different batch budgets group the corpus differently; the output cannot."""
    texts = _texts()
    tokenizer = _CharTokenizer()

    def run(max_tokens, max_rows):
        return cache_teacher_embeddings(
            model_teacher=_IdentityTeacher(),
            texts=texts,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            use_amp=False,
            max_tokens_per_batch=max_tokens,
            max_rows_per_batch=max_rows,
        )

    reference = run(10_000, 10_000)  # one batch
    for max_tokens, max_rows in ((64, 4), (32, 2), (16, 1)):
        assert torch.equal(reference, run(max_tokens, max_rows))


def test_length_sorted_batches_respect_both_budgets():
    lengths = [1, 50, 2, 50, 3, 50, 4]
    batches = _length_sorted_batches(lengths, max_tokens=100, max_rows=3)

    seen = sorted(i for batch in batches for i in batch)
    assert seen == list(range(len(lengths))), "every example must appear once"
    for batch in batches:
        width = max(lengths[i] for i in batch)
        assert len(batch) <= 3
        # A single example wider than the budget still has to go somewhere.
        assert width * len(batch) <= 100 or len(batch) == 1
    # The property that matters: grouping by length costs far fewer padded
    # tokens than the corpus order does. One boundary batch straddling short and
    # long examples is unavoidable for any greedy grouping, so this measures the
    # total rather than demanding pure batches.
    def padded(groups):
        return sum(max(lengths[i] for i in g) * len(g) for g in groups)

    corpus_order = [list(range(i, min(i + 3, len(lengths)))) for i in
                    range(0, len(lengths), 3)]
    assert padded(batches) < padded(corpus_order)


def test_cache_round_trip_writes_provenance(tmp_path):
    texts = _texts()
    path = str(tmp_path / "teacher.pt")

    first = cache_teacher_embeddings(
        model_teacher=_IdentityTeacher(),
        texts=texts,
        tokenizer=_CharTokenizer(),
        device=torch.device("cpu"),
        use_amp=False,
        cache_path=path,
        teacher_model_name="teacher/A",
        pooling_method="last_token",
        normalize=False,
    )
    sidecar = json.loads((tmp_path / "teacher.pt.provenance.json").read_text())
    assert sidecar["teacher_model_name"] == "teacher/A"

    # A second call must hit the cache and return the same tensor.
    second = cache_teacher_embeddings(
        model_teacher=_IdentityTeacher(),
        texts=texts,
        tokenizer=_CharTokenizer(),
        device=torch.device("cpu"),
        use_amp=False,
        cache_path=path,
        teacher_model_name="teacher/A",
    )
    assert torch.equal(first, second)

    # And a different teacher must be refused rather than silently reused.
    with pytest.raises(ValueError, match="different configuration"):
        check_cache_provenance(
            path,
            {
                "teacher_model_name": "teacher/B",
                "pooling_method": "last_token",
                "normalize": False,
            },
        )


def test_empty_corpus_is_an_explicit_error():
    with pytest.raises(ValueError, match="No texts"):
        cache_teacher_embeddings(
            model_teacher=_IdentityTeacher(),
            texts=[],
            tokenizer=_CharTokenizer(),
            device=torch.device("cpu"),
            use_amp=False,
        )
