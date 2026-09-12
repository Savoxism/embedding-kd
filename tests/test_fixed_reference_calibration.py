import numpy as np
import pytest
import torch

from config import GGPKDConfig
from src.criterions.ggpkd_distillation import GGPKDDistillation
from src.data_utils.ggpkd_dataset import GGPKDCollate
from src.methods.ggpkd import build_fixed_reference_indices


class _CharTokenizer:
    pad_token_id = 0

    def __call__(self, texts, truncation=True, max_length=128, **kwargs):
        return {
            "input_ids": [
                [(ord(c) % 60) + 2 for c in text[:max_length]] or [2]
                for text in texts
            ]
        }


def test_reference_selection_is_teacher_free_and_order_stable():
    texts = [f"corpus sentence {index}" for index in range(30)]
    selected = {
        texts[index] for index in build_fixed_reference_indices(texts, 7).tolist()
    }
    reordered = list(reversed(texts))
    selected_after_reorder = {
        reordered[index]
        for index in build_fixed_reference_indices(reordered, 7).tolist()
    }
    assert selected_after_reorder == selected


def test_collate_appends_references_without_changing_graph_targets():
    texts = [f"text {index}" for index in range(8)]
    collate = GGPKDCollate(
        _CharTokenizer(),
        "single_cls",
        32,
        corpus_texts=texts,
        n_scales=1,
        reference_indices=np.asarray([2, 4]),
    )
    batch = collate(
        [
            {
                "idx": 0,
                "candidate_idx": torch.tensor([2, 3]),
                "teacher_probs": torch.tensor([[0.7, 0.3]]),
            },
            {
                "idx": 1,
                "candidate_idx": torch.tensor([5, 6]),
                "teacher_probs": torch.tensor([[0.4, 0.6]]),
            },
        ]
    )

    assert batch["candidate_idx"].tolist() == [[2, 3, 2, 4], [5, 6, 2, 4]]
    assert batch["candidate_is_reference"].tolist() == [
        [False, False, True, True],
        [False, False, True, True],
    ]
    assert torch.equal(
        batch["teacher_probs"][:, :, :2],
        torch.tensor([[[0.7, 0.3]], [[0.4, 0.6]]]),
    )
    assert float(batch["teacher_probs"][:, :, 2:].abs().sum()) == 0.0
    # Index 2 occurs as both a graph column and a reference, but is encoded once.
    encoded = sum(chunk["input_ids"].size(0) for chunk in batch["candidate_chunks"])
    assert encoded == 5


def test_fixed_reference_masks_other_anchors_columns_from_calibration():
    torch.manual_seed(0)
    n_items, dim = 8, 6
    teacher = torch.randn(n_items, dim)
    student = torch.randn(n_items, dim)
    anchor_idx = torch.tensor([0, 1])
    candidate_idx = torch.tensor([[2, 3, 4, 5], [6, 7, 4, 5]])
    is_reference = torch.tensor(
        [[False, False, True, True], [False, False, True, True]]
    )
    teacher_probs = torch.tensor([[[0.7, 0.3, 0.0, 0.0]], [[0.4, 0.6, 0.0, 0.0]]])
    candidates = student.index_select(0, candidate_idx.reshape(-1))

    criterion = GGPKDDistillation(
        diffusion_scales=(1,),
        teacher_embeddings=teacher,
        direct_temp=0.2,
        row_temps=torch.full((n_items,), 0.15),
        row_weight=0.0,
        calibration_mode="fixed_reference",
    )
    (
        _,
        _,
        self_mask,
        own_mask,
        reference_columns,
        columns,
    ) = criterion._build_shared_pool(
        candidates,
        teacher_probs,
        candidate_idx,
        anchor_idx,
        is_reference,
    )
    exclusion = criterion._calibration_exclusion_mask(
        self_mask, own_mask, reference_columns
    )
    positions = {int(node): pos for pos, node in enumerate(columns.tolist())}
    assert exclusion[0, positions[6]]
    assert exclusion[0, positions[7]]
    assert not exclusion[0, positions[2]]
    assert not exclusion[0, positions[3]]
    assert not exclusion[0, positions[4]]
    assert not exclusion[0, positions[5]]

    loss, metrics = criterion(
        anchor_embeddings=student.index_select(0, anchor_idx),
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
        candidate_idx=candidate_idx,
        anchor_idx=anchor_idx,
        candidate_is_reference=is_reference,
    )
    assert torch.isfinite(loss)
    assert metrics["loss_amb"] > 0.0
    assert metrics["loss_cal"] == metrics["loss_amb"]
    assert metrics["calibration_columns"] == 4.0
    assert metrics["pool_columns_amb"] == 4.0

    graph_only = GGPKDDistillation(
        diffusion_scales=(1,),
        row_temps=torch.full((n_items,), 0.15),
        row_weight=0.0,
        calibration_mode="none",
    )
    _, graph_metrics = graph_only(
        anchor_embeddings=student.index_select(0, anchor_idx),
        candidate_embeddings=candidates,
        teacher_probs=teacher_probs,
        candidate_idx=candidate_idx,
        anchor_idx=anchor_idx,
        candidate_is_reference=is_reference,
    )
    assert metrics["loss_nbr"] == graph_metrics["loss_nbr"]


def test_calibration_mode_is_the_only_ambient_switch():
    """`use_ambient` was a second name for calibration_mode and has been removed."""
    assert GGPKDConfig(calibration_mode="fixed_reference").calibration_mode == (
        "fixed_reference"
    )
    assert GGPKDConfig(calibration_mode="none").calibration_mode == "none"
    assert not hasattr(GGPKDConfig(), "use_ambient")
    with pytest.raises(AttributeError):
        GGPKDConfig(use_ambient=False)
