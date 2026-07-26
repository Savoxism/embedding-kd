import torch

from src.criterions.tmkd_distillation import (
    TMKDDistillation,
    build_common_refinement,
    gather_refinement_states,
    tmkd_kernel_loss_blockwise,
    tmkd_kernel_loss_explicit,
)


def _masks(length):
    return torch.ones((1, length), dtype=torch.long), torch.zeros(
        (1, length), dtype=torch.long
    )


def test_common_refinement_and_length_masses():
    teacher_offsets = torch.tensor([[[0, 2], [2, 5]]])
    student_offsets = torch.tensor([[[0, 1], [1, 4], [4, 5]]])
    teacher_attention, teacher_special = _masks(2)
    student_attention, student_special = _masks(3)

    refinement = build_common_refinement(
        teacher_offsets,
        student_offsets,
        teacher_attention,
        student_attention,
        teacher_special,
        student_special,
    )

    assert refinement.teacher_token.tolist() == [0, 0, 1, 1]
    assert refinement.student_token.tolist() == [0, 1, 1, 2]
    assert torch.allclose(
        refinement.within_sentence_mass,
        torch.tensor([0.2, 0.2, 0.4, 0.2]),
    )
    assert refinement.valid_sentences == 1


def test_special_padding_gaps_and_unshared_truncation_are_excluded():
    teacher_offsets = torch.tensor([[[0, 0], [0, 2], [3, 5], [0, 0]]])
    student_offsets = torch.tensor([[[0, 0], [0, 1], [1, 5], [0, 0]]])
    teacher_attention = torch.tensor([[1, 1, 1, 0]])
    student_attention = torch.tensor([[1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1]])

    refinement = build_common_refinement(
        teacher_offsets,
        student_offsets,
        teacher_attention,
        student_attention,
        special,
        special,
    )

    assert refinement.teacher_token.tolist() == [1, 1, 2]
    assert refinement.student_token.tolist() == [1, 2, 2]
    assert torch.allclose(
        refinement.within_sentence_mass,
        torch.tensor([0.25, 0.25, 0.5]),
    )


def test_explicit_and_blockwise_losses_and_gradients_match():
    generator = torch.Generator().manual_seed(7)
    teacher = torch.nn.functional.normalize(torch.randn(9, 7, generator=generator), dim=-1)
    student_explicit = torch.randn(9, 5, generator=generator, requires_grad=True)
    student_blockwise = student_explicit.detach().clone().requires_grad_(True)
    student_explicit_norm = torch.nn.functional.normalize(student_explicit, dim=-1)
    student_blockwise_norm = torch.nn.functional.normalize(student_blockwise, dim=-1)
    masses = torch.rand(9, generator=generator)
    masses = masses / masses.sum()

    explicit = tmkd_kernel_loss_explicit(teacher, student_explicit_norm, masses)
    blockwise = tmkd_kernel_loss_blockwise(
        teacher, student_blockwise_norm, masses, block_size=3
    )
    explicit.backward()
    blockwise.backward()

    assert torch.allclose(explicit, blockwise, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        student_explicit.grad, student_blockwise.grad, atol=2e-6, rtol=2e-6
    )


def test_independent_orthogonal_transforms_leave_loss_unchanged():
    generator = torch.Generator().manual_seed(11)
    teacher = torch.nn.functional.normalize(torch.randn(8, 6, generator=generator), dim=-1)
    student = torch.nn.functional.normalize(torch.randn(8, 4, generator=generator), dim=-1)
    teacher_q, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator))
    student_q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    masses = torch.full((8,), 1 / 8)

    original = tmkd_kernel_loss_explicit(teacher, student, masses)
    transformed = tmkd_kernel_loss_explicit(
        teacher @ teacher_q, student @ student_q, masses
    )
    assert torch.allclose(original, transformed, atol=1e-6, rtol=1e-6)


def test_refinement_invariance_for_identical_split_states():
    teacher_offsets = torch.tensor([[[0, 2], [2, 4]]])
    student_offsets = torch.tensor([[[0, 4]]])
    teacher_attention, teacher_special = _masks(2)
    student_attention, student_special = _masks(1)
    refinement = build_common_refinement(
        teacher_offsets,
        student_offsets,
        teacher_attention,
        student_attention,
        teacher_special,
        student_special,
    )
    state = torch.tensor([1.0, 2.0, 3.0])
    teacher_hidden = torch.stack([state, state]).unsqueeze(0)
    student_hidden = state.reshape(1, 1, -1).clone().requires_grad_(True)
    teacher_atoms, student_atoms = gather_refinement_states(
        teacher_hidden, student_hidden, refinement
    )
    loss = tmkd_kernel_loss_explicit(
        teacher_atoms,
        student_atoms,
        refinement.within_sentence_mass,
    )
    assert loss.item() < 1e-7


def test_module_loss_is_bounded_and_backpropagates_only_to_student():
    teacher_hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True
    )
    student_hidden = torch.tensor(
        [[[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]], requires_grad=True
    )
    offsets = torch.tensor([[[0, 1], [1, 2]]])
    attention, special = _masks(2)
    criterion = TMKDDistillation(block_size=1)

    loss, metrics = criterion(
        teacher_hidden_states=[teacher_hidden],
        student_hidden_states=[student_hidden],
        teacher_offsets=[offsets],
        student_offsets=[offsets],
        teacher_attention_masks=[attention],
        student_attention_masks=[attention],
        teacher_special_masks=[special],
        student_special_masks=[special],
    )
    loss.backward()

    assert 0.0 <= loss.item() <= 4.0
    assert metrics["valid_texts"] == 1.0
    assert teacher_hidden.grad is None
    assert student_hidden.grad is not None
    assert torch.isfinite(student_hidden.grad).all()
