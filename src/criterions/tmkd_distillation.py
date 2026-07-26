from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RefinementIndex:
    """Indices and masses for one or more batches of raw-text intervals."""

    teacher_batch: torch.Tensor
    teacher_token: torch.Tensor
    student_batch: torch.Tensor
    student_token: torch.Tensor
    sentence: torch.Tensor
    within_sentence_mass: torch.Tensor
    valid_sentences: int

    @property
    def num_atoms(self) -> int:
        return int(self.teacher_token.numel())


def _valid_intervals(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: Optional[torch.Tensor],
) -> list[Tuple[int, int, int]]:
    intervals: list[Tuple[int, int, int]] = []
    for token_idx, (start, end) in enumerate(offsets.tolist()):
        if not bool(attention_mask[token_idx].item()):
            continue
        if special_tokens_mask is not None and bool(special_tokens_mask[token_idx].item()):
            continue
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        intervals.append((start, end, token_idx))

    intervals.sort(key=lambda item: (item[0], item[1], item[2]))
    previous_end = -1
    for start, end, _ in intervals:
        if start < previous_end:
            raise ValueError(
                "TMKD requires non-overlapping tokenizer offsets within each text; "
                f"found interval [{start}, {end}) after an interval ending at {previous_end}."
            )
        previous_end = end
    return intervals


def build_common_refinement(
    teacher_offsets: torch.Tensor,
    student_offsets: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    student_attention_mask: torch.Tensor,
    teacher_special_tokens_mask: Optional[torch.Tensor] = None,
    student_special_tokens_mask: Optional[torch.Tensor] = None,
    sentence_offset: int = 0,
) -> RefinementIndex:
    """Build exact intersection atoms on support covered by both tokenizers.

    Offset gaps (for example whitespace), padding, special tokens, and regions
    truncated by either tokenizer are outside the empirical measure.
    """

    if teacher_offsets.ndim != 3 or teacher_offsets.shape[-1] != 2:
        raise ValueError("teacher_offsets must have shape [batch, tokens, 2]")
    if student_offsets.ndim != 3 or student_offsets.shape[-1] != 2:
        raise ValueError("student_offsets must have shape [batch, tokens, 2]")
    if teacher_offsets.shape[0] != student_offsets.shape[0]:
        raise ValueError("teacher and student offset batches must have equal size")

    teacher_offsets = teacher_offsets.detach().cpu()
    student_offsets = student_offsets.detach().cpu()
    teacher_attention_mask = teacher_attention_mask.detach().cpu()
    student_attention_mask = student_attention_mask.detach().cpu()
    if teacher_special_tokens_mask is not None:
        teacher_special_tokens_mask = teacher_special_tokens_mask.detach().cpu()
    if student_special_tokens_mask is not None:
        student_special_tokens_mask = student_special_tokens_mask.detach().cpu()

    teacher_batch: list[int] = []
    teacher_token: list[int] = []
    student_batch: list[int] = []
    student_token: list[int] = []
    sentence: list[int] = []
    within_mass: list[float] = []
    valid_sentences = 0

    batch_size = teacher_offsets.shape[0]
    for batch_idx in range(batch_size):
        teacher_special = (
            None
            if teacher_special_tokens_mask is None
            else teacher_special_tokens_mask[batch_idx]
        )
        student_special = (
            None
            if student_special_tokens_mask is None
            else student_special_tokens_mask[batch_idx]
        )
        teacher_intervals = _valid_intervals(
            teacher_offsets[batch_idx],
            teacher_attention_mask[batch_idx],
            teacher_special,
        )
        student_intervals = _valid_intervals(
            student_offsets[batch_idx],
            student_attention_mask[batch_idx],
            student_special,
        )

        atom_records: list[Tuple[int, int, int]] = []
        atom_lengths: list[int] = []
        teacher_pos = 0
        student_pos = 0
        while teacher_pos < len(teacher_intervals) and student_pos < len(student_intervals):
            t_start, t_end, t_idx = teacher_intervals[teacher_pos]
            s_start, s_end, s_idx = student_intervals[student_pos]
            atom_start = max(t_start, s_start)
            atom_end = min(t_end, s_end)
            if atom_end > atom_start:
                atom_records.append((t_idx, s_idx, batch_idx))
                atom_lengths.append(atom_end - atom_start)

            if t_end <= s_end:
                teacher_pos += 1
            if s_end <= t_end:
                student_pos += 1

        total_length = sum(atom_lengths)
        if total_length == 0:
            continue

        sentence_idx = sentence_offset + batch_idx
        valid_sentences += 1
        for (t_idx, s_idx, record_batch), atom_length in zip(atom_records, atom_lengths):
            teacher_batch.append(record_batch)
            teacher_token.append(t_idx)
            student_batch.append(record_batch)
            student_token.append(s_idx)
            sentence.append(sentence_idx)
            within_mass.append(atom_length / total_length)

    return RefinementIndex(
        teacher_batch=torch.tensor(teacher_batch, dtype=torch.long),
        teacher_token=torch.tensor(teacher_token, dtype=torch.long),
        student_batch=torch.tensor(student_batch, dtype=torch.long),
        student_token=torch.tensor(student_token, dtype=torch.long),
        sentence=torch.tensor(sentence, dtype=torch.long),
        within_sentence_mass=torch.tensor(within_mass, dtype=torch.float32),
        valid_sentences=valid_sentences,
    )


def gather_refinement_states(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    refinement: RefinementIndex,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if refinement.num_atoms == 0:
        return (
            teacher_hidden.new_empty((0, teacher_hidden.shape[-1])),
            student_hidden.new_empty((0, student_hidden.shape[-1])),
        )

    device = student_hidden.device
    teacher_batch = refinement.teacher_batch.to(device)
    teacher_token = refinement.teacher_token.to(device)
    student_batch = refinement.student_batch.to(device)
    student_token = refinement.student_token.to(device)
    teacher_hidden = teacher_hidden.to(device, non_blocking=True)

    teacher_atoms = teacher_hidden[teacher_batch, teacher_token].detach().float()
    student_atoms = student_hidden[student_batch, student_token].float()
    teacher_atoms = F.normalize(teacher_atoms, p=2, dim=-1, eps=eps)
    student_atoms = F.normalize(student_atoms, p=2, dim=-1, eps=eps)
    return teacher_atoms, student_atoms


def tmkd_kernel_loss_explicit(
    teacher_atoms: torch.Tensor,
    student_atoms: torch.Tensor,
    masses: torch.Tensor,
) -> torch.Tensor:
    """Compute the exact weighted Gram distortion with an explicit N x N matrix."""

    if teacher_atoms.shape[0] != student_atoms.shape[0]:
        raise ValueError("teacher and student atom counts must match")
    if masses.numel() != teacher_atoms.shape[0]:
        raise ValueError("mass count must match atom count")
    if masses.numel() == 0:
        return student_atoms.sum() * 0.0

    masses = masses.to(student_atoms.device, dtype=torch.float32)
    teacher_gram = teacher_atoms @ teacher_atoms.transpose(0, 1)
    student_gram = student_atoms @ student_atoms.transpose(0, 1)
    distortion = (teacher_gram - student_gram).float()
    return (distortion.square() * masses[:, None] * masses[None, :]).sum()


def tmkd_kernel_loss_blockwise(
    teacher_atoms: torch.Tensor,
    student_atoms: torch.Tensor,
    masses: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute the same objective as the explicit form with O(block_size^2) memory."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if teacher_atoms.shape[0] != student_atoms.shape[0]:
        raise ValueError("teacher and student atom counts must match")
    if masses.numel() != teacher_atoms.shape[0]:
        raise ValueError("mass count must match atom count")
    if masses.numel() == 0:
        return student_atoms.sum() * 0.0

    masses = masses.to(student_atoms.device, dtype=torch.float32)
    num_atoms = teacher_atoms.shape[0]
    loss = student_atoms.sum() * 0.0
    for row_start in range(0, num_atoms, block_size):
        row_end = min(row_start + block_size, num_atoms)
        teacher_rows = teacher_atoms[row_start:row_end]
        student_rows = student_atoms[row_start:row_end]
        row_mass = masses[row_start:row_end]
        for col_start in range(0, num_atoms, block_size):
            col_end = min(col_start + block_size, num_atoms)
            teacher_gram = teacher_rows @ teacher_atoms[col_start:col_end].transpose(0, 1)
            student_gram = student_rows @ student_atoms[col_start:col_end].transpose(0, 1)
            distortion = (teacher_gram - student_gram).float()
            col_mass = masses[col_start:col_end]
            loss = loss + (
                distortion.square() * row_mass[:, None] * col_mass[None, :]
            ).sum()
    return loss


class TMKDDistillation(nn.Module):
    def __init__(
        self,
        lambda_tmkd: float = 1.0,
        block_size: int = 512,
        mode: str = "full",
        eps_norm: float = 1e-8,
    ):
        super().__init__()
        if lambda_tmkd < 0:
            raise ValueError("lambda_tmkd must be non-negative")
        if mode not in {"full", "within"}:
            raise ValueError("mode must be 'full' or 'within'")
        self.lambda_tmkd = float(lambda_tmkd)
        self.block_size = int(block_size)
        self.mode = mode
        self.eps_norm = float(eps_norm)

    def forward(
        self,
        teacher_hidden_states: Sequence[torch.Tensor],
        student_hidden_states: Sequence[torch.Tensor],
        teacher_offsets: Sequence[torch.Tensor],
        student_offsets: Sequence[torch.Tensor],
        teacher_attention_masks: Sequence[torch.Tensor],
        student_attention_masks: Sequence[torch.Tensor],
        teacher_special_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
        student_special_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        group_count = len(teacher_hidden_states)
        inputs = (
            student_hidden_states,
            teacher_offsets,
            student_offsets,
            teacher_attention_masks,
            student_attention_masks,
        )
        if any(len(values) != group_count for values in inputs):
            raise ValueError("all TMKD input sequences must have equal length")
        if teacher_special_masks is None:
            teacher_special_masks = [None] * group_count
        if student_special_masks is None:
            student_special_masks = [None] * group_count
        if len(teacher_special_masks) != group_count or len(student_special_masks) != group_count:
            raise ValueError("special-mask sequences must match hidden-state group count")

        teacher_atom_groups: list[torch.Tensor] = []
        student_atom_groups: list[torch.Tensor] = []
        within_mass_groups: list[torch.Tensor] = []
        sentence_groups: list[torch.Tensor] = []
        sentence_offset = 0
        valid_sentences = 0

        for group_idx in range(group_count):
            refinement = build_common_refinement(
                teacher_offsets=teacher_offsets[group_idx],
                student_offsets=student_offsets[group_idx],
                teacher_attention_mask=teacher_attention_masks[group_idx],
                student_attention_mask=student_attention_masks[group_idx],
                teacher_special_tokens_mask=teacher_special_masks[group_idx],
                student_special_tokens_mask=student_special_masks[group_idx],
                sentence_offset=sentence_offset,
            )
            sentence_offset += int(teacher_offsets[group_idx].shape[0])
            if refinement.num_atoms == 0:
                continue
            teacher_atoms, student_atoms = gather_refinement_states(
                teacher_hidden_states[group_idx],
                student_hidden_states[group_idx],
                refinement,
                eps=self.eps_norm,
            )
            teacher_atom_groups.append(teacher_atoms)
            student_atom_groups.append(student_atoms)
            within_mass_groups.append(
                refinement.within_sentence_mass.to(student_atoms.device)
            )
            sentence_groups.append(refinement.sentence.to(student_atoms.device))
            valid_sentences += refinement.valid_sentences

        if not student_atom_groups or valid_sentences == 0:
            reference = student_hidden_states[0]
            zero = reference.sum() * 0.0
            return zero, {
                "loss_tmkd": 0.0,
                "num_atoms": 0.0,
                "valid_texts": 0.0,
            }

        teacher_atoms = torch.cat(teacher_atom_groups, dim=0)
        student_atoms = torch.cat(student_atom_groups, dim=0)
        within_masses = torch.cat(within_mass_groups, dim=0)
        sentence_ids = torch.cat(sentence_groups, dim=0)

        if self.mode == "full":
            masses = within_masses / valid_sentences
            kd_loss = tmkd_kernel_loss_blockwise(
                teacher_atoms,
                student_atoms,
                masses,
                block_size=self.block_size,
            )
        else:
            kd_loss = student_atoms.sum() * 0.0
            unique_sentences = torch.unique(sentence_ids)
            for sentence_id in unique_sentences:
                keep = sentence_ids == sentence_id
                kd_loss = kd_loss + tmkd_kernel_loss_blockwise(
                    teacher_atoms[keep],
                    student_atoms[keep],
                    within_masses[keep],
                    block_size=self.block_size,
                )
            kd_loss = kd_loss / unique_sentences.numel()

        weighted_loss = self.lambda_tmkd * kd_loss
        return weighted_loss, {
            "loss_tmkd": float(kd_loss.detach().item()),
            "weighted_tmkd": float(weighted_loss.detach().item()),
            "num_atoms": float(student_atoms.shape[0]),
            "valid_texts": float(valid_sentences),
        }
