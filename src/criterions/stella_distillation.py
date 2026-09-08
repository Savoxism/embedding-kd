"""Stella's stage-1 and stage-2 losses. The student itself is in src/models/stella.py."""

import torch
import torch.nn.functional as F


def stella_stage1_loss(
    S_emb: torch.Tensor,
    T_emb: torch.Tensor,
    w_cos: float = 10.0,
    w_sim: float = 50.0,
    w_tri: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    from src.loss import (
        cosine_embedding_loss,
        pair_inbatch_similarity_loss,
        pair_inbatch_triplet_loss,
    )

    # Normalize embeddings before computing losses
    S_emb_n = F.normalize(S_emb, p=2, dim=-1)
    T_emb_n = F.normalize(T_emb, p=2, dim=-1)

    loss_cos = cosine_embedding_loss(S_emb_n, T_emb_n)
    loss_sim = pair_inbatch_similarity_loss(S_emb_n, T_emb_n)
    loss_tri = pair_inbatch_triplet_loss(S_emb_n, T_emb_n)

    kd_sum = w_cos * loss_cos + w_sim * loss_sim + w_tri * loss_tri

    metrics = {
        "loss_total": kd_sum.item(),
        "loss_cos": loss_cos.item(),
        "loss_sim": loss_sim.item(),
        "loss_tri": loss_tri.item(),
    }

    return kd_sum, metrics


def stella_stage2_loss(
    S_cls1: torch.Tensor,
    S_cls2: torch.Tensor,
    S_emb1: torch.Tensor,
    S_emb2: torch.Tensor,
    S_emb3: torch.Tensor,
    S_emb4: torch.Tensor,
    T_emb: torch.Tensor,
    temperature: float = 0.1,
    w_task: float = 0.4,
    w_cos: float = 10.0,
    w_sim: float = 50.0,
    w_tri: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    from src.loss import (
        cosine_embedding_loss,
        info_nce,
        pair_inbatch_similarity_loss,
        pair_inbatch_triplet_loss,
    )

    loss_task, _ = info_nce(S_cls1, S_cls2, temperature=temperature)

    # Normalize all embeddings before computing losses
    S_emb1_n = F.normalize(S_emb1, p=2, dim=-1)
    S_emb2_n = F.normalize(S_emb2, p=2, dim=-1)
    S_emb3_n = F.normalize(S_emb3, p=2, dim=-1)
    S_emb4_n = F.normalize(S_emb4, p=2, dim=-1)
    T_emb_n = F.normalize(T_emb, p=2, dim=-1)

    loss_cos = cosine_embedding_loss(S_emb1_n, T_emb_n)
    loss_sim = pair_inbatch_similarity_loss(S_emb1_n, T_emb_n)
    loss_tri = pair_inbatch_triplet_loss(S_emb1_n, T_emb_n)

    loss_sim_emb2 = pair_inbatch_similarity_loss(S_emb1_n, S_emb2_n)
    loss_sim_emb3 = pair_inbatch_similarity_loss(S_emb1_n, S_emb3_n)
    loss_sim_emb4 = pair_inbatch_similarity_loss(S_emb1_n, S_emb4_n)
    loss_tri_emb2 = pair_inbatch_triplet_loss(S_emb1_n, S_emb2_n)
    loss_tri_emb3 = pair_inbatch_triplet_loss(S_emb1_n, S_emb3_n)
    loss_tri_emb4 = pair_inbatch_triplet_loss(S_emb1_n, S_emb4_n)

    kd_sum = w_cos * loss_cos + w_sim * loss_sim + w_tri * loss_tri
    kd_sum += w_sim * (loss_sim_emb2 + loss_sim_emb3 + loss_sim_emb4)
    kd_sum += w_tri * (loss_tri_emb2 + loss_tri_emb3 + loss_tri_emb4)

    total_loss = w_task * loss_task + (1 - w_task) * kd_sum

    metrics = {
        "loss_total": total_loss.item(),
        "loss_task": loss_task.item(),
        "loss_cos": loss_cos.item(),
        "loss_sim": loss_sim.item(),
        "loss_tri": loss_tri.item(),
    }

    return total_loss, metrics
