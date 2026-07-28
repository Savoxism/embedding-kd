import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Sequence, Tuple


class TWMDDistillation(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        spectral_dim: int,
        scale_weights: Sequence[float],
        lambda_rw_path: float = 1.0,
        lambda_diff: float = 1.0,
        lambda_spec: float = 0.1,
        lambda_anchor: float = 0.05,
        lambda_vicreg_var: float = 25.0,
        lambda_vicreg_cov: float = 1.0,
        vicreg_gamma: float = 1.0,
        vicreg_proj_dim: int = 2048,
        student_temp: float = 0.07,
        eps_norm: float = 1e-8,
        tau_rw: float = 0.05,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.spectral_dim = spectral_dim
        self.lambda_rw_path = lambda_rw_path
        self.lambda_diff = lambda_diff
        self.lambda_spec = lambda_spec
        self.lambda_anchor = lambda_anchor
        self.lambda_vicreg_var = lambda_vicreg_var
        self.lambda_vicreg_cov = lambda_vicreg_cov
        self.vicreg_gamma = vicreg_gamma
        self.student_temp = student_temp
        self.eps_norm = eps_norm
        self.tau_rw = tau_rw

        # VICReg Expander (Projector) Network (3 layers)
        self.vicreg_projector = nn.Sequential(
            nn.Linear(student_dim, vicreg_proj_dim, bias=False),
            nn.BatchNorm1d(vicreg_proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vicreg_proj_dim, vicreg_proj_dim, bias=False),
            nn.BatchNorm1d(vicreg_proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vicreg_proj_dim, vicreg_proj_dim, bias=True)
        )

        self.anchor_proj = nn.Linear(student_dim, teacher_dim, bias=False)
        if spectral_dim > 0:
            self.spec_proj = nn.Linear(student_dim, spectral_dim, bias=False)
        else:
            self.spec_proj = None

        weights = torch.tensor(list(scale_weights), dtype=torch.float32)
        if weights.numel() == 0:
            weights = torch.ones(1, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-12)
        self.register_buffer("scale_weights", weights)

        nn.init.normal_(self.anchor_proj.weight, mean=0.0, std=1e-3)
        if self.spec_proj is not None:
            nn.init.normal_(self.spec_proj.weight, mean=0.0, std=1e-3)

    def _active_scales(self, epoch: int, n_scales: int) -> int:
        if epoch <= 0:
            return min(1, n_scales)
        if epoch == 1:
            return min(2, n_scales)
        return n_scales
        
    def twmd_path_pull_loss(self, student_anchors, student_paths):
        """
        TWMD 2.0 Contrastive-Free Walk: 
        Only pulls the student anchor towards the topological random walk mean.
        Completely eliminates False Negative pushing.
        """
        student_anchors = F.normalize(student_anchors, dim=-1, eps=self.eps_norm)
        student_paths = F.normalize(student_paths, dim=-1, eps=self.eps_norm)
        
        # Pull: 1 - cosine similarity
        loss = 1.0 - F.cosine_similarity(student_anchors, student_paths, dim=-1)
        
        return loss # Return unreduced loss [B] for gating

    def vicreg_loss(self, student_emb: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
        # student_emb: [batch_size, dim]
        # 1. Variance Loss
        std = torch.sqrt(student_emb.var(dim=0) + eps)
        loss_var = torch.mean(F.relu(gamma - std))
        
        # 2. Covariance Loss
        batch_size = student_emb.size(0)
        if batch_size > 1:
            x = student_emb - student_emb.mean(dim=0)
            cov = (x.T @ x) / (batch_size - 1)
            # mask out the diagonal
            mask = ~torch.eye(cov.size(0), dtype=torch.bool, device=cov.device)
            loss_cov = (cov[mask] ** 2).sum() / student_emb.size(-1)
        else:
            loss_cov = torch.tensor(0.0, device=student_emb.device, dtype=student_emb.dtype)
            
        return loss_var, loss_cov
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        walk_indices: torch.Tensor,
        hard_neg_indices: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_cls: torch.Tensor,
        spectral_target: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = anchor_embeddings.size(0)
        candidate_size = teacher_probs.size(-1)
        n_scales = teacher_probs.size(1)
        dim = anchor_embeddings.size(-1)

        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1, eps=self.eps_norm)
        candidate_embeddings = candidate_embeddings.view(batch_size, candidate_size, -1)
        candidate_norm = F.normalize(candidate_embeddings, p=2, dim=-1, eps=self.eps_norm)
        
        # --- TWMD Path Loss ---
        if walk_indices is not None and hard_neg_indices is not None:
            num_walks = walk_indices.size(1)
            walk_length = walk_indices.size(2)
            
            walk_emb = candidate_embeddings.gather(1, walk_indices.view(batch_size, -1).unsqueeze(-1).expand(-1, -1, dim))
            walk_emb = walk_emb.view(batch_size, num_walks, walk_length, dim)
            s_path = walk_emb.mean(dim=2)
            
            loss_rw_path_unreduced = self.twmd_path_pull_loss(anchor_embeddings, s_path)
        else:
            loss_rw_path_unreduced = torch.zeros(batch_size, device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)

        # --- MDD (HeatGeo) Barycenter Matching Loss ---
        logits = torch.einsum("bd,bcd->bc", anchor_norm, candidate_norm) / self.student_temp
        log_probs_student = F.log_softmax(logits, dim=-1)

        active_scales = self._active_scales(epoch, n_scales)
        teacher_probs = teacher_probs[:, :active_scales, :].clamp_min(1e-12)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.scale_weights.numel() < active_scales:
            pad = self.scale_weights[-1:].repeat(active_scales - self.scale_weights.numel())
            weights = torch.cat([self.scale_weights, pad], dim=0)
        else:
            weights = self.scale_weights[:active_scales]
        weights = weights / weights.sum().clamp_min(1e-12)

        log_teacher = teacher_probs.log()
        kl_per_scale = (teacher_probs * (log_teacher - log_probs_student.unsqueeze(1))).sum(dim=-1)
        
        # --- TWMD 2.0 Entropy-Gated Diffusion ---
        # Tính Entropy của Teacher Probs: H = -sum(p * log(p))
        entropy = -(teacher_probs * log_teacher).sum(dim=-1) # shape: (B, active_scales)
        max_entropy = math.log(teacher_probs.size(-1))
        normalized_entropy = entropy / max_entropy # [0, 1]
        
        # Gate: 1.0 (diffusion mạnh) khi entropy thấp (0.0), 0.0 (không diffusion) khi entropy cao (1.0)
        gate = 1.0 - normalized_entropy
        
        # Áp dụng Gate vào KL Divergence
        loss_diff = (kl_per_scale * weights.view(1, -1) * gate).sum(dim=-1).mean()
        
        # --- TWMD 3.0: Gating Random Walk ---
        mean_gate = gate.mean(dim=-1) # shape: [B]
        loss_rw_path = (loss_rw_path_unreduced * mean_gate).mean()

        # --- MDD Spectral Anchoring Loss ---
        projected = self.anchor_proj(anchor_embeddings)
        projected = F.normalize(projected, p=2, dim=-1, eps=self.eps_norm)
        teacher_norm = F.normalize(teacher_cls, p=2, dim=-1, eps=self.eps_norm)
        loss_anchor = 1.0 - F.cosine_similarity(projected, teacher_norm, dim=-1).mean()

        use_spec = (
            self.spec_proj is not None
            and spectral_target.numel() > 0
            and spectral_target.size(-1) == self.spectral_dim
            and epoch >= 2
        )
        if use_spec:
            spectral_pred = self.spec_proj(anchor_embeddings)
            loss_spec = F.mse_loss(spectral_pred, spectral_target)
        else:
            loss_spec = torch.tensor(0.0, device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)

        # --- VICReg Loss ---
        projected_emb = self.vicreg_projector(anchor_embeddings)
        loss_var, loss_cov = self.vicreg_loss(projected_emb, gamma=self.vicreg_gamma)
        
        total_loss = (
            self.lambda_rw_path * loss_rw_path
            + self.lambda_diff * loss_diff
            + self.lambda_spec * loss_spec
            + self.lambda_anchor * loss_anchor
            + self.lambda_vicreg_var * loss_var
            + self.lambda_vicreg_cov * loss_cov
        )
        
        metrics = {
            "loss_total": float(total_loss.detach()),
            "loss_rw_path": float(loss_rw_path.detach()),
            "loss_diff": float(loss_diff.detach()),
            "loss_spec": float(loss_spec.detach()),
            "loss_anchor": float(loss_anchor.detach()),
            "loss_var": float(loss_var.detach()),
            "loss_cov": float(loss_cov.detach()),
            "active_scales": float(active_scales),
        }
        return total_loss, metrics
