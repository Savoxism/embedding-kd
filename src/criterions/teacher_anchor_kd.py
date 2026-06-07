import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from src.loss import pair_inbatch_similarity_loss

class TeacherAnchorKD(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_layers: int = 13,  # BERT-base has 13 layers (embedding + 12 transformer layers)
        last_layer_idx: Optional[int] = None,
        start_rkd: int = 0,
        w_task: float = 0.1,
        w_kd: float = 0.75,
        w_struct: float = 10.0,
        eps_norm: float = 1e-8
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        
        # Initialize projection heads immediately in __init__
        self.n_base = num_layers
        self.last_layer_idx_config = last_layer_idx if last_layer_idx is not None else 2
        self.last_layers_idx = list(range(max(0, self.n_base - self.last_layer_idx_config), self.n_base))
        
        self.start_rkd = start_rkd
        self.w_task = w_task
        self.w_kd = w_kd
        self.w_struct = w_struct
        self.eps_norm = eps_norm
        
        # Initialize projection heads in __init__ (like DSKD)
        self.kd_proj_heads = nn.ModuleList([
            nn.Linear(self.student_dim, self.teacher_dim, bias=False)
            for _ in range(self.n_base)
        ])
        
        # Init weights with small values (from notebook)
        for head in self.kd_proj_heads:
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
    
    def compute_self_kd_loss(
        self,
        cls_base: List[torch.Tensor]
    ) -> torch.Tensor:
        if self.n_base < 2:
            return torch.tensor(0.0, device=cls_base[0].device, dtype=cls_base[0].dtype)
        
        mse_terms = []
        for i in range(self.start_rkd, self.n_base - 1):
            source = F.normalize(cls_base[i], dim=-1, eps=self.eps_norm)
            target = F.normalize(cls_base[i + 1], dim=-1, eps=self.eps_norm)
            
            loss = pair_inbatch_similarity_loss(source, target)
            mse_terms.append(loss)
        
        return torch.stack(mse_terms).mean()
    

    def compute_kd_loss(
        self,
        cls_base: List[torch.Tensor],
        teacher_cls: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        T_kd_n = F.normalize(teacher_cls, dim=-1, eps=self.eps_norm)
        
        kd_terms = []
        kd_cos_values = []
        
        for idx in self.last_layers_idx:
            if idx < 0:
                continue
            
            S_kd = self.kd_proj_heads[idx](cls_base[idx])  # [B, d_t]
            S_kd_n = F.normalize(S_kd, dim=-1, eps=self.eps_norm)
            
            kd_terms.append(1.0 - F.cosine_similarity(S_kd_n, T_kd_n, dim=-1).mean())
        
        if len(kd_terms) > 0:
            kd_loss = torch.stack(kd_terms).mean()
        else:
            kd_loss = torch.tensor(0.0, device=teacher_cls.device, dtype=teacher_cls.dtype)
        
        return kd_loss, kd_cos_values
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_cls: torch.Tensor,
        task_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        hidden_states = student_outputs['hidden_states']  # All layers
        cls_base = [h[:, 0, :] for h in hidden_states]  # Extract [CLS] from all layers
        
        # Verify number of layers matches
        if len(cls_base) != self.n_base:
            raise ValueError(f"Expected {self.n_base} layers but got {len(cls_base)}")
        
        loss_struct = self.compute_self_kd_loss(cls_base)
        
        loss_kd, kd_cos_values = self.compute_kd_loss(cls_base, teacher_cls)
        
        total_loss = (
            self.w_task * task_loss +
            self.w_kd * loss_kd +
            self.w_struct * loss_struct
        )
        metrics = {
            'loss_total': float(total_loss.detach()),
            'loss_task': float(task_loss.detach()),
            'loss_kd': float(loss_kd.detach()),
            'loss_struct': float(loss_struct.detach()),
        }
        
        for i, cos_val in enumerate(kd_cos_values):
            layer_idx = self.last_layers_idx[i]
            metrics[f'cos_layer_{layer_idx}'] = cos_val
        
        return total_loss, metrics



