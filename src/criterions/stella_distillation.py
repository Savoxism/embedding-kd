import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from src.pooling import last_token_pool, mean_pooling


class StellaModel(nn.Module):
    
    def __init__(self, model_name: str, output_dim1: int = 1024, 
                 pooling: str = 'cls', output_dim2: int = 512, 
                 output_dim3: int = 256, output_dim4: int = 128):
        super().__init__()
        from transformers import AutoModel
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(self.model_name)
        self.fc1 = nn.Linear(self.backbone.config.hidden_size, output_dim1)
        self.fc2 = nn.Linear(self.backbone.config.hidden_size, output_dim2)
        self.fc3 = nn.Linear(self.backbone.config.hidden_size, output_dim3)
        self.fc4 = nn.Linear(self.backbone.config.hidden_size, output_dim4)
        self.pooling = pooling
        
    def to(self, device):
        self.device = torch.device(device)
        return super().to(device)
    
    def forward(self, input_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        x = x.last_hidden_state
        
        if self.pooling == 'cls':
            pooled = x[:, 0, :]
        elif self.pooling == 'mean':
            pooled = mean_pooling(x, attention_mask)
        else:
            pooled = last_token_pool(x, attention_mask)
        
        z1 = self.fc1(pooled)
        z2 = self.fc2(pooled)
        z3 = self.fc3(pooled)
        z4 = self.fc4(pooled)
        
        return {
            "pooled": pooled,
            "fc1": z1,
            "fc2": z2,
            "fc3": z3,
            "fc4": z4,
        }


def stella_stage1_loss(
    S_emb: torch.Tensor,
    T_emb: torch.Tensor,
    w_cos: float = 10.0,
    w_sim: float = 50.0,
    w_tri: float = 10.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    from src.loss import cosine_embedding_loss, pair_inbatch_similarity_loss, pair_inbatch_triplet_loss
    
    # Normalize embeddings before computing losses
    S_emb_n = F.normalize(S_emb, p=2, dim=-1)
    T_emb_n = F.normalize(T_emb, p=2, dim=-1)
    
    loss_cos = cosine_embedding_loss(S_emb_n, T_emb_n)
    loss_sim = pair_inbatch_similarity_loss(S_emb_n, T_emb_n)
    loss_tri = pair_inbatch_triplet_loss(S_emb_n, T_emb_n)
    
    kd_sum = w_cos * loss_cos + w_sim * loss_sim + w_tri * loss_tri
    
    metrics = {
        'loss_total': kd_sum.item(),
        'loss_cos': loss_cos.item(),
        'loss_sim': loss_sim.item(),
        'loss_tri': loss_tri.item(),
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
    w_tri: float = 10.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    from src.loss import info_nce, cosine_embedding_loss, pair_inbatch_similarity_loss, pair_inbatch_triplet_loss
    
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
        'loss_total': total_loss.item(),
        'loss_task': loss_task.item(),
        'loss_cos': loss_cos.item(),
        'loss_sim': loss_sim.item(),
        'loss_tri': loss_tri.item(),
    }
    
    return total_loss, metrics
