import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DualSpaceKD(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        attention_dim: Optional[int] = None,
        w_task: float = 0.5,
        alpha_dtw: float = 0.5,
        kd_tok_weight: float = 1.0
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.d_att = attention_dim if attention_dim is not None else min(student_dim, teacher_dim)
        self.w_task = w_task
        self.alpha_dtw = alpha_dtw
        self.kd_tok_weight = kd_tok_weight
        
        self.proj_s2t = nn.Linear(student_dim, teacher_dim, bias=False)
        self.proj_t2s = nn.Linear(teacher_dim, student_dim, bias=False)
        self.cla_q_s2t = nn.Linear(student_dim, self.d_att, bias=False)
        self.cla_k_s2t = nn.Linear(teacher_dim, self.d_att, bias=False)
        self.cla_v_s2t = nn.Linear(teacher_dim, student_dim, bias=False)
        self.cla_q_t2s = nn.Linear(teacher_dim, self.d_att, bias=False)
        self.cla_k_t2s = nn.Linear(student_dim, self.d_att, bias=False)
        self.cla_v_t2s = nn.Linear(student_dim, teacher_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def cla_align(
        self,
        Q_seq: torch.Tensor,
        K_seq: torch.Tensor,
        V_seq: torch.Tensor,
        mask_q: torch.Tensor,
        mask_k: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ) -> torch.Tensor:
        Q = q_proj(Q_seq) 
        K = k_proj(K_seq)  
        V = v_proj(V_seq) 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1))  # [B, Lq, Lk]
        
        mask_k_exp = mask_k.unsqueeze(1)  # [B, 1, Lk]
        scores = scores.masked_fill(~mask_k_exp, float("-inf"))
        
        A = torch.softmax(scores, dim=-1)  
        aligned = torch.matmul(A, V)  # [B, Lq, d_out]
        
        return aligned
    
    def compute_dskd_loss(
        self,
        S_last: torch.Tensor,
        T_last: torch.Tensor,
        S_cls: torch.Tensor,
        T_cls: torch.Tensor,
        mask_student: torch.Tensor,
        mask_teacher: torch.Tensor,
        task_loss: torch.Tensor,
        special_tokens_mask_student: Optional[torch.Tensor] = None,
        special_tokens_mask_teacher: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if device is None:
            device = S_cls.device
        
        L2 = lambda x: F.normalize(x, p=2, dim=-1)
        
        # ===== CLS-level bidirectional KD =====
        S1_to_T = self.proj_s2t(S_cls)  # [B, d_t]
        kd_s2t = F.mse_loss(L2(S1_to_T), L2(T_cls.detach()))
        
        T1_to_S = self.proj_t2s(T_cls)  
        kd_t2s = F.mse_loss(L2(S_cls), L2(T1_to_S.detach()))
        kd_cls = kd_s2t + kd_t2s
        
        # ===== Token-level bidirectional KD with CLA =====
        mask_s1 = mask_student.bool()
        mask_t1 = mask_teacher.bool()
        
        T_to_S_1 = self.cla_align(
            Q_seq=S_last,
            K_seq=T_last,
            V_seq=T_last,
            mask_q=mask_s1,
            mask_k=mask_t1,
            q_proj=self.cla_q_s2t,
            k_proj=self.cla_k_s2t,
            v_proj=self.cla_v_s2t,
        )  
        
        S_to_T_1 = self.cla_align(
            Q_seq=T_last,
            K_seq=S_last,
            V_seq=S_last,
            mask_q=mask_t1,
            mask_k=mask_s1,
            q_proj=self.cla_q_t2s,
            k_proj=self.cla_k_t2s,
            v_proj=self.cla_v_t2s,
        )  
        if special_tokens_mask_student is not None:
            keep_s1 = mask_s1 & (~special_tokens_mask_student.bool())
        else:
            keep_s1 = mask_s1
        
        if special_tokens_mask_teacher is not None:
            keep_t1 = mask_t1 & (~special_tokens_mask_teacher.bool())
        else:
            keep_t1 = mask_t1
        
        if keep_s1.any():
            kd_tok_s1 = F.mse_loss(L2(S_last[keep_s1]), L2(T_to_S_1[keep_s1]))
        else:
            kd_tok_s1 = S_last.new_tensor(0.0)
        
        if keep_t1.any():
            kd_tok_t1 = F.mse_loss(L2(T_last[keep_t1]), L2(S_to_T_1[keep_t1]))
        else:
            kd_tok_t1 = T_last.new_tensor(0.0)
        
        kd_tok = (kd_tok_s1 + kd_tok_t1) * self.kd_tok_weight
        
        # ===== Total loss =====
        kd_all = kd_cls + kd_tok
        total_loss = (self.w_task * task_loss + self.alpha_dtw * kd_all).float()
        
        metrics = {
            'loss_total': float(total_loss.detach()),
            'loss_task': float(task_loss.detach()),
            'kd_cls': float(kd_cls.detach()),
            'kd_s2t': float(kd_s2t.detach()),
            'kd_t2s': float(kd_t2s.detach()),
            'kd_tok': float(kd_tok.detach()),
            'kd_tok_s1': float(kd_tok_s1.detach()),
            'kd_tok_t1': float(kd_tok_t1.detach()),
        }
        
        return total_loss, metrics


