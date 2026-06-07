import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional
import editdistance


class CKALoss(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1,dS).to(SH.device,torch.float64)
        TH = TH.view(-1,dT).to(SH.device,torch.float64)
        
        slen = SH.size(0)
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
                
        num = torch.norm(SH.t().matmul(TH),'fro')
        den1 = torch.norm(SH.t().matmul(SH),'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH),'fro') + self.eps
        
        return 1 - num/torch.sqrt(den1*den2)


def build_reciprocal_mapping_from_token_lists(
    teacher_tokens, student_tokens,
    teacher_special="<s>", student_special="[CLS]"
):
    teacher_to_student = align_tokens(teacher_tokens, student_tokens, teacher_special, student_special)
    student_to_teacher = align_tokens(student_tokens, teacher_tokens, student_special, teacher_special)

    reciprocal_mapping = {}
    for t, s in teacher_to_student.items():
        if s in student_to_teacher and student_to_teacher[s] == t:
            reciprocal_mapping[t] = s
    return reciprocal_mapping


def build_reciprocal_mapping_from_token_lists_old(
    tokens_t: List[str],
    tokens_s: List[str],
    teacher_special: str = "<s>",
    student_special: str = "[CLS]"
) -> Dict[int, int]:
    indices_t = [i for i, tok in enumerate(tokens_t) if not tok.startswith(teacher_special)]
    indices_s = [i for i, tok in enumerate(tokens_s) if not tok.startswith(student_special)]
    
    pure_tokens_t = [tokens_t[i] for i in indices_t]
    pure_tokens_s = [tokens_s[i] for i in indices_s]
    
    n_t = len(pure_tokens_t)
    n_s = len(pure_tokens_s)
    
    dist_matrix = torch.zeros(n_t, n_s)
    for i in range(n_t):
        for j in range(n_s):
            dist_matrix[i, j] = editdistance.eval(pure_tokens_t[i], pure_tokens_s[j])
    
    mapping = {}
    used_s = set()
    
    for i in range(n_t):
        min_dist = float('inf')
        best_j = -1
        for j in range(n_s):
            if j not in used_s and dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
                best_j = j
        
        if best_j >= 0:
            reciprocal = True
            for k in range(n_t):
                if dist_matrix[k, best_j] < dist_matrix[i, best_j]:
                    reciprocal = False
                    break
            
            if reciprocal:
                mapping[indices_t[i]] = indices_s[best_j]
                used_s.add(best_j)
    
    return mapping


def compute_token_importance(attention_weights, tokens):
    device = attention_weights.device
    
    # Check if attention_weights is 3D (with multiple heads) or 2D (single attention matrix)
    if len(attention_weights.shape) == 3:
        # Average attention across heads: [seq_len, seq_len]
        avg_attention = attention_weights.mean(dim=0)
    else:
        # Already a 2D attention matrix
        avg_attention = attention_weights
    
    # Ensure dimensions match
    seq_len = min(avg_attention.shape[0], len(tokens))
    
    # Truncate attention matrix if needed
    avg_attention = avg_attention[:seq_len, :seq_len]
    
    # Sum attention that each token receives: [seq_len]
    token_importance = avg_attention.sum(dim=0)
    
    # Normalize importance scores (add small epsilon to avoid division by zero)
    norm_importance = torch.softmax(token_importance, dim=0)
    
    return norm_importance


def project_importance(teacher_importance, teacher_tokens, student_tokens, mapping):
    device = teacher_importance.device
    student_importance = torch.zeros(len(student_tokens), device=device)
    
    # Get valid teacher tokens based on attention mask
    valid_teacher_tokens = teacher_tokens[:teacher_importance.shape[0]]
    
    # Map valid tokens to importance scores
    teacher_token_to_importance = {token: score.item() for token, score in zip(valid_teacher_tokens, teacher_importance)}
    
    # Keep track of mapped student indices
    mapped_student_indices = set()
    
    # Project importance scores
    for t_idx, t in enumerate(valid_teacher_tokens):
        if t in mapping:
            s = mapping[t]
            # Find all occurrences of this student token
            s_indices = [i for i, token in enumerate(student_tokens) if token == s]
            for s_idx in s_indices:
                if s_idx < len(student_importance):  # Ensure index is valid
                    student_importance[s_idx] = teacher_importance[t_idx]
                    mapped_student_indices.add(s_idx)
    
    # Find minimum importance score from teacher for unmapped tokens
    min_importance = teacher_importance.min().item() if len(teacher_importance) > 0 else 0.0
    for s_idx in range(len(student_tokens)):
        if s_idx not in mapped_student_indices and s_idx < len(student_importance):
            student_importance[s_idx] = min_importance
    student_importance = torch.softmax(student_importance, dim=0)
    
    return student_importance


def sinkhorn(
    cost_matrix: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.1,
    max_iter: int = 100,
    stop_thr: float = 1e-9,
    eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sinkhorn Optimal Transport for 2D cost matrix (single batch item).
    cost_matrix: [m, n]
    a: [m, 1] or [m] - source marginal
    b: [n, 1] or [n] - target marginal
    """
    m, n = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype
    
    if m == 0 or n == 0:
        return (
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.zeros((m, n), device=device, dtype=dtype)
        )
    
    a = a.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)
    
    # Ensure correct shape
    if a.dim() == 1:
        a = a.view(-1, 1)
    if b.dim() == 1:
        b = b.view(-1, 1)
    
    # Fallback to uniform if dimensions don't match
    if a.shape[0] != m:
        a = torch.ones(m, 1, device=device, dtype=dtype) / m
    if b.shape[0] != n:
        b = torch.ones(n, 1, device=device, dtype=dtype) / n
    
    # Normalize marginals
    if torch.sum(a) < eps or torch.sum(b) < eps:
        a = torch.ones(m, 1, device=device, dtype=dtype) / m
        b = torch.ones(n, 1, device=device, dtype=dtype) / n
    else:
        a = a / torch.sum(a)
        b = b / torch.sum(b)
    
    # Sinkhorn iterations
    K = torch.exp(-cost_matrix / alpha)
    u = torch.ones(m, 1, device=device, dtype=dtype)
    v = torch.ones(n, 1, device=device, dtype=dtype)
    
    for _ in range(max_iter):
        u_prev = u.clone()
        KTu = torch.matmul(K.t(), u)  # [n, 1]
        v = b / (KTu + eps)
        Kv = torch.matmul(K, v)  # [m, 1]
        u = a / (Kv + eps)
        
        # Check convergence
        err = torch.norm(u - u_prev, p=float('inf'))
        if err < stop_thr:
            break
    
    # Compute transport matrix
    P = torch.diag(u.squeeze()) @ K @ torch.diag(v.squeeze())  # [m, n]
    
    # Compute OT loss
    ot_loss = torch.sum(P * cost_matrix)
    
    return ot_loss, P


def pairwise_attention_distance(x, y, eps=1e-8):
    d = x.shape[1]
    sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
    attention_weights = torch.softmax(sim_mt, dim=1)
    dist_mt = 1.0 - attention_weights
    return dist_mt


def align_tokens(teacher_tokens, student_tokens, teacher_special="<s>", student_special="[CLS]"):
    # Create mapping dictionary
    teacher_to_student = {}
    
    # Handle empty token lists
    if not teacher_tokens or not student_tokens:
        return teacher_to_student
    if teacher_special in teacher_tokens and student_special in student_tokens:
        teacher_to_student[teacher_special] = student_special
    student_token_set = set(student_tokens)
    
    for t in teacher_tokens:
        tmp_t = t.replace(teacher_special, student_special)
        if tmp_t in student_token_set:
            teacher_to_student[t] = tmp_t
            continue
        
        best_s = None
        best_dist = float('inf')
        
        for s in student_tokens:
            if s == student_special:
                continue
                
            # Calculate edit distance
            d = editdistance.eval(tmp_t, s)
            if d < best_dist:
                best_s = s
                best_dist = d
        if best_s is not None:
            teacher_to_student[t] = best_s
    
    return teacher_to_student


def compute_att_loss_2(
    teacher_atts,          # list: mỗi phần tử [B, H, L_t, L_t] trên device_s
    student_atts,          # list: mỗi phần tử [B, H, L_s, L_s]
    input_ids_tea,         # [B, L_t] trên device_s
    input_ids_stu,         # [B, L_s] trên device_s
    attention_mask_tea,    # [B, L_t] trên device_s
    attention_mask_stu,    # [B, L_s] trên device_s
    tok_teacher,
    tok_student,
    k,                     # số last layers dùng (thường = 1)
    device,
    teacher_special="<s>",
    student_special="[CLS]",
):

    att_loss_total = 0.0
    batch_size = input_ids_stu.size(0)

    teacher_layer_num = len(teacher_atts)
    student_layer_num = len(student_atts)
    layers_per_block = teacher_layer_num // student_layer_num
    new_teacher_atts = [
        teacher_atts[idx * layers_per_block + layers_per_block - 1]
        for idx in range(student_layer_num)
    ]

    teacher_last_k_layers = new_teacher_atts[-k:]   
    student_last_k_layers = student_atts[-k:]       

    cka = CKALoss(eps=1e-8).to(device)

    for b in range(batch_size):
        L_t_valid = int(attention_mask_tea[b].sum().item())
        L_s_valid = int(attention_mask_stu[b].sum().item())
        if L_t_valid == 0 or L_s_valid == 0:
            continue

        ids_tea = input_ids_tea[b, :L_t_valid]
        ids_stu = input_ids_stu[b, :L_s_valid]

        teacher_tokens = tok_teacher.convert_ids_to_tokens(ids_tea.detach().cpu().tolist())
        student_tokens = tok_student.convert_ids_to_tokens(ids_stu.detach().cpu().tolist())

        last_teacher_att_full = teacher_atts[-1][b]        # [H, L_t, L_t]
        last_teacher_att = last_teacher_att_full[:, :L_t_valid, :L_t_valid]

        teacher_importance = compute_token_importance(
            last_teacher_att, teacher_tokens
        )  # [L_t_valid]

        reciprocal = build_reciprocal_mapping_from_token_lists(
            teacher_tokens, student_tokens,
            teacher_special=teacher_special,
            student_special=student_special
        )
        if len(reciprocal) == 0:
            continue

        n_map = len(reciprocal)
        k_top = max(1, n_map // 3)

        token_scores = {}
        for idx_t, tok_t in enumerate(teacher_tokens):
            if tok_t in reciprocal:
                score = teacher_importance[idx_t].item()
                if tok_t not in token_scores or score > token_scores[tok_t]:
                    token_scores[tok_t] = score

        if len(token_scores) == 0:
            continue

        top_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:k_top]
        allowed_teacher_tokens = {t for t, _ in top_tokens}

        aligned_teacher_indices = []
        aligned_student_indices = []

        for t_tok in allowed_teacher_tokens:
            s_tok = reciprocal[t_tok]
            # teacher index: lấy xuất hiện đầu tiên
            try:
                t_idx = teacher_tokens.index(t_tok)
            except ValueError:
                continue
            # student index: xuất hiện đầu tiên
            try:
                s_idx = student_tokens.index(s_tok)
            except ValueError:
                continue
            if t_idx < L_t_valid and s_idx < L_s_valid:
                aligned_teacher_indices.append(t_idx)
                aligned_student_indices.append(s_idx)

        N = len(aligned_teacher_indices)
        if N == 0:
            continue

        for teacher_att_layer, student_att_layer in zip(teacher_last_k_layers, student_last_k_layers):
            # teacher_att_layer: [B, H, L_t, L_t]
            # student_att_layer: [B, H, L_s, L_s]

            tea_sub = teacher_att_layer[b, :, :L_t_valid, :L_t_valid]  # [H, L_t_valid, L_t_valid]
            stu_sub = student_att_layer[b, :, :L_s_valid, :L_s_valid]  # [H, L_s_valid, L_s_valid]

            # attention của N token đã align, trung bình các head
            tea_tok_att = tea_sub[:, aligned_teacher_indices, :].mean(dim=0)  # [N, L_t_valid]
            stu_tok_att = stu_sub[:, aligned_student_indices, :].mean(dim=0)  # [N, L_s_valid]

            # xử lý mask rất nhỏ
            tea_tok_att = torch.where(
                tea_tok_att <= -1e2,
                torch.zeros_like(tea_tok_att, device=device),
                tea_tok_att
            )
            stu_tok_att = torch.where(
                stu_tok_att <= -1e2,
                torch.zeros_like(stu_tok_att, device=device),
                stu_tok_att
            )

            # 10) ép feature dim giống nhau để CKA không lỗi
            d_s = stu_tok_att.size(1)
            d_t = tea_tok_att.size(1)
            d = min(d_s, d_t)
            if d == 0:
                continue

            SH = stu_tok_att[:, :d]  # [N, d]
            TH = tea_tok_att[:, :d]  # [N, d]

            att_loss_total += cka(SH, TH)

    return att_loss_total


def compute_ot_loss(
    teacher_last,           # [B, L_t, d_t]  trên device_s
    student_last,           # [B, L_s, d_s]  trên device_s
    teacher_att_last,       # [B, H, L_t, L_t] (attention layer cuối) trên device_s
    attention_mask_teacher, # [B, L_t] trên device_s
    attention_mask_student, # [B, L_s] trên device_s
    input_ids_tea,          # [B, L_t] trên device_s
    input_ids_stu,          # [B, L_s] trên device_s
    tok_teacher,
    tok_student,
    projector,              # proj_t2s: Linear(d_t -> d_s)
    teacher_special="<s>",
    student_special="[CLS]",
):
    device = teacher_last.device
    batch_size = teacher_last.size(0)
    total_loss = 0.0

    for b in range(batch_size):
        valid_teacher_len = int(attention_mask_teacher[b].sum().item())
        valid_student_len = int(attention_mask_student[b].sum().item())

        valid_teacher_input_ids = input_ids_tea[b, :valid_teacher_len]
        valid_student_input_ids = input_ids_stu[b, :valid_student_len]

        teacher_tokens = tok_teacher.convert_ids_to_tokens(
            valid_teacher_input_ids.detach().cpu().tolist()
        )
        student_tokens = tok_student.convert_ids_to_tokens(
            valid_student_input_ids.detach().cpu().tolist()
        )

        teacher_seq = teacher_last[b, :valid_teacher_len, :]   # [Lt', d_t]
        student_seq = student_last[b, :valid_student_len, :]   # [Ls', d_s]

        projected_teacher_seq = projector(teacher_seq)         # [Lt', d_s]

        teacher_attention_full = teacher_att_last[b]           # [H, L_t, L_t]
        valid_teacher_attention = teacher_attention_full[:, :valid_teacher_len, :valid_teacher_len]

        teacher_importance = compute_token_importance(
            valid_teacher_attention, teacher_tokens
        )  # [Lt']

        token_mapping = align_tokens(
            teacher_tokens, student_tokens,
            teacher_special=teacher_special,
            student_special=student_special
        )

        student_importance = project_importance(
            teacher_importance,
            teacher_tokens,
            student_tokens,
            token_mapping
        )   # [Ls']

        tea_mass = teacher_importance.view(-1, 1).to(device=device, dtype=torch.float32)
        stu_mass = student_importance.view(-1, 1).to(device=device, dtype=torch.float32)

        student_seq_f = student_seq.to(device=device, dtype=torch.float32)
        proj_teacher_seq_f = projected_teacher_seq.to(device=device, dtype=torch.float32)

        cost_matrix = pairwise_attention_distance(student_seq_f, proj_teacher_seq_f)
        cost_matrix = cost_matrix.to(device=device, dtype=torch.float32)

        ot_loss_b, _ = sinkhorn(cost_matrix, stu_mass, tea_mass)
        total_loss += ot_loss_b

    avg_loss = total_loss / batch_size
    return avg_loss


class EMODistillation(nn.Module):
    
    def __init__(
        self,
        d_teacher: int,
        d_student: int,
        k_layers: int = 1,
        alpha_ot: float = 0.1,
        max_iter: int = 100,
        teacher_special: str = "<s>",
        student_special: str = "[CLS]"
    ):
        super().__init__()
        
        self.k_layers = k_layers
        self.alpha_ot = alpha_ot
        self.max_iter = max_iter
        self.teacher_special = teacher_special
        self.student_special = student_special
        
        self.proj_t2s = nn.Linear(d_teacher, d_student, bias=False)
    
    def compute_emo_loss(
        self,
        teacher_outputs,
        student_outputs,
        input_ids_tea: torch.Tensor,
        input_ids_stu: torch.Tensor,
        attention_mask_tea: torch.Tensor,
        attention_mask_stu: torch.Tensor,
        tok_teacher,
        tok_student,
        att_loss_weight: float = 0.1,
        ot_loss_weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = student_outputs.last_hidden_state.device
        
        att_loss = compute_att_loss_2(
            teacher_atts=list(teacher_outputs.attentions),
            student_atts=list(student_outputs.attentions),
            input_ids_tea=input_ids_tea,
            input_ids_stu=input_ids_stu,
            attention_mask_tea=attention_mask_tea,
            attention_mask_stu=attention_mask_stu,
            tok_teacher=tok_teacher,
            tok_student=tok_student,
            k=self.k_layers,
            device=device,
            teacher_special=self.teacher_special,
            student_special=self.student_special
        )
        
        ot_loss = compute_ot_loss(
            teacher_last=teacher_outputs.last_hidden_state,
            student_last=student_outputs.last_hidden_state,
            teacher_att_last=teacher_outputs.attentions[-1],
            attention_mask_teacher=attention_mask_tea,
            attention_mask_student=attention_mask_stu,
            input_ids_tea=input_ids_tea,
            input_ids_stu=input_ids_stu,
            tok_teacher=tok_teacher,
            tok_student=tok_student,
            projector=self.proj_t2s,
            teacher_special=self.teacher_special,
            student_special=self.student_special
        )
        
        total_loss = att_loss_weight * att_loss + ot_loss_weight * ot_loss
        
        loss_dict = {
            "att_loss": att_loss.item(),
            "ot_loss": ot_loss.item(),
            "total_kd": total_loss.item()
        }
        
        return total_loss, loss_dict
