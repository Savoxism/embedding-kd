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
    return align_tokens(
        teacher_tokens,
        student_tokens,
        teacher_special,
        student_special,
    )


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
    
    token_importance = token_importance.clamp_min(0)
    total = token_importance.sum()
    if total <= 1e-12:
        norm_importance = torch.full_like(token_importance, 1.0 / max(seq_len, 1))
    else:
        norm_importance = token_importance / total
    
    return norm_importance


def project_importance(teacher_importance, teacher_tokens, student_tokens, mapping):
    device = teacher_importance.device
    if len(student_tokens) == 0:
        return torch.empty(0, device=device)
    min_importance = (
        teacher_importance.min()
        if teacher_importance.numel() > 0
        else torch.tensor(1.0, device=device)
    )
    student_importance = torch.full(
        (len(student_tokens),),
        float(min_importance.item()),
        device=device,
        dtype=teacher_importance.dtype,
    )
    for teacher_idx, student_idx in mapping.items():
        if teacher_idx < teacher_importance.numel() and student_idx < len(student_tokens):
            student_importance[student_idx] = teacher_importance[teacher_idx]
    total = student_importance.sum()
    if total <= 1e-12:
        return torch.full_like(student_importance, 1.0 / len(student_tokens))
    return student_importance / total


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


def _normalize_alignment_token(token: str) -> str:
    return token.replace("##", "").lstrip("Ġ▁").lower()


def align_tokens(
    teacher_tokens,
    student_tokens,
    teacher_special="<s>",
    student_special="[CLS]",
):
    """Order-preserving one-to-one MinED alignment over token strings.

    The returned mapping is index based, so repeated token strings remain
    distinct and cannot overwrite each other.
    """

    if not teacher_tokens or not student_tokens:
        return {}
    teacher_normalized = [_normalize_alignment_token(token) for token in teacher_tokens]
    student_normalized = [_normalize_alignment_token(token) for token in student_tokens]
    teacher_count = len(teacher_tokens)
    student_count = len(student_tokens)
    dp = [[0.0] * (student_count + 1) for _ in range(teacher_count + 1)]
    backtrace = [[None] * (student_count + 1) for _ in range(teacher_count + 1)]
    for teacher_idx in range(1, teacher_count + 1):
        dp[teacher_idx][0] = float(teacher_idx)
        backtrace[teacher_idx][0] = "delete"
    for student_idx in range(1, student_count + 1):
        dp[0][student_idx] = float(student_idx)
        backtrace[0][student_idx] = "insert"

    for teacher_idx in range(1, teacher_count + 1):
        for student_idx in range(1, student_count + 1):
            teacher_token = teacher_normalized[teacher_idx - 1]
            student_token = student_normalized[student_idx - 1]
            denominator = max(len(teacher_token), len(student_token), 1)
            substitution_cost = (
                editdistance.eval(teacher_token, student_token) / denominator
            )
            options = (
                (dp[teacher_idx - 1][student_idx - 1] + substitution_cost, "match"),
                (dp[teacher_idx - 1][student_idx] + 1.0, "delete"),
                (dp[teacher_idx][student_idx - 1] + 1.0, "insert"),
            )
            dp[teacher_idx][student_idx], backtrace[teacher_idx][student_idx] = min(
                options, key=lambda option: option[0]
            )

    mapping = {}
    teacher_idx = teacher_count
    student_idx = student_count
    while teacher_idx > 0 or student_idx > 0:
        operation = backtrace[teacher_idx][student_idx]
        if operation == "match":
            source_idx = teacher_idx - 1
            target_idx = student_idx - 1
            if (
                teacher_tokens[source_idx] != teacher_special
                and student_tokens[target_idx] != student_special
            ):
                mapping[source_idx] = target_idx
            teacher_idx -= 1
            student_idx -= 1
        elif operation == "delete":
            teacher_idx -= 1
        else:
            student_idx -= 1
    return dict(sorted(mapping.items()))


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

    att_loss_total = student_atts[-1].sum() * 0.0
    att_loss_terms = 0
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

        ranked_teacher_indices = sorted(
            reciprocal,
            key=lambda index: float(teacher_importance[index].item()),
            reverse=True,
        )[:k_top]
        aligned_teacher_indices = ranked_teacher_indices
        aligned_student_indices = [
            reciprocal[teacher_idx] for teacher_idx in ranked_teacher_indices
        ]

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

            if stu_tok_att.size(1) == 0 or tea_tok_att.size(1) == 0:
                continue

            att_loss_total = att_loss_total + cka(stu_tok_att, tea_tok_att)
            att_loss_terms += 1

    if att_loss_terms == 0:
        return att_loss_total
    return att_loss_total / att_loss_terms


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
    alpha: float = 0.1,
    max_iter: int = 100,
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

        ot_loss_b, _ = sinkhorn(
            cost_matrix,
            stu_mass,
            tea_mass,
            alpha=alpha,
            max_iter=max_iter,
        )
        total_loss += ot_loss_b

    avg_loss = total_loss / batch_size
    return avg_loss


class EMODistillation(nn.Module):
    
    def __init__(
        self,
        d_teacher: int,
        d_student: int,
        k_layers: int = 2,
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
        att_loss_weight: float = 1.0,
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
            alpha=self.alpha_ot,
            max_iter=self.max_iter,
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
