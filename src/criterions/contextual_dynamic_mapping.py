import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import editdistance
from typing import List, Tuple, Optional, Sequence, Dict
from collections import deque
import time


def dist_fn(a: str, b: str, 
            blending_model_special_token: str = "G",
            base_model_special_token: str = "##",
            specTok_mapper: Optional[Dict] = None) -> float:

    if specTok_mapper is None:
        specTok_mapper = {}
        
    if a in specTok_mapper and b in specTok_mapper.values():
        return 0.0
    if b in specTok_mapper and a in specTok_mapper.values():
        return 0.0
        
    aa = a.replace(blending_model_special_token, "").replace(" ", "")
    bb = b.replace(base_model_special_token, "").replace(" ", "")
    
    dist = editdistance.eval(aa, bb)
    if len(aa) == len(bb) == 0:
        return 0.0
        
    dist = dist / (len(aa) + len(bb))
    return dist


def cost_fn(a: str, b: str,
            blending_model_special_token: str = "G", 
            base_model_special_token: str = "##",
            specTok_mapper: Optional[Dict] = None) -> float:

    if specTok_mapper is None:
        specTok_mapper = {}
        
    if a in specTok_mapper and b in specTok_mapper.values():
        return 0.0
    if b in specTok_mapper and a in specTok_mapper.values():
        return 0.0
        
    aa = a.replace(blending_model_special_token, "").replace(" ", "")
    bb = b.replace(base_model_special_token, "").replace(" ", "")
    
    dist = editdistance.eval(aa, bb)
    return dist


def dtw(series_1: List[str], 
        series_2: List[str], 
        series1_factor: Optional[List] = None,
        series2_factor: Optional[List] = None, 
        norm_func=None) -> Tuple[List[Tuple[int, int]], float, List, List, np.ndarray]:

    if norm_func is None:
        norm_func = cost_fn
        
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    
    if series1_factor is not None and series2_factor is not None:
        for i, (vec1, fc1) in enumerate(zip(series_1, series1_factor)): 
            for j, (vec2, fc2) in enumerate(zip(series_2, series2_factor)):
                cost = norm_func(vec1, vec2) * fc1 * fc2 
                matrix[i + 1, j + 1] = cost + min(
                    matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
                )
    else:
        for i, vec1 in enumerate(series_1): 
            for j, vec2 in enumerate(series_2):
                cost = norm_func(vec1, vec2)
                matrix[i + 1, j + 1] = cost + min(
                    matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
                )

    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1 
    j = matrix.shape[1] - 1
    
    matches = []
    mappings_series_1 = [list() for _ in range(matrix.shape[0])]
    mappings_series_2 = [list() for _ in range(matrix.shape[1])]
    
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
            
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


def _normalize_token(t: str, marker: Optional[str] = None) -> str:
    markers = []
    if marker:
        markers.append(marker)
    markers += ['▁', 'Ġ', '##']
    
    for m in markers:
        t = t.replace(m, '')
    return t.lower()


def align_strict_one_to_one(
    base_vals: torch.Tensor,
    blend_vals: torch.Tensor,
    path: Sequence[Tuple[int, int]],
    base_tokens: List[str],
    blend_tokens: List[str],
    base_marker: str,
    blend_marker: str,
    specTok_mapper: Optional[Dict] = None,
    *,
    debug: bool = False,
    max_print: int = 20,
    dtw_matrix: Optional[np.ndarray] = None,
    dtw_crop: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if specTok_mapper is None:
        specTok_mapper = {}

    base_counts, blend_counts = {}, {}
    for i, j in path:
        base_counts[i] = base_counts.get(i, 0) + 1
        blend_counts[j] = blend_counts.get(j, 0) + 1

    base_norm = [_normalize_token(t, base_marker) for t in base_tokens]
    blend_norm = [_normalize_token(t, blend_marker) for t in blend_tokens]

    specTok_mapper_rev = {v: k for k, v in specTok_mapper.items()} if specTok_mapper else {}

    def _is_special_pair_ok(b_tok: str, s_tok: str) -> bool:
        if b_tok in specTok_mapper and specTok_mapper[b_tok] == s_tok:
            return True
        if s_tok in specTok_mapper_rev and specTok_mapper_rev[s_tok] == b_tok:
            return True
        return False

    one_to_one = [(i, j) for (i, j) in path
                  if base_counts.get(i, 0) == 1 and blend_counts.get(j, 0) == 1]

    kept_pairs, name_mismatch, multi_align = [], [], []
    for (i, j) in path:
        if base_counts.get(i, 0) != 1 or blend_counts.get(j, 0) != 1:
            if len(multi_align) < max_print:
                multi_align.append((i, j, base_tokens[i], blend_tokens[j],
                                    base_counts.get(i, 0), blend_counts.get(j, 0)))
            continue

        bi_raw, sj_raw = base_tokens[i], blend_tokens[j]
        if _is_special_pair_ok(bi_raw, sj_raw) or (base_norm[i] == blend_norm[j]):
            kept_pairs.append((i, j))
        else:
            if len(name_mismatch) < max_print:
                name_mismatch.append((i, j, bi_raw, sj_raw, base_norm[i], blend_norm[j]))

    if len(kept_pairs) == 0:
        A_base = base_vals.new_empty((0, base_vals.size(-1)))
        A_blend = blend_vals.new_empty((0, blend_vals.size(-1)))
    else:
        A_base = torch.stack([base_vals[i] for (i, j) in kept_pairs], dim=0)
        A_blend = torch.stack([blend_vals[j] for (i, j) in kept_pairs], dim=0)

    if debug:
        print("\n================= [ALIGN DEBUG] =================")
        print(f"L_base={base_vals.size(0)}, L_blend={blend_vals.size(0)}, |path|={len(path)}")
        print(f"1–1 candidates: {len(one_to_one)} / {len(path)}")
        print(f"Final kept (strict name match + special map): {len(kept_pairs)}")

        if multi_align:
            print(f"\n[Examples dropped for multi-align] (show up to {max_print})")
            for (i, j, braw, sraw, bc, sc) in multi_align[:max_print]:
                print(f"  (i={i}, j={j}) teacher='{braw}' student='{sraw}'  counts=({bc},{sc})")

        if name_mismatch:
            print(f"\n[Examples dropped for name mismatch after normalize] (show up to {max_print})")
            for (i, j, braw, sraw, bn, sn) in name_mismatch[:max_print]:
                print(f"  (i={i}, j={j}) teacher='{braw}'→'{bn}'  vs  student='{sraw}'→'{sn}'")

        if len(kept_pairs) > 0:
            print(f"\n[First kept pairs] (up to {max_print}):")
            for (i, j) in kept_pairs[:max_print]:
                print(f"  (i={i}, j={j})  '{base_tokens[i]}' ↔ '{blend_tokens[j]}'  "
                      f"norm='{base_norm[i]}' ↔ '{blend_norm[j]}'")

        print(f"\nAligned 1–1 shapes: A_t={tuple(A_base.shape)}, A_s={tuple(A_blend.shape)}")

        if dtw_matrix is not None:
            H, W = dtw_matrix.shape
            h, w = min(dtw_crop, H), min(dtw_crop, W)
            print(f"\n[DTW matrix] shape={dtw_matrix.shape}  (show {h}x{w} top-left)")
            print(np.array2string(dtw_matrix[:h, :w], precision=2, suppress_small=True))
        print("=================================================\n")

    return A_base, A_blend


def align_by_path_pool_many(
    base_vals: torch.Tensor,
    blend_vals: torch.Tensor,
    path: List[Tuple[int, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    A_base, A_blend = [], []
    k = 0
    P = len(path)
    
    while k < P:
        i0, j0 = path[k]
        if k == P - 1:
            A_base.append(base_vals[i0])
            A_blend.append(blend_vals[j0])
            break

        i1, j1 = path[k+1]
        di, dj = i1 - i0, j1 - j0

        # Many teacher -> one student
        if dj == 0 and di == 1:
            i_run = [i0]
            j_fix = j0
            kk = k + 1
            while kk < P and path[kk][1] == j_fix and path[kk][0] == i_run[-1] + 1:
                i_run.append(path[kk][0])
                kk += 1
            A_base.append(base_vals[i_run].mean(dim=0))
            A_blend.append(blend_vals[j_fix])
            k = kk
            continue

        if di == 0 and dj == 1:
            j_run = [j0]
            i_fix = i0
            kk = k + 1
            while kk < P and path[kk][0] == i_fix and path[kk][1] == j_run[-1] + 1:
                j_run.append(path[kk][1])
                kk += 1
            A_base.append(base_vals[i_fix])
            A_blend.append(blend_vals[j_run].mean(dim=0))
            k = kk
            continue

        # Diagonal / direction change: 1-1
        A_base.append(base_vals[i0])
        A_blend.append(blend_vals[j0])
        k += 1

    A_base = torch.stack(A_base, dim=0)
    A_blend = torch.stack(A_blend, dim=0)
    return A_base, A_blend


class ContextualDynamicMapping:
    
    def __init__(
        self,
        tok_student,
        tok_teacher,
        blending_model_special_token: str = "G",
        base_model_special_token: str = "##",
        w_task: float = 0.5,
        alpha_dtw: float = 0.5,
        debug_align: bool = False
    ):
        self.tok_student = tok_student
        self.tok_teacher = tok_teacher
        self.blending_model_special_token = blending_model_special_token
        self.base_model_special_token = base_model_special_token
        self.w_task = w_task
        self.alpha_dtw = alpha_dtw
        self.debug_align = debug_align
        
        self.specTok_mapper = {}
        if tok_student.cls_token and tok_teacher.bos_token:
            self.specTok_mapper[tok_student.cls_token] = tok_teacher.bos_token
        if tok_student.sep_token and tok_teacher.eos_token:
            self.specTok_mapper[tok_student.sep_token] = tok_teacher.eos_token
        if tok_student.pad_token and tok_teacher.pad_token:
            self.specTok_mapper[tok_student.pad_token] = tok_teacher.pad_token
        if tok_student.unk_token and tok_teacher.unk_token:
            self.specTok_mapper[tok_student.unk_token] = tok_teacher.unk_token
        if tok_student.mask_token and tok_teacher.mask_token:
            self.specTok_mapper[tok_student.mask_token] = tok_teacher.mask_token
    
    def compute_cdm_loss(
        self,
        S_last: torch.Tensor,
        T_last: torch.Tensor,
        batch_input_ids_stu: torch.Tensor,
        batch_input_ids_tea: torch.Tensor,
        keep_mask_stu: torch.Tensor,
        keep_mask_tea: torch.Tensor,
        proj_s2t: nn.Module,
        device_s: torch.device,
        epoch: int = 0,
        step: int = 0
    ) -> torch.Tensor:
        kd_sum, denom = 0.0, 0
        base_dtype = S_last.dtype
        Bsz = S_last.size(0)
        
        for i in range(Bsz):
            # Get tokens
            stu_tok_full = self.tok_student.convert_ids_to_tokens(
                batch_input_ids_stu[i].cpu().tolist(), 
                skip_special_tokens=False
            )
            tea_tok_full = self.tok_teacher.convert_ids_to_tokens(
                batch_input_ids_tea[i].cpu().tolist(), 
                skip_special_tokens=False
            )
            
            s_tok_i = [t for t, m in zip(stu_tok_full, keep_mask_stu[i].detach().cpu().tolist()) if m]
            t_tok_i = [t for t, m in zip(tea_tok_full, keep_mask_tea[i].detach().cpu().tolist()) if m]
            
            Si = S_last[i][keep_mask_stu[i]]  # [Ns_i, d_s]
            Ti = T_last[i][keep_mask_tea[i]]  # [Nt_i, d_t]
            
            if Si.numel() > 0 and Ti.numel() > 0 and len(s_tok_i) > 0 and len(t_tok_i) > 0:
                
                matches, dtw_cost, _, _, dtw_mat = dtw(
                    series_1=t_tok_i,
                    series_2=s_tok_i,
                    norm_func=lambda a, b: cost_fn(
                        a, b, 
                        self.blending_model_special_token,
                        self.base_model_special_token,
                        self.specTok_mapper
                    )
                )
                
                debug_here = self.debug_align and (epoch == 0) and (step < 1) and (i < 1)
                
                A_t, A_s = align_strict_one_to_one(
                    base_vals=Ti,
                    blend_vals=Si,
                    base_tokens=t_tok_i,
                    blend_tokens=s_tok_i,
                    base_marker=self.base_model_special_token,
                    blend_marker=self.blending_model_special_token,
                    specTok_mapper=self.specTok_mapper,
                    path=matches,
                    debug=debug_here,
                    dtw_matrix=dtw_mat,
                    dtw_crop=12
                )
                
                if A_t.size(0) > 0:
                    S_proj_tok = proj_s2t(A_s).to(base_dtype)
                    A_t = A_t.to(base_dtype)
                    S_proj_tok = F.normalize(S_proj_tok, p=2, dim=-1)
                    A_t = F.normalize(A_t, p=2, dim=-1)
                    kd_sum += F.mse_loss(S_proj_tok, A_t, reduction="sum")
                    denom += A_t.numel()
                    del S_proj_tok, A_t, A_s
        
        if denom == 0:
            return torch.tensor(0.0, device=device_s, dtype=base_dtype)
        else:
            loss = (kd_sum / denom).to(device=device_s, dtype=base_dtype)
            return loss
