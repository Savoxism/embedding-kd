import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer
from .pooling import last_token_pool, mean_pooling

def cache_teacher_embeddings(
    model_teacher: AutoModel,
    dataloader: DataLoader,
    device: torch.device,
    pooling_method: str = "last_token",
    cache_path: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = True,
    normalize: bool = False,
    kd_teacher_layers: Optional[list[int]] = None
) -> torch.Tensor:
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached teacher embeddings from: {cache_path}")
        cached_data = torch.load(cache_path, map_location="cpu")
        print(f"Done loading cached embeddings: {cached_data.shape}")
        return cached_data
    
    print("Pre-computing teacher embeddings...")
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)
    
    data_cls = []
    pbar = tqdm(dataloader, desc="Caching teacher CLS embeddings")
    
    with torch.inference_mode():
        for batch in pbar:
            batch_t = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                if k.endswith("_tea"):
                    batch_t[k] = v.to(device, non_blocking=True)
            
            with autocast(
                "cuda",
                enabled=use_amp and torch.cuda.is_available(),
            ):
                t_out1 = model_teacher(
                    input_ids=batch_t["input_ids1_tea"],
                    attention_mask=batch_t["attention_mask1_tea"],
                    return_dict=True,
                    output_hidden_states=kd_teacher_layers is not None
                )
                
                batch_layers = []
                if kd_teacher_layers is not None:
                    # Multi-layer extraction
                    for l_idx in kd_teacher_layers:
                        # hidden_states is a tuple of (embedding_output, layer_1, ..., layer_n)
                        # So index 0 is embeddings, 1 is layer 1, etc.
                        # Handle out-of-bounds indices for smaller models
                        idx = l_idx
                        if idx >= len(t_out1.hidden_states):
                            idx = len(t_out1.hidden_states) - 1
                        elif idx < -len(t_out1.hidden_states):
                            idx = 0
                        T_last1 = t_out1.hidden_states[idx]
                        if pooling_method == "last_token":
                            T_cls = last_token_pool(T_last1, batch_t["attention_mask1_tea"])
                        elif pooling_method == "mean":
                            T_cls = mean_pooling(T_last1, batch_t["attention_mask1_tea"])
                        elif pooling_method == "cls":
                            T_cls = T_last1[:, 0, :]
                        else:
                            raise ValueError(f"Unknown pooling method: {pooling_method}")
                        
                        if normalize:
                            T_cls = F.normalize(T_cls, p=2, dim=-1)
                        batch_layers.append(T_cls.to(dtype).unsqueeze(1))
                    
                    # [Batch, Num_Layers, Dim]
                    T_cls1 = torch.cat(batch_layers, dim=1)
                else:
                    # Single-layer fallback
                    T_last1 = t_out1.last_hidden_state  # [B, L, d_t]
                    if pooling_method == "last_token":
                        T_cls1 = last_token_pool(T_last1, batch_t["attention_mask1_tea"])
                    elif pooling_method == "mean":
                        T_cls1 = mean_pooling(T_last1, batch_t["attention_mask1_tea"])
                    elif pooling_method == "cls":
                        T_cls1 = T_last1[:, 0, :]
                    else:
                        raise ValueError(f"Unknown pooling method: {pooling_method}")
                    
                    if normalize:
                        T_cls1 = F.normalize(T_cls1, p=2, dim=-1)
                    T_cls1 = T_cls1.to(dtype)
            data_cls.append(T_cls1.cpu())
    teacher_cls_all = torch.cat(data_cls, dim=0)  
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
        torch.save(teacher_cls_all, cache_path)
        print(f"Saved cached teacher embeddings to: {cache_path}")
    
    print(f"Done caching teacher embeddings: {teacher_cls_all.shape}")
    return teacher_cls_all



def load_cached_embeddings(cache_path: str) -> torch.Tensor:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    print(f"Loading cached embeddings from: {cache_path}")
    embeddings = torch.load(cache_path, map_location="cpu")
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings


def validate_cached_embeddings(
    embeddings: torch.Tensor,
    expected_rows: int,
    *,
    cache_path: str = "<memory>",
    require_single_layer: bool = False,
) -> torch.Tensor:
    """Reject partial, stale, or numerically invalid teacher-cache tensors."""
    if not torch.is_tensor(embeddings):
        raise TypeError(
            f"Teacher cache {cache_path} must contain a tensor, got "
            f"{type(embeddings).__name__}"
        )
    allowed_dims = (2,) if require_single_layer else (2, 3)
    if embeddings.ndim not in allowed_dims:
        raise ValueError(
            f"Teacher cache {cache_path} has shape {tuple(embeddings.shape)}; "
            f"expected {'[N, D]' if require_single_layer else '[N, D] or [N, L, D]'}"
        )
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Teacher cache {cache_path} row mismatch: cache has "
            f"{embeddings.shape[0]} rows, data has {expected_rows}"
        )
    if embeddings.shape[-1] <= 0:
        raise ValueError(f"Teacher cache {cache_path} has an empty embedding dimension")
    if not embeddings.is_floating_point():
        raise TypeError(
            f"Teacher cache {cache_path} must be floating point, got {embeddings.dtype}"
        )
    if not bool(torch.isfinite(embeddings).all().item()):
        raise ValueError(f"Teacher cache {cache_path} contains NaN or Inf")
    return embeddings


def clear_cache_and_free_memory():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print("Done clearing GPU cache and freeing memory")
