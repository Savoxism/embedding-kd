import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
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
    normalize: bool = False
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
            
            with autocast(enabled=use_amp and torch.cuda.is_available()):
                t_out1 = model_teacher(
                    input_ids=batch_t["input_ids1_tea"],
                    attention_mask=batch_t["attention_mask1_tea"],
                    return_dict=True
                )
                T_last1 = t_out1.last_hidden_state  # [B, L, d_t]
                
                if pooling_method == "last_token":
                    T_cls1 = last_token_pool(T_last1, batch_t["attention_mask1_tea"])
                elif pooling_method == "mean":
                    T_cls1 = mean_pooling(T_last1, batch_t["attention_mask1_tea"])
                elif pooling_method == "cls":
                    T_cls1 = T_last1[:, 0, :]  # CLS token
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


def clear_cache_and_free_memory():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print("Done clearing GPU cache and freeing memory")
