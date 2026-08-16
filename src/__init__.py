# Code utilities for Knowledge Distillation
from .loss import (
    info_nce,
    cosine_embedding_loss,
    pair_inbatch_similarity_loss,
    pair_inbatch_triplet_loss,
    get_score_diff,
    compute_variance
)
from .pooling import last_token_pool, mean_pooling
from .cache_teacher import (
    cache_teacher_embeddings,
    load_cached_embeddings,
    clear_cache_and_free_memory
)

__all__ = [
    'info_nce',
    'cosine_embedding_loss',
    'pair_inbatch_similarity_loss',
    'pair_inbatch_triplet_loss',
    'get_score_diff',
    'last_token_pool',
    'mean_pooling',
    'cache_teacher_embeddings',
    'load_cached_embeddings',
    'clear_cache_and_free_memory',
    'compute_variance'
]
