import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
def info_nce(q, k, temperature=0.1, neg_valid_mask=None):
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    logits = torch.matmul(q, k.T) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss_inbatch = F.cross_entropy(logits, labels) 
    return loss_inbatch, logits

def cosine_embedding_loss(  # *10 (cosine loss)
    student_embeddings, # [batch_size,dim]
    teacher_embeddings, # [batch_size,dim]
):

    # normalization
    student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
    teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)
    # get cosine loss
    target = torch.ones(student_embeddings.size(0), device=student_embeddings.device)
    loss = F.cosine_embedding_loss(student_embeddings, teacher_embeddings, target)
    return loss

def pair_inbatch_similarity_loss( # *200 (sim loss)
    student_embeddings, # [batch_size,dim]
    teacher_embeddings, # [batch_size,dim]
):

    student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
    teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)
    # get mse loss
    #[batch_size,batch_size]<- [batch_size,dim],[dim,batch_size]
    student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)
    teacher_similarity = teacher_embeddings @ teacher_embeddings.transpose(-1, -2)

    loss = F.mse_loss(student_similarity, teacher_similarity)
    return loss

def pair_inbatch_triplet_loss( # *20
    student_embeddings, # [batch_size,dim]
    teacher_embeddings,
    triplet_margin=0.015,
):
    student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
    teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)
    triplet_label = torch.where(get_score_diff(teacher_embeddings) < 0, 1, -1)
    # get triplets loss
    loss = F.relu(get_score_diff(student_embeddings) * triplet_label + triplet_margin).mean()

    return loss

def get_score_diff(
    embedding
):
    scores = torch.matmul(embedding, embedding.T)
    scores = scores[torch.triu(torch.ones_like(scores), diagonal=1).bool()]
    score_diff = scores.reshape((1, -1)) - scores.reshape((-1, 1))
    score_diff = score_diff[torch.triu(torch.ones_like(score_diff), diagonal=1).bool()]
    return score_diff

def compute_variance(domain_loss_list: List[torch.Tensor]) -> torch.Tensor:
    loss_variance = 0.0
    for i, loss_i in enumerate(domain_loss_list):
        for j, loss_j in enumerate(domain_loss_list):
            loss_variance += (loss_i - loss_j) ** 2
    loss_variance /= (2 * len(domain_loss_list) ** 2)
    return loss_variance
