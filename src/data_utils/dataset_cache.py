import torch
from torch.utils.data import Dataset
from typing import Dict
import pandas as pd


class TextPairWithTeacherAndHeatGeo(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        teacher_cls: torch.Tensor,
        heatgeo_artifact: Dict[str, torch.Tensor],
    ):
        self.task = task
        self.teacher_cls = teacher_cls
        self.candidate_indices = heatgeo_artifact["candidate_indices"]
        self.teacher_probs = heatgeo_artifact["teacher_probs"]
        self.spectral_coords = heatgeo_artifact["spectral_coords"]
        self.walk_indices = heatgeo_artifact.get("walk_indices", None)
        self.hard_neg_indices = heatgeo_artifact.get("hard_neg_indices", None)

        if task == "single_cls":
            self.samples = [(t, int(y)) for t, y in zip(df["text"].astype(str),
                                                        df["label"].astype(int))]
        elif task == "pair_cls":
            self.samples = [(a, b) for a, b in zip(df["premise"].astype(str),
                                                   df["hypothesis"].astype(str))]
        else:
            self.samples = [(a, b) for a, b in zip(df["sentence1"].astype(str),
                                                   df["sentence2"].astype(str))]

    def __len__(self):
        return len(self.samples)

    def _anchor_text(self, sample):
        if self.task == "single_cls":
            return sample[0]
        return sample[0]

    def __getitem__(self, idx):
        candidate_idx = self.candidate_indices[idx]
        candidate_samples = [self._anchor_text(self.samples[int(j)]) for j in candidate_idx]
        out = {
            "idx": idx,
            "sample": self.samples[idx],
            "teacher_cls": self.teacher_cls[idx],
            "candidate_idx": candidate_idx,
            "candidate_texts": candidate_samples,
            "teacher_probs": self.teacher_probs[:, idx, :],
            "spectral_target": self.spectral_coords[idx],
        }
        if self.walk_indices is not None:
            out["walk_indices"] = self.walk_indices[idx]
            out["hard_neg_indices"] = self.hard_neg_indices[idx]
        return out


class HeatGeoCollate:
    def __init__(self, tok_student, task: str, max_len: int):
        self.ts = tok_student
        self.task = task
        self.max_len = max_len

    def __call__(self, batch):
        samples = [item["sample"] for item in batch]
        teacher_cls = torch.stack([item["teacher_cls"] for item in batch], dim=0)
        idx = torch.tensor([item["idx"] for item in batch], dtype=torch.long)
        candidate_idx = torch.stack([item["candidate_idx"] for item in batch], dim=0).long()
        teacher_probs = torch.stack([item["teacher_probs"] for item in batch], dim=0).float()
        spectral_target = torch.stack([item["spectral_target"] for item in batch], dim=0).float()
        candidate_texts_nested = [item["candidate_texts"] for item in batch]
        candidate_texts = [text for texts in candidate_texts_nested for text in texts]

        if self.task == "single_cls":
            s1s = [sample[0] for sample in samples]
            ys = [sample[1] for sample in samples]
            s2s = s1s
        else:
            s1s, s2s = zip(*samples)
            s1s = list(s1s)
            s2s = list(s2s)

        s1_enc = self.ts(s1s, max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)
        s2_enc = self.ts(s2s, max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)
        cand_enc = self.ts(candidate_texts, max_length=self.max_len, truncation=True,
                           padding=True, return_tensors="pt",
                           return_special_tokens_mask=True)

        out = {
            "idx": idx,
            "candidate_idx": candidate_idx,
            "input_ids1_stu": s1_enc["input_ids"],
            "attention_mask1_stu": s1_enc["attention_mask"],
            "special_tokens_mask1_stu": s1_enc["special_tokens_mask"],
            "input_ids2_stu": s2_enc["input_ids"],
            "attention_mask2_stu": s2_enc["attention_mask"],
            "special_tokens_mask2_stu": s2_enc["special_tokens_mask"],
            "candidate_input_ids_stu": cand_enc["input_ids"],
            "candidate_attention_mask_stu": cand_enc["attention_mask"],
            "candidate_special_tokens_mask_stu": cand_enc["special_tokens_mask"],
            "teacher_cls": teacher_cls,
            "teacher_probs": teacher_probs,
            "spectral_target": spectral_target,
        }
        
        if "walk_indices" in batch[0]:
            out["walk_indices"] = torch.stack([item["walk_indices"] for item in batch], dim=0).long()
            out["hard_neg_indices"] = torch.stack([item["hard_neg_indices"] for item in batch], dim=0).long()

        if self.task == "single_cls":
            out["labels"] = torch.tensor(ys, dtype=torch.long)
        if "token_type_ids" in s1_enc:
            out["token_type_ids1_stu"] = s1_enc["token_type_ids"]
        if "token_type_ids" in s2_enc:
            out["token_type_ids2_stu"] = s2_enc["token_type_ids"]
        if "token_type_ids" in cand_enc:
            out["candidate_token_type_ids_stu"] = cand_enc["token_type_ids"]

        return out
