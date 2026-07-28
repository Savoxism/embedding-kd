import torch
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pandas as pd
import numpy as np
class DualTokenizerCollateWithTeacher:
    def __init__(self, tok_student, task: str, max_len: int):
        self.ts = tok_student
        self.task = task
        self.max_len = max_len

    def __call__(self, batch):
        samples, teacher_cls = zip(*batch)
        teacher_cls = torch.stack(teacher_cls, dim=0)  # [B, d_t]

        if self.task == "single_cls":
            s1s, ys = zip(*samples)
            s_enc = self.ts(list(s1s), max_length=self.max_len, truncation=True,
                            padding=True, return_tensors="pt",
                            return_special_tokens_mask=True)
            out = {
                "input_ids_stu": s_enc["input_ids"],
                "attention_mask_stu": s_enc["attention_mask"],
                "special_tokens_mask_stu": s_enc["special_tokens_mask"],
                "teacher_cls": teacher_cls,
                "labels": torch.tensor(ys, dtype=torch.long),
            }
            if "token_type_ids" in s_enc:
                out["token_type_ids_stu"] = s_enc["token_type_ids"]
            return out

        # ---------- pair ----------
        s1s, s2s = zip(*samples)

        s1_enc = self.ts(list(s1s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)
        s2_enc = self.ts(list(s2s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)

        out = {
            "input_ids1_stu": s1_enc["input_ids"],
            "attention_mask1_stu": s1_enc["attention_mask"],
            "special_tokens_mask1_stu": s1_enc["special_tokens_mask"],
            "input_ids2_stu": s2_enc["input_ids"],
            "attention_mask2_stu": s2_enc["attention_mask"],
            "special_tokens_mask2_stu": s2_enc["special_tokens_mask"],
            "teacher_cls": teacher_cls,
        }

        if "token_type_ids" in s1_enc:
            out["token_type_ids1_stu"] = s1_enc["token_type_ids"]
        if "token_type_ids" in s2_enc:
            out["token_type_ids2_stu"] = s2_enc["token_type_ids"]

        return out
    
class TextPairWithTeacher(Dataset):
    def __init__(self, df: pd.DataFrame, task: str, teacher_cls: torch.Tensor):
        self.task = task
        self.teacher_cls = teacher_cls   # [N, d_t]

        if task == "single_cls":
            self.samples = [(t, int(y)) for t, y in zip(df["text"].astype(str),
                                                        df["label"].astype(int))]
        elif task == "pair_cls":
            self.samples = [(a, b) for a,b in zip(df["premise"].astype(str),
                                                  df["hypothesis"].astype(str))]
        else:
            self.samples = [(a, b) for a,b in zip(df["sentence1"].astype(str),
                                                  df["sentence2"].astype(str))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tcls = self.teacher_cls[idx]   # lấy đúng teacher CLS của sample này
        return item, tcls


class TextPairWithTeacherAndHeatGeo(Dataset):
    """HeatGeo anchors with a candidate set redrawn every epoch.

    `anchor_texts` is passed in explicitly rather than re-derived from the frame:
    the teacher graph is built over exactly these strings, and silently pulling a
    different column here (e.g. `premise` when the teacher encoded `text`) would
    misalign every node in the graph without raising anything.
    """

    def __init__(
        self,
        anchor_texts: List[str],
        teacher_cls: torch.Tensor,
        sampler,
        labels: Optional[List[int]] = None,
    ):
        self.anchor_texts = [str(text) for text in anchor_texts]
        self.teacher_cls = teacher_cls
        self.sampler = sampler
        self.labels = labels

        if len(self.anchor_texts) != int(teacher_cls.size(0)):
            raise ValueError(
                f"anchor_texts has {len(self.anchor_texts)} rows but teacher_cls has "
                f"{int(teacher_cls.size(0))}"
            )
        if labels is not None and len(labels) != len(self.anchor_texts):
            raise ValueError("labels length does not match anchor_texts length")

    def __len__(self):
        return len(self.anchor_texts)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __getitem__(self, idx):
        candidate_idx, teacher_probs = self.sampler.sample_torch(idx)
        candidate_texts = [self.anchor_texts[int(j)] for j in candidate_idx]
        item = {
            "idx": idx,
            "anchor_text": self.anchor_texts[idx],
            "teacher_cls": self.teacher_cls[idx],
            "candidate_idx": candidate_idx,
            "candidate_texts": candidate_texts,
            "teacher_probs": teacher_probs,
        }
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


class HeatGeoCollate:
    def __init__(self, tok_student, task: str, max_len: int):
        self.ts = tok_student
        self.task = task
        self.max_len = max_len

    def __call__(self, batch):
        teacher_cls = torch.stack([item["teacher_cls"] for item in batch], dim=0)
        idx = torch.tensor([item["idx"] for item in batch], dtype=torch.long)
        candidate_idx = torch.stack([item["candidate_idx"] for item in batch], dim=0).long()
        teacher_probs = torch.stack([item["teacher_probs"] for item in batch], dim=0).float()
        candidate_texts_nested = [item["candidate_texts"] for item in batch]
        candidate_texts = [text for texts in candidate_texts_nested for text in texts]

        # Only the anchor text is encoded: the objective scores anchor-vs-candidates,
        # so the second view of the pair has no consumer.
        s1s = [item["anchor_text"] for item in batch]
        ys = [item["label"] for item in batch] if "label" in batch[0] else None

        s1_enc = self.ts(s1s, max_length=self.max_len, truncation=True,
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
            "candidate_input_ids_stu": cand_enc["input_ids"],
            "candidate_attention_mask_stu": cand_enc["attention_mask"],
            "candidate_special_tokens_mask_stu": cand_enc["special_tokens_mask"],
            "teacher_cls": teacher_cls,
            "teacher_probs": teacher_probs,
        }

        if ys is not None:
            out["labels"] = torch.tensor(ys, dtype=torch.long)
        if "token_type_ids" in s1_enc:
            out["token_type_ids1_stu"] = s1_enc["token_type_ids"]
        if "token_type_ids" in cand_enc:
            out["candidate_token_type_ids_stu"] = cand_enc["token_type_ids"]

        return out
