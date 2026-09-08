"""Datasets and collates for methods whose teacher embeddings are cached."""

import pandas as pd
import torch
from torch.utils.data import Dataset


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
            s_enc = self.ts(
                list(s1s),
                max_length=self.max_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            # Numbered exactly like the pair branch. These used to be
            # `input_ids_stu` and friends, which no consumer reads: the rkd and
            # talas steps and the training loop's batch-size probe all index
            # `input_ids1_stu`, so `--task_type single_cls` raised KeyError on
            # the first batch for every cached-teacher method.
            out = {
                "input_ids1_stu": s_enc["input_ids"],
                "attention_mask1_stu": s_enc["attention_mask"],
                "special_tokens_mask1_stu": s_enc["special_tokens_mask"],
                "teacher_cls": teacher_cls,
                "labels": torch.tensor(ys, dtype=torch.long),
            }
            if "token_type_ids" in s_enc:
                out["token_type_ids1_stu"] = s_enc["token_type_ids"]
            return out

        # ---------- pair ----------
        s1s, s2s = zip(*samples)

        s1_enc = self.ts(
            list(s1s),
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        s2_enc = self.ts(
            list(s2s),
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

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
        self.teacher_cls = teacher_cls  # [N, d_t]

        if task == "single_cls":
            self.samples = [
                (t, int(y))
                for t, y in zip(df["text"].astype(str), df["label"].astype(int))
            ]
        elif task == "pair_cls":
            self.samples = list(
                zip(df["premise"].astype(str), df["hypothesis"].astype(str))
            )
        else:
            self.samples = list(
                zip(df["sentence1"].astype(str), df["sentence2"].astype(str))
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tcls = self.teacher_cls[idx]
        return item, tcls
