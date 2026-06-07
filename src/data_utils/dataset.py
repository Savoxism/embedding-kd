import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class TextPairRaw(Dataset):
    def __init__(self, df: pd.DataFrame, task: str):
        self.task = task
        if task == "single_cls":
            self.samples = [(t, None, int(y)) for t, y in zip(df["text"].astype(str), df["label"].astype(int))]
        elif task == "pair_cls":
            self.samples = [(a, b) for a,b in zip(df["premise"].astype(str),
                                                            df["hypothesis"].astype(str))]
        else:  # pair_reg
            self.samples = [(a, b) for a,b in zip(df["sentence1"].astype(str),
                                                              df["sentence2"].astype(str))]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx] 
from typing import List, Tuple, Optional



class DualTokenizerCollate:
    def __init__(self, tok_student, tok_teacher, task: str, max_len: int):
        self.ts = tok_student
        self.tt = tok_teacher
        self.task = task
        self.max_len = max_len

    def __call__(self, batch: List[Tuple[str, Optional[str], float]]):
        s1s, s2s = zip(*batch)

        if self.task == "single_cls":
            s_enc = self.ts(list(s1s), max_length=self.max_len, truncation=True,
                            padding=True, return_tensors="pt",
                            return_special_tokens_mask=True)
            t_enc = self.tt(list(s1s), max_length=self.max_len, truncation=True,
                            padding=True, return_tensors="pt",
                            return_special_tokens_mask=True)

            out = {
                "input_ids_stu": s_enc["input_ids"],
                "attention_mask_stu": s_enc["attention_mask"],
                "special_tokens_mask_stu": s_enc["special_tokens_mask"],
                "input_ids_tea": t_enc["input_ids"],
                "attention_mask_tea": t_enc["attention_mask"],
                "special_tokens_mask_tea": t_enc["special_tokens_mask"],
                "labels": torch.tensor(ys, dtype=torch.long),
            }
            if "token_type_ids" in s_enc:
                out["token_type_ids_stu"] = s_enc["token_type_ids"]
            if "token_type_ids" in t_enc:
                out["token_type_ids_tea"] = t_enc["token_type_ids"]
            return out

        # ------- pair (bi-encoder) -------
        s1_enc = self.ts(list(s1s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)
        s2_enc = self.ts(list(s2s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)

        t1_enc = self.tt(list(s1s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)
        t2_enc = self.tt(list(s2s), max_length=self.max_len, truncation=True,
                         padding=True, return_tensors="pt",
                         return_special_tokens_mask=True)

        out = {
            # student
            "input_ids1_stu": s1_enc["input_ids"],
            "attention_mask1_stu": s1_enc["attention_mask"],
            "special_tokens_mask1_stu": s1_enc["special_tokens_mask"],
            "input_ids2_stu": s2_enc["input_ids"],
            "attention_mask2_stu": s2_enc["attention_mask"],
            "special_tokens_mask2_stu": s2_enc["special_tokens_mask"],
            # teacher
            "input_ids1_tea": t1_enc["input_ids"],
            "attention_mask1_tea": t1_enc["attention_mask"],
            "special_tokens_mask1_tea": t1_enc["special_tokens_mask"],
            "input_ids2_tea": t2_enc["input_ids"],
            "attention_mask2_tea": t2_enc["attention_mask"],
            "special_tokens_mask2_tea": t2_enc["special_tokens_mask"],
        }
        # chỉ thêm token_type_ids nếu tồn tại
        if "token_type_ids" in s1_enc:
            out["token_type_ids1_stu"] = s1_enc["token_type_ids"]
        if "token_type_ids" in s2_enc:
            out["token_type_ids2_stu"] = s2_enc["token_type_ids"]
        if "token_type_ids" in t1_enc:
            out["token_type_ids1_tea"] = t1_enc["token_type_ids"]
        if "token_type_ids" in t2_enc:
            out["token_type_ids2_tea"] = t2_enc["token_type_ids"]

        return out