from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class TextPairRaw(Dataset):
    """Canonical raw-text dataset used by online teacher/student methods."""

    def __init__(self, df: pd.DataFrame, task: str):
        self.task = task
        if task == "single_cls":
            if "text" not in df.columns or "label" not in df.columns:
                raise ValueError("single_cls data requires 'text' and 'label' columns")
            self.samples = [
                (text, None, int(label))
                for text, label in zip(df["text"].astype(str), df["label"].astype(int))
            ]
        elif task == "pair_cls":
            if "premise" not in df.columns or "hypothesis" not in df.columns:
                raise ValueError("pair_cls data requires 'premise' and 'hypothesis' columns")
            labels = df["label"].astype(int).tolist() if "label" in df.columns else [None] * len(df)
            self.samples = [
                (premise, hypothesis, label)
                for premise, hypothesis, label in zip(
                    df["premise"].astype(str),
                    df["hypothesis"].astype(str),
                    labels,
                )
            ]
        elif task == "pair_reg":
            if "sentence1" not in df.columns or "sentence2" not in df.columns:
                raise ValueError("pair_reg data requires 'sentence1' and 'sentence2' columns")
            score_column = "score" if "score" in df.columns else "label"
            if score_column not in df.columns:
                raise ValueError("pair_reg data requires a 'score' or 'label' column")
            self.samples = [
                (sentence1, sentence2, float(score))
                for sentence1, sentence2, score in zip(
                    df["sentence1"].astype(str),
                    df["sentence2"].astype(str),
                    df[score_column].astype(float),
                )
            ]
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, Optional[str], Optional[float]]:
        return self.samples[idx]


class DualTokenizerCollate:
    def __init__(
        self,
        tok_student,
        tok_teacher,
        task: str,
        max_len: int,
    ):
        self.ts = tok_student
        self.tt = tok_teacher
        self.task = task
        self.max_len = max_len

    def _encode(self, tokenizer, texts: List[str]):
        return tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

    @staticmethod
    def _add_encoding(out, encoding, side: int, model_suffix: str):
        key_suffix = f"{side}_{model_suffix}"
        out[f"input_ids{key_suffix}"] = encoding["input_ids"]
        out[f"attention_mask{key_suffix}"] = encoding["attention_mask"]
        out[f"special_tokens_mask{key_suffix}"] = encoding["special_tokens_mask"]
        if "token_type_ids" in encoding:
            out[f"token_type_ids{key_suffix}"] = encoding["token_type_ids"]

    def __call__(self, batch: List[Tuple[str, Optional[str], Optional[float]]]):
        text1s = [sample[0] for sample in batch]
        text2s = [sample[1] for sample in batch]
        labels = [sample[2] for sample in batch]

        student1 = self._encode(self.ts, text1s)
        teacher1 = self._encode(self.tt, text1s)
        out = {}
        self._add_encoding(out, student1, 1, "stu")
        self._add_encoding(out, teacher1, 1, "tea")

        has_second_text = all(text is not None for text in text2s)
        if any(text is not None for text in text2s) and not has_second_text:
            raise ValueError("A batch cannot mix single-text and pair-text samples")
        if has_second_text:
            pair_texts = [str(text) for text in text2s]
            student2 = self._encode(self.ts, pair_texts)
            teacher2 = self._encode(self.tt, pair_texts)
            self._add_encoding(out, student2, 2, "stu")
            self._add_encoding(out, teacher2, 2, "tea")

        has_labels = all(label is not None for label in labels)
        if any(label is not None for label in labels) and not has_labels:
            raise ValueError("A batch cannot mix labeled and unlabeled samples")
        if has_labels:
            dtype = torch.float32 if self.task == "pair_reg" else torch.long
            out["labels"] = torch.tensor(labels, dtype=dtype)
        return out
