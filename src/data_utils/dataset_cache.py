import numpy as np
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
        anchor_texts: list[str],
        teacher_cls: torch.Tensor,
        sampler,
        labels: list[int] | None = None,
    ):
        """
        teacher_cls is [N, Num_Layers, Dim] or [N, Dim]
        """
        self.anchor_texts = anchor_texts
        self.teacher_cls = teacher_cls
        self.sampler = sampler
        self.labels = labels

    def __len__(self):
        return len(self.anchor_texts)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __getitem__(self, idx):
        # Only corpus indices cross the worker boundary. Carrying the candidate
        # *texts* here shipped candidate_size strings per item through shared
        # memory and re-tokenized every one of them; HeatGeoCollate holds the
        # corpus tokenized once and looks them up by index instead.
        candidate_idx, teacher_probs = self.sampler.sample_torch(idx)
        item = {
            "idx": idx,
            "candidate_idx": candidate_idx,
            "teacher_probs": teacher_probs,
        }
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


class HeatGeoCollate:
    """Deduplicated, length-bucketed candidate batches.

    Three things drive the cost of a HeatGeo step, and none of them are the loss:

    * **Duplicate encodes.** In-batch sharing makes the criterion run
      ``torch.unique`` over the flattened candidate indices and keeps exactly one
      representative embedding per corpus index -- every other copy is encoded and
      then thrown away. Deduplicating here instead makes that waste impossible.
    * **Padding.** ``padding=True`` over the whole ``batch_size * candidate_size``
      block pads every sequence to the longest one in it. On a corpus whose median
      length is 13 tokens, the longest of ~2000 draws is ~70, so most of the
      student's FLOPs went into pad tokens. Sorting by length and encoding in
      chunks cuts the padded token count by ~3x; ``candidate_inverse`` maps the
      flattened ``[B, C]`` layout onto the encode order so the criterion sees
      exactly the tensor it saw before.
    * **Re-tokenization.** Each corpus text is drawn as a candidate ~candidate_size
      times per epoch and was tokenized every single time. The corpus is tokenized
      once in ``__init__`` and looked up by index afterwards.

    Only the anchor text is encoded on the anchor side: the objective scores
    anchor-vs-candidates, so the second view of the pair has no consumer.

    Deduplication also makes repeated candidates share one forward pass (hence one
    dropout draw); this is the canonical path rather than an optional mode.
    """

    def __init__(
        self,
        tok_student,
        task: str,
        max_len: int,
        corpus_texts: list[str],
        encode_chunk_size: int = 256,
        pad_to_multiple_of: int = 8,
    ):
        self.ts = tok_student
        self.task = task
        self.max_len = max_len
        self.encode_chunk_size = int(encode_chunk_size)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

        self.pad_id = tok_student.pad_token_id
        if self.pad_id is None:
            raise ValueError("student tokenizer has no pad_token_id")

        encoded = tok_student(
            [str(text) for text in corpus_texts],
            max_length=max_len,
            truncation=True,
        )["input_ids"]
        self.corpus_ids = [np.asarray(ids, dtype=np.int64) for ids in encoded]
        self.corpus_len = np.asarray(
            [len(ids) for ids in self.corpus_ids], dtype=np.int64
        )

    def _pad(self, nodes) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.corpus_ids[int(node)] for node in nodes]
        width = max(len(ids) for ids in rows)
        if self.pad_to_multiple_of > 1:
            multiple = self.pad_to_multiple_of
            width = ((width + multiple - 1) // multiple) * multiple

        input_ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=np.int64)
        for row, ids in enumerate(rows):
            input_ids[row, : ids.size] = ids
            attention_mask[row, : ids.size] = 1
        return torch.from_numpy(input_ids), torch.from_numpy(attention_mask)

    def __call__(self, batch):
        idx = torch.tensor([item["idx"] for item in batch], dtype=torch.long)
        candidate_idx = torch.stack(
            [item["candidate_idx"] for item in batch], dim=0
        ).long()
        teacher_probs = torch.stack(
            [item["teacher_probs"] for item in batch], dim=0
        ).float()
        ys = [item["label"] for item in batch] if "label" in batch[0] else None

        unique_idx, inverse = torch.unique(candidate_idx.reshape(-1), return_inverse=True)
        unique_nodes = unique_idx.numpy()

        # Encode short sequences together so a chunk is padded to its own longest
        # member rather than the batch's. `rank` sends each unique candidate to its
        # row in the concatenated encode output.
        order = np.argsort(self.corpus_len[unique_nodes], kind="stable")
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size, dtype=np.int64)
        candidate_inverse = torch.from_numpy(rank)[inverse]

        sorted_nodes = unique_nodes[order]
        candidate_chunks = []
        for start in range(0, sorted_nodes.size, self.encode_chunk_size):
            chunk_ids, chunk_mask = self._pad(sorted_nodes[start : start + self.encode_chunk_size])
            candidate_chunks.append(
                {"input_ids": chunk_ids, "attention_mask": chunk_mask}
            )

        anchor_ids, anchor_mask = self._pad(idx.tolist())

        out = {
            "idx": idx,
            "candidate_idx": candidate_idx,
            "input_ids1_stu": anchor_ids,
            "attention_mask1_stu": anchor_mask,
            "candidate_chunks": candidate_chunks,
            "candidate_inverse": candidate_inverse,
            "teacher_probs": teacher_probs,
        }
        if ys is not None:
            out["labels"] = torch.tensor(ys, dtype=torch.long)
        return out
