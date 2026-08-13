import functools
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Evaluation is forward-only, so the batch that fits is much larger than the
# training batch. Overridable because the probe train sets (up to 24k rows) are
# the dominant cost and their optimal batch depends on the GPU.
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "256"))
# Tokenisation of a batch runs in a worker thread while the GPU consumes the
# previous one. Fast tokenizers release the GIL, so threads are enough here and
# avoid the per-DataLoader process fork that dominates on the small splits.
EVAL_PREFETCH = int(os.environ.get("EVAL_PREFETCH", "2"))


@functools.cache
def _read_eval_csv(file_path: str) -> pd.DataFrame:
    """Read an evaluation CSV once per process.

    Every split is re-read on every epoch's evaluation pass and the files never
    change during a run, so the parse is pure repeated work.
    """
    full_path = (
        BASE_DIR / file_path if not os.path.isabs(file_path) else Path(file_path)
    )
    return pd.read_csv(full_path)


def _column_as_list(frame: pd.DataFrame, column: str) -> list[str]:
    return frame[column].astype(str).tolist()


def _pooled_embedding(output):
    """Extract the sentence embedding from a student forward pass.

    Supports StellaModel (dict with 'pooled') and AutoModel (object or dict with
    last_hidden_state).
    """
    if isinstance(output, dict) and "pooled" in output:
        return output["pooled"]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    return output["last_hidden_state"][:, 0, :]


def _embed_texts(model, tokenizer, texts, max_len, desc=None):
    """Embed `texts` and return a float32 array in the original order.

    Batches are formed over length-sorted indices: with `padding=True` the cost
    of a batch is `len(batch) * max_len_in_batch`, so grouping similar lengths
    removes padding the model would otherwise attend over. Measured on
    tweet_train (23.7k rows, batch 64) this is 1.4x fewer padded tokens; the gap
    widens at batch 256. Padding is masked out, so the per-sequence result is
    unchanged.
    """
    device = model.device
    total = len(texts)
    order = sorted(range(total), key=lambda index: len(texts[index]))
    batches = [
        order[start : start + EVAL_BATCH_SIZE]
        for start in range(0, total, EVAL_BATCH_SIZE)
    ]

    def tokenize(indices):
        return tokenizer(
            [texts[index] for index in indices],
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )

    embeddings = None
    autocast = torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
    )

    with ThreadPoolExecutor(max_workers=1) as pool, autocast, torch.inference_mode():
        pending = deque()
        next_batch = 0
        while next_batch < len(batches) and len(pending) <= EVAL_PREFETCH:
            pending.append(
                (batches[next_batch], pool.submit(tokenize, batches[next_batch]))
            )
            next_batch += 1

        for _ in tqdm(range(len(batches)), desc=desc, leave=False):
            indices, future = pending.popleft()
            if next_batch < len(batches):
                pending.append(
                    (batches[next_batch], pool.submit(tokenize, batches[next_batch]))
                )
                next_batch += 1

            encoded = future.result()
            output = model(
                input_ids=encoded["input_ids"].to(device, non_blocking=True),
                attention_mask=encoded["attention_mask"].to(device, non_blocking=True),
            )
            batch_embeddings = _pooled_embedding(output).float().cpu().numpy()
            if embeddings is None:
                embeddings = np.empty(
                    (total, batch_embeddings.shape[1]), dtype=np.float32
                )
            embeddings[indices] = batch_embeddings

    if embeddings is None:
        return np.empty((0, 0), dtype=np.float32)
    return embeddings


def _cosine_similarity(left, right):
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left * right, axis=1)


class STSDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = _read_eval_csv(file_path)
        self.sentence1 = _column_as_list(self.dataset, "sentence1")
        self.sentence2 = _column_as_list(self.dataset, "sentence2")
        self.labels = self.dataset["score"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.sentence1)

    def __getitem__(self, idx):
        return {
            "sentence1": self.sentence1[idx],
            "sentence2": self.sentence2[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def collate_fn(batch, tokenizer, max_len=128):
    s1_list = [item["sentence1"] for item in batch]
    s2_list = [item["sentence2"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    enc1 = tokenizer(
        s1_list,
        truncation=True,
        padding=True,  # chỉ pad theo câu dài nhất trong batch
        max_length=max_len,
        return_tensors="pt",
    )
    enc2 = tokenizer(
        s2_list, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
    )

    return {
        "input_ids1": enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "input_ids2": enc2["input_ids"],
        "attention_mask2": enc2["attention_mask"],
        "labels": labels,
    }


def eval_sts(model, tokenizer, path):
    dataset = STSDataset(path)
    emb1 = _embed_texts(model, tokenizer, dataset.sentence1, 128, desc="sts s1")
    emb2 = _embed_texts(model, tokenizer, dataset.sentence2, 128, desc="sts s2")

    # scale [-1,1] -> [0,5]
    preds = (_cosine_similarity(emb1, emb2) + 1) * 2.5

    spearman_corr, _ = spearmanr(preds, dataset.labels)
    print(f"Spearman: {spearman_corr:.4f}")

    return spearman_corr


def eval_sts_task(model, path_list, tokenizer):
    model.eval()
    print(" eval_sts_task")
    results = {}
    for path in path_list:
        print(path)
        results[path] = eval_sts(model, tokenizer, path)
    model.train()
    return results


def eval_cls(model, tokenizer, path):
    dataset = ClasssifyDataset(path)
    features = _embed_texts(model, tokenizer, dataset.texts, 512, desc=Path(path).stem)
    return features, dataset.labels


class ClasssifyDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = _read_eval_csv(file_path)
        self.texts = _column_as_list(self.dataset, "text")
        self.labels = self.dataset["label"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _normalized_text_keys(dataset):
    return {" ".join(str(text).strip().casefold().split()) for text in dataset["text"]}


@functools.cache
def _validate_classification_pair(train_path, eval_path):
    """Check train/eval leakage once per (train, eval) pair per process.

    The files are fixed for the duration of a run, so re-normalising ~55k texts
    on every epoch's evaluation only repeats a verdict already reached.
    """
    eval_file = BASE_DIR / eval_path
    train_frame = _read_eval_csv(train_path)
    eval_frame = _read_eval_csv(eval_path)
    overlap = _normalized_text_keys(train_frame) & _normalized_text_keys(eval_frame)

    if "val_set" in Path(eval_path).parts:
        if overlap:
            raise ValueError(
                f"Classification train-validation leakage for {eval_file.stem}: "
                f"{len(overlap)} normalized texts overlap"
            )
        return

    dataset_name = eval_file.stem.removesuffix("_test")
    validation_path = f"data/val_set/{dataset_name}_validation.csv"
    validation_overlap = set()
    if (BASE_DIR / validation_path).is_file():
        validation_frame = _read_eval_csv(validation_path)
        validation_overlap = _normalized_text_keys(
            validation_frame
        ) & _normalized_text_keys(eval_frame)

    if overlap or validation_overlap:
        warnings.warn(
            f"Published test split {eval_file.stem} has normalized-text "
            f"overlap: train={len(overlap)}, validation={len(validation_overlap)}.",
            RuntimeWarning,
            stacklevel=2,
        )


def clf_collate_fn(batch, tokenizer, max_len=512):
    s1_list = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    enc1 = tokenizer(
        s1_list,
        truncation=True,
        padding=True,  # chỉ pad theo câu dài nhất trong batch
        max_length=max_len,
        return_tensors="pt",
    )

    return {
        "input_ids1": enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "labels": labels,
    }


def eval_classification_task(model, path_list, tokenizer):
    model.eval()
    print(" eval classifier")

    results = {}
    for train_path, dev_path in path_list:
        print(dev_path)
        _validate_classification_pair(train_path, dev_path)

        X_train, y_train = eval_cls(model, tokenizer, train_path)
        X_test, y_test = eval_cls(model, tokenizer, dev_path)

        clf = LogisticRegression(
            random_state=42,
            max_iter=1000,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        scores = {}
        accuracy = accuracy_score(y_test, y_pred)
        scores["accuracy"] = accuracy
        f1 = f1_score(y_test, y_pred, average="macro")
        scores["f1"] = f1
        print(scores)
        results[dev_path] = scores

    model.train()
    return results


class PairDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = _read_eval_csv(file_path)
        self.sentence1 = _column_as_list(self.dataset, "sentence1")
        self.sentence2 = _column_as_list(self.dataset, "sentence2")
        self.labels = self.dataset["label"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.sentence1)

    def __getitem__(self, idx):
        return {
            "sentence1": self.sentence1[idx],
            "sentence2": self.sentence2[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def eval_pair(model, tokenizer, path, threshold=None):
    dataset = PairDataset(path)
    emb1 = _embed_texts(model, tokenizer, dataset.sentence1, 128, desc="pair s1")
    emb2 = _embed_texts(model, tokenizer, dataset.sentence2, 128, desc="pair s2")

    preds = (_cosine_similarity(emb1, emb2) + 1) / 2

    metric = get_metric_pair_classification(preds, dataset.labels, threshold=threshold)
    print(metric)

    return metric


def get_metric_pair_classification(scores, labels, threshold=None):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if threshold is None:
        # Accuracy over a fixed threshold grid, evaluated as one vectorised sweep
        # instead of 200 separate sklearn calls.
        candidates = np.linspace(0, 1, 200)
        decisions = scores[None, :] >= candidates[:, None]
        accuracies = (decisions == labels[None, :].astype(bool)).mean(axis=1)
        # np.argmax keeps the first maximum, matching the strict `>` update the
        # sequential loop used, so the selected threshold is unchanged.
        best_index = int(np.argmax(accuracies))
        best_acc, best_thr = (
            float(accuracies[best_index]),
            float(candidates[best_index]),
        )
    else:
        best_thr = float(threshold)
        best_acc = accuracy_score(labels, (scores >= best_thr).astype(int))
    preds = (scores >= best_thr).astype(int)
    return {
        "best_threshold": best_thr,
        "accuracy": best_acc,
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "average_precision": average_precision_score(labels, scores),
    }


def eval_pair_task(model, path_list, tokenizer, thresholds=None):
    model.eval()
    print(" eval_pair_task")
    results = {}
    selected_thresholds = {}
    for index, path in enumerate(path_list):
        print(path)
        threshold = None if thresholds is None else thresholds[index]
        metric = eval_pair(model, tokenizer, path, threshold=threshold)
        results[path] = metric
        selected_thresholds[index] = metric["best_threshold"]
    model.train()
    return results, selected_thresholds


# Evaluation datasets grouped by physical split.
eval_cls_tasks = [
    (
        "data/train_set/banking77_train.csv",
        "data/val_set/banking77_validation.csv",
    ),
    (
        "data/train_set/emotion_train.csv",
        "data/val_set/emotion_validation.csv",
    ),
    (
        "data/train_set/tweet_train.csv",
        "data/val_set/tweet_validation.csv",
    ),
]

eval_sts_tasks = [
    "data/val_set/sick_validation.csv",
    "data/val_set/sts12_validation.csv",
    "data/val_set/stsb_validation.csv",
]

eval_pair_tasks = [
    "data/val_set/mrpc_validation.csv",
    "data/val_set/scitail_validation.csv",
    "data/val_set/wic_validation.csv",
]

test_cls_tasks = [
    (
        "data/train_set/banking77_train.csv",
        "data/test_set/banking77_test.csv",
    ),
    (
        "data/train_set/emotion_train.csv",
        "data/test_set/emotion_test.csv",
    ),
    (
        "data/train_set/tweet_train.csv",
        "data/test_set/tweet_test.csv",
    ),
]

test_sts_tasks = [
    "data/test_set/sick_test.csv",
    "data/test_set/sts12_test.csv",
    "data/test_set/stsb_test.csv",
]

test_pair_tasks = [
    "data/test_set/mrpc_test.csv",
    "data/test_set/scitail_test.csv",
    "data/test_set/wic_test.csv",
]
