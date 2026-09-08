import os
import warnings
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Inference batch. 64 was the training batch size carried over; with a p95 length
# of ~34 tokens it leaves the GPU idle on a forward-only pass.
EVAL_BATCH_SIZE = 256

# Iterations for the linear probe. The probe is a measuring instrument, not part
# of the method, so the ceiling is fixed: a fit that has not converged by here is
# reported and used as it stands. Refitting a hard case at a higher ceiling would
# make the instrument depend on the student being measured, which is exactly what
# a probe must not do -- the arm whose features are hardest to separate would be
# scored with a better-fit probe than the arm it is compared against.
CLASSIFIER_MAX_ITER = 200

# Pre-tokenized, length-sorted batches, keyed by (file, tokenizer, max_len, batch).
# Evaluation runs the same files every epoch against a changing student, so the
# tokenization is identical every time and was being redone on the main thread --
# where, with no DataLoader workers, it blocked the GPU.
_BATCH_CACHE: dict[tuple, list[dict]] = {}


def _tokenizer_key(tokenizer) -> tuple:
    # No `id()`: CPython reuses the id of a collected object, so a second
    # tokenizer could land on a freed entry's key and read another tokenizer's
    # token ids. The remaining fields identify the vocabulary, which is the only
    # thing the cached encoding depends on.
    return (
        type(tokenizer).__name__,
        getattr(tokenizer, "name_or_path", None),
        getattr(tokenizer, "vocab_size", None),
    )


def _pad(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for position, row in enumerate(rows):
        ids[position, : len(row)] = row
        mask[position, : len(row)] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


def _sorted_batches(
    file_path: str,
    text_columns: tuple[str, ...],
    label_column: str,
    label_dtype: torch.dtype,
    tokenizer,
    max_len: int,
    batch_size: int = EVAL_BATCH_SIZE,
) -> list[dict]:
    """Tokenize once, sort by length, and batch.

    Sorting cuts padding waste from ~2.8x to ~1.0x at these length distributions.
    The order is never restored, and it does not need to be: every metric
    downstream (Spearman, average precision, accuracy, F1, and the logistic probe)
    is invariant to the order of examples, and labels travel inside the same
    batches as the texts they belong to. With a correct attention mask the padding
    width does not enter the encoder's output either, so this is a speed change
    and not a numerical one.
    """
    key = (str(file_path), text_columns, _tokenizer_key(tokenizer), max_len, batch_size)
    cached = _BATCH_CACHE.get(key)
    if cached is not None:
        return cached

    full_path = BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
    frame = pd.read_csv(full_path)
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    # One batched call per column: fast tokenizers parallelize this in Rust, which
    # per-batch calls from a collate function never get to use.
    encoded = [
        tokenizer(
            frame[column].astype(str).tolist(), truncation=True, max_length=max_len
        )["input_ids"]
        for column in text_columns
    ]
    labels = torch.tensor(frame[label_column].tolist(), dtype=label_dtype)

    lengths = np.max([[len(row) for row in column] for column in encoded], axis=0)
    order = np.argsort(lengths, kind="stable")

    batches = []
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        batch = {"labels": labels[index]}
        for position, column in enumerate(encoded):
            ids, mask = _pad([column[int(i)] for i in index], pad_id)
            batch[f"input_ids{position + 1}"] = ids
            batch[f"attention_mask{position + 1}"] = mask
        batches.append(batch)

    _BATCH_CACHE[key] = batches
    return batches


def clear_eval_cache() -> None:
    """Drop the tokenization cache (a different tokenizer keys differently anyway)."""
    _BATCH_CACHE.clear()


@contextmanager
def _evaluating(model):
    """Put the model in eval mode and put it back the way it was found.

    The three task functions used to end with an unconditional `model.train()`,
    which is wrong for the final test evaluation -- the model is already in eval
    mode there and has no business being switched back.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def eval_sts(model, eval_loader):
    preds, labels = [], []
    device = model.device

    with (
        torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
        ),
        torch.no_grad(),
    ):
        for batch in tqdm(eval_loader):
            input_ids1 = batch["input_ids1"].to(device)
            attn1 = batch["attention_mask1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attn2 = batch["attention_mask2"].to(device)
            label = batch["labels"]

            out1 = model(input_ids=input_ids1, attention_mask=attn1)
            out2 = model(input_ids=input_ids2, attention_mask=attn2)

            # Support both StellaModel (dict) and AutoModel (object)
            if isinstance(out1, dict) and "pooled" in out1:
                emb1 = out1["pooled"]
                emb2 = out2["pooled"]
            else:
                emb1 = (
                    out1.last_hidden_state[:, 0, :]
                    if hasattr(out1, "last_hidden_state")
                    else out1["last_hidden_state"][:, 0, :]
                )
                emb2 = (
                    out2.last_hidden_state[:, 0, :]
                    if hasattr(out2, "last_hidden_state")
                    else out2["last_hidden_state"][:, 0, :]
                )

            # cosine similarity
            sim = F.cosine_similarity(emb1, emb2)
            score = (sim + 1) * 2.5  # scale [-1,1] -> [0,5]

            preds.extend(score.cpu().numpy())
            labels.extend(label.numpy())

    spearman_corr, _ = spearmanr(preds, labels)
    print(f"Spearman: {spearman_corr:.4f}")

    return spearman_corr


def eval_sts_task(model, path_list, tokenizer):
    print(" eval_sts_task")
    results = {}
    # Restore whatever mode the caller had, rather than assuming training. The
    # final test evaluation runs with the model already in eval mode, and putting
    # it back into train mode there was silently wrong.
    with _evaluating(model):
        for path in path_list:
            print(path)
            results[path] = eval_sts(
                model,
                _sorted_batches(
                    path,
                    ("sentence1", "sentence2"),
                    "score",
                    torch.float,
                    tokenizer,
                    max_len=128,
                ),
            )
    return results


def eval_cls(model, eval_loader):
    preds, labels = [], []
    device = model.device

    with torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
    ):
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                label = batch["labels"]

                out1 = model(input_ids=input_ids1, attention_mask=attn1)

                # Support both StellaModel (dict with 'pooled') and AutoModel (object/dict with last_hidden_state)
                if isinstance(out1, dict) and "pooled" in out1:
                    emb1 = out1["pooled"]
                else:
                    emb1 = (
                        out1.last_hidden_state[:, 0, :]
                        if hasattr(out1, "last_hidden_state")
                        else out1["last_hidden_state"][:, 0, :]
                    )

                preds.append(emb1.float().cpu().numpy())
                labels.append(label.numpy())

    # Concatenated blocks, not a list of N single-row arrays. The probe below
    # converts whatever it is given into one 2-D array anyway, and doing that
    # from ~10k separate arrays is the slow way to reach the same matrix.
    if not preds:
        return np.empty((0, 0), dtype=np.float32), np.empty(0)
    return np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


def _normalized_text_keys(dataset):
    return {" ".join(str(text).strip().casefold().split()) for text in dataset["text"]}


@cache
def _validate_classification_pair(train_path, eval_path):
    """Whether a train/eval pair overlaps is a property of two files on disk.

    Cached because it is not: it used to re-read both CSVs, normalize every
    text and intersect the two sets on every evaluation, i.e. once per dataset
    per epoch, to recompute the same answer. Caching also means a leakage
    warning is printed once rather than once an epoch.
    """
    train_file = BASE_DIR / train_path
    eval_file = BASE_DIR / eval_path
    train_frame = pd.read_csv(train_file)
    eval_frame = pd.read_csv(eval_file)
    overlap = _normalized_text_keys(train_frame) & _normalized_text_keys(eval_frame)

    if "val_set" in Path(eval_path).parts:
        if overlap:
            raise ValueError(
                f"Classification train-validation leakage for {eval_file.stem}: "
                f"{len(overlap)} normalized texts overlap"
            )
        return

    dataset_name = eval_file.stem.removesuffix("_test")
    validation_file = BASE_DIR / "data" / "val_set" / (f"{dataset_name}_validation.csv")
    validation_overlap = set()
    if validation_file.is_file():
        validation_frame = pd.read_csv(validation_file)
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


def eval_classification_task(model, path_list, tokenizer):
    print(" eval classifier")

    results = {}
    with _evaluating(model):
        for train_path, dev_path in path_list:
            print(dev_path)
            _validate_classification_pair(train_path, dev_path)
            train_batches = _sorted_batches(
                train_path, ("text",), "label", torch.long, tokenizer, max_len=512
            )
            eval_batches = _sorted_batches(
                dev_path, ("text",), "label", torch.long, tokenizer, max_len=512
            )

            X_train, y_train = eval_cls(model, train_batches)
            X_test, y_test = eval_cls(model, eval_batches)

            # lbfgs on 77 classes was taking ~7 s per epoch at max_iter=200, on
            # the CPU, in series with the GPU work -- more than a third of the
            # whole evaluation. Most probes converge well inside that ceiling, but
            # not all of them do (emotion, 28 classes, has missed it), so the
            # non-convergence is printed rather than swallowed: a silently
            # under-fit probe must not be mistaken for a weaker student.
            clf = LogisticRegression(
                random_state=42,
                max_iter=CLASSIFIER_MAX_ITER,
                verbose=0,
            )
            # Record the warning instead of escalating it. `simplefilter("error")`
            # raises from inside the solver loop, before `coef_` is assigned, so the
            # estimator caught there is unfitted -- and the retry it triggered ran at
            # the same ceiling and inside the same escalating context, so a probe
            # that missed 200 iters raised a second time with nothing to catch it and
            # took the whole final evaluation down after training had completed.
            # Letting the fit finish keeps the probe usable and identical across arms.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X_train, y_train)
            if any(issubclass(entry.category, ConvergenceWarning) for entry in caught):
                print(
                    f"  probe did not converge in {CLASSIFIER_MAX_ITER} iters for "
                    f"{Path(dev_path).stem}; scoring with the fit as it stands"
                )
            y_pred = clf.predict(X_test)

            scores = {}
            accuracy = accuracy_score(y_test, y_pred)
            scores["accuracy"] = accuracy
            f1 = f1_score(y_test, y_pred, average="macro")
            scores["f1"] = f1
            print(scores)
            results[dev_path] = scores

    return results


def eval_pair(model, eval_loader, threshold=None):
    preds, labels = [], []
    device = model.device

    with torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
    ):
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                input_ids2 = batch["input_ids2"].to(device)
                attn2 = batch["attention_mask2"].to(device)
                label = batch["labels"]

                out1 = model(input_ids=input_ids1, attention_mask=attn1)
                out2 = model(input_ids=input_ids2, attention_mask=attn2)

                # Support both StellaModel (dict with 'pooled') and AutoModel (object/dict with last_hidden_state)
                if isinstance(out1, dict) and "pooled" in out1:
                    emb1 = out1["pooled"]
                    emb2 = out2["pooled"]
                else:
                    emb1 = (
                        out1.last_hidden_state[:, 0, :]
                        if hasattr(out1, "last_hidden_state")
                        else out1["last_hidden_state"][:, 0, :]
                    )
                    emb2 = (
                        out2.last_hidden_state[:, 0, :]
                        if hasattr(out2, "last_hidden_state")
                        else out2["last_hidden_state"][:, 0, :]
                    )

                # cosine similarity
                sim = F.cosine_similarity(emb1, emb2)
                score = (sim + 1) / 2

                preds.extend(score.cpu().numpy())
                labels.extend(label.numpy())

    metric = get_metric_pair_classification(preds, labels, threshold=threshold)
    print(metric)

    return metric


def get_metric_pair_classification(scores, labels, threshold=None):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if threshold is None:
        # One [N, 200] comparison instead of 200 passes through accuracy_score.
        # `argmax` returns the first maximum, which is the same threshold the
        # loop's strict `>` kept, so the selected value is unchanged.
        grid = np.linspace(0, 1, 200)
        accuracies = ((scores[:, None] >= grid[None, :]) == labels[:, None]).mean(
            axis=0
        )
        best = int(accuracies.argmax())
        best_acc, best_thr = float(accuracies[best]), float(grid[best])
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
    print(" eval_pair_task")
    results = {}
    selected_thresholds = {}
    with _evaluating(model):
        for index, path in enumerate(path_list):
            print(path)
            batches = _sorted_batches(
                path,
                ("sentence1", "sentence2"),
                "label",
                torch.float,
                tokenizer,
                max_len=128,
            )
            threshold = None if thresholds is None else thresholds[index]
            metric = eval_pair(model, batches, threshold=threshold)
            results[path] = metric
            selected_thresholds[index] = metric["best_threshold"]
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
