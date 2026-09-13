"""The base student's own corpus embeddings, for the student_knn ladder arm.

Only the graph's columns come from these. The artifact built on them keeps the
teacher's row temperatures, and the arm reads its targets off the teacher bank,
so the texts an anchor is compared against are the one thing that differs from
the teacher arm.

The support is frozen at the base student, before any distillation: following
the student as it trains would make the arm a moving target rather than a
controlled comparison.
"""

import torch


@torch.no_grad()
def encode_base_student(
    model_name: str,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """CLS-pooled embeddings from a freshly loaded, untrained student.

    Takes the training run's tokenizer rather than loading one by name: the
    MiniLMv2 checkpoints ship a stub tokenizer that `AutoTokenizer` resolves
    wrongly, and the run has already resolved the right one.
    """
    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    outputs = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        result = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        # CLS pooling, matching the student path in training and evaluation.
        outputs.append(result.last_hidden_state[:, 0, :].float().cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)
