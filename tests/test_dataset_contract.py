import pandas as pd
import torch

from src.data_utils.dataset import DualTokenizerCollate, TextPairRaw


class FakeFastTokenizer:
    is_fast = True

    def __call__(
        self,
        texts,
        max_length,
        truncation,
        padding,
        return_tensors,
        return_special_tokens_mask,
        return_offsets_mapping,
    ):
        width = max(len(text.split()) for text in texts) + 2
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        special = torch.zeros_like(input_ids)
        offsets = torch.zeros((len(texts), width, 2), dtype=torch.long)
        for row, text in enumerate(texts):
            input_ids[row, 0] = 101
            attention[row, 0] = 1
            special[row, 0] = 1
            cursor = 0
            for column, token in enumerate(text.split(), start=1):
                start = text.index(token, cursor)
                end = start + len(token)
                cursor = end
                input_ids[row, column] = column
                attention[row, column] = 1
                offsets[row, column] = torch.tensor([start, end])
            last = len(text.split()) + 1
            input_ids[row, last] = 102
            attention[row, last] = 1
            special[row, last] = 1
        result = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "special_tokens_mask": special,
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def test_single_classification_contract_includes_labels_and_offsets():
    frame = pd.DataFrame({"text": ["hello world", "short"], "label": [1, 0]})
    dataset = TextPairRaw(frame, "single_cls")
    collate = DualTokenizerCollate(
        FakeFastTokenizer(),
        FakeFastTokenizer(),
        "single_cls",
        max_len=16,
        return_offsets=True,
    )
    batch = collate([dataset[0], dataset[1]])

    assert batch["labels"].tolist() == [1, 0]
    assert "input_ids1_stu" in batch
    assert "input_ids2_stu" not in batch
    assert batch["offset_mapping1_tea"].shape[-1] == 2


def test_pair_contract_preserves_repeated_tokens_and_labels():
    frame = pd.DataFrame(
        {
            "premise": ["same same", "left"],
            "hypothesis": ["same", "right right"],
            "label": [1, 0],
        }
    )
    dataset = TextPairRaw(frame, "pair_cls")
    collate = DualTokenizerCollate(
        FakeFastTokenizer(),
        FakeFastTokenizer(),
        "pair_cls",
        max_len=16,
        return_offsets=True,
    )
    batch = collate([dataset[0], dataset[1]])

    assert batch["labels"].tolist() == [1, 0]
    assert batch["raw_texts1"] == ["same same", "left"]
    assert batch["raw_texts2"] == ["same", "right right"]
    assert "offset_mapping2_stu" in batch
