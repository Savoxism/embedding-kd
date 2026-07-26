from src.criterions.emo_embedding_distillation import align_tokens


def test_alignment_is_one_to_one_and_index_based_for_repeated_tokens():
    teacher = ["<s>", "the", "the", "cat"]
    student = ["[CLS]", "the", "the", "cat"]
    mapping = align_tokens(teacher, student)

    assert mapping == {1: 1, 2: 2, 3: 3}
    assert len(set(mapping.values())) == len(mapping)


def test_alignment_preserves_sequence_order():
    mapping = align_tokens(
        ["<s>", "play", "##ing", "ball"],
        ["[CLS]", "playing", "with", "ball"],
    )
    pairs = list(mapping.items())
    assert pairs == sorted(pairs)
    assert list(mapping.values()) == sorted(mapping.values())
