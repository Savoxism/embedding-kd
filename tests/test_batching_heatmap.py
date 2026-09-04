import numpy as np

from scripts.ablation.batching_heatmap import aggregate_edges, block_ids_from_teacher_order


def test_block_ids_follow_teacher_order():
    order = np.array([3, 1, 0, 2])
    blocks = block_ids_from_teacher_order(order, 2)
    assert blocks.tolist() == [1, 0, 1, 0]


def test_aggregate_edges_normalizes_by_anchor_block_size():
    blocks = np.array([0, 0, 1, 1])
    rows = np.array([0, 1, 2, 3])
    columns = np.array([2, 3, 0, 1])
    values = np.ones(4)
    matrix = aggregate_edges(rows, columns, values, blocks, 2)
    assert np.array_equal(matrix, np.array([[0.0, 1.0], [1.0, 0.0]]))
