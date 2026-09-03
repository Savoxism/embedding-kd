"""Figure 2 must compare S1 and full within one model pair."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from scripts.ablation import figures


def _row(pair, ablation, policy, distortion=0.05, sts=65.0):
    return {
        "pair": pair,
        "ablation": ablation,
        "policy": policy,
        "teacher_weighted_distortion": str(distortion),
        "sts_avg": str(sts),
    }


def test_figure_two_excludes_other_pairs_full_runs(monkeypatch, tmp_path):
    rows = [
        _row("main", "s1", "uniform"),
        _row("main", "s1", "batch_local"),
        _row("main", "s1", "uniform_corpus"),
        *[
            _row("main", "full", "topk", distortion, sts)
            for distortion, sts in [(0.02, 70.0), (0.03, 71.0), (0.04, 72.0)]
        ],
        _row("transfer", "full", "topk", 0.001, 99.0),
        _row("transfer", "x1", "uniform", 0.002, 98.0),
    ]
    coverage = [
        {"policy": policy, "epoch": "5", "coverage_mean": str(mass)}
        for policy, mass in [("topk", 0.8), ("uniform", 0.2)]
    ]
    saved = []
    monkeypatch.setattr(figures, "save", lambda fig, path: saved.append(fig))
    figures.figure_two(rows, coverage, tmp_path)
    fig = saved[0]
    try:
        for index, expected_x, expected_y in [(0, 0.8, 0.03), (1, 0.03, 71.0)]:
            containers = {
                item.get_label(): item for item in fig.axes[index].containers
            }
            topk = containers[figures.POLICY_LABEL["topk"]]
            line = topk.lines[0]
            assert list(line.get_xdata()) == pytest.approx([expected_x])
            assert list(line.get_ydata()) == pytest.approx([expected_y])
            # y-error bars must use the three main-pair seeds, not the transfer.
            y_segment = topk.lines[2][1].get_segments()[0]
            expected_std = 0.01 if index == 0 else 1.0
            assert y_segment[:, 1] == pytest.approx(
                [expected_y - expected_std, expected_y + expected_std]
            )
            assert figures.POLICY_LABEL["batch_local"] in containers
            assert figures.POLICY_LABEL["uniform_corpus"] in containers
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "rows",
    [
        [_row("main", "full", "topk")],
        [_row("main", "s1", "uniform"), _row("transfer", "s1", "uniform")],
        [_row("", "s1", "uniform")],
        [{"ablation": "s1", "policy": "uniform"}],
    ],
)
def test_figure_two_rejects_missing_or_ambiguous_pair(rows, tmp_path):
    coverage = [{"policy": "uniform", "epoch": "5", "coverage_mean": "0.2"}]
    try:
        with pytest.raises(ValueError, match="exactly one model pair"):
            figures.figure_two(rows, coverage, tmp_path)
    finally:
        plt.close("all")
