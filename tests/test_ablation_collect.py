"""Contracts for compact run artifacts and the locked six-table matrix."""

import csv
import json

from scripts.ablation import compact_run
from scripts.ablation.collect import (
    MAIN_PAIRS,
    PRIMARY_PAIR,
    TABLES,
    aggregate_table,
    read_compact_result,
)


def test_locked_suite_contains_exactly_66_unique_runs():
    unique = {
        (pair, group, arm)
        for specs in TABLES.values()
        for _, pair, group, arm in specs
    }
    assert len(unique) == 22
    assert len(unique) * 3 == 66


def test_sensitivity_grid_matches_the_paper_request():
    labels = [label for label, *_ in TABLES["table5_sensitivity"]]
    assert labels == [
        "topk:0.5",
        "topk:0.75",
        "topk:1",
        "topk:1.5",
        "topk:2",
        "topk:2.5",
        "topk:3",
        "row_weight:0",
        "row_weight:0.1",
        "row_weight:0.25",
        "row_weight:0.5",
        "row_weight:0.75",
        "row_weight:1",
    ]


def test_efficiency_uses_only_cache_warm_runs_for_warm_wall_time():
    runs = []
    for seed, warm, wall in ((42, False, 100.0), (43, True, 60.0), (44, True, 64.0)):
        runs.append(
            {
                "pair": PRIMARY_PAIR,
                "group": "full",
                "arm": "full",
                "seed": seed,
                "avg": 75.0,
                "wall_clock_seconds": wall,
                "teacher_cache_warm_before": warm,
                "graph_cache_warm_before": warm,
            }
        )
    table = aggregate_table(
        "table6_efficiency",
        [(MAIN_PAIRS[0], PRIMARY_PAIR, "full", "full")],
        {(PRIMARY_PAIR, "full", "full"): runs},
        (42, 43, 44),
        coverage={},
        allow_incomplete=False,
    )
    assert table[0]["warm_end_to_end_n"] == 2
    assert table[0]["warm_end_to_end_seconds_mean"] == 62.0


def test_successful_run_is_compacted_to_exactly_one_csv(monkeypatch, tmp_path):
    run_dir = tmp_path / "full" / "full" / "seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("temporary log")
    (run_dir / "run.json").write_text("{}")
    weights = run_dir / "weights"
    weights.mkdir()
    (weights / "student.pt").write_bytes(b"weights")
    graph_log = tmp_path / "logs" / "graph_base"
    graph_log.mkdir(parents=True)
    (graph_log / "knn_graph_neighbors.jsonl").write_text("{}\n")

    row = {
        "pair": PRIMARY_PAIR,
        "experiment": "paper_r1_v2",
        "group": "full",
        "arm": "full",
        "seed": 42,
        "avg": 75.0,
        "eval_every": 0,
        "final_weights_only": True,
        "request_sha256": "abc",
    }
    monkeypatch.setattr(compact_run, "load_run", lambda *args, **kwargs: row)
    output = compact_run.compact(run_dir, "paper_r1_v2", graph_log)

    assert output == run_dir / "result.csv"
    assert [path.name for path in run_dir.iterdir()] == ["result.csv"]
    assert not graph_log.exists()
    with output.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {key: str(value) for key, value in row.items()}
        ]


def test_compact_result_restores_numeric_and_boolean_types(tmp_path):
    path = tmp_path / "result.csv"
    row = {
        "experiment": "paper_r1_v2",
        "seed": "42",
        "avg": "75.1",
        "eval_every": "0",
        "final_weights_only": "True",
        "teacher_cache_warm_before": "False",
        "graph_cache_warm_before": "True",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    loaded = read_compact_result(path, "paper_r1_v2")
    assert loaded["seed"] == 42
    assert loaded["avg"] == 75.1
    assert loaded["teacher_cache_warm_before"] is False
    assert loaded["graph_cache_warm_before"] is True


def test_compactor_reads_real_run_telemetry_before_deleting_it(tmp_path):
    run_dir = (
        tmp_path
        / PRIMARY_PAIR
        / "paper_r1_v2"
        / "full"
        / "full"
        / "seed42"
    )
    run_dir.mkdir(parents=True)
    request = {
        "experiment": "paper_r1_v2",
        "code_sha256": "code",
    }
    (run_dir / "request.json").write_text(json.dumps(request))
    (run_dir / "arm.json").write_text(
        json.dumps(
            {
                "group": "full",
                "arm": "full",
                "seed": 42,
                "graph_key": "base",
                "wall_clock_seconds": 12,
                "teacher_cache_warm_before": True,
                "graph_cache_warm_before": True,
            }
        )
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "git": {"sha": "abc", "dirty": False},
                "env": {"gpu": "test-gpu"},
                "config": {
                    "eval_every": 0,
                    "final_weights_only": True,
                    "diffusion_scales": [1],
                },
            }
        )
    )
    test = {
        "classification": {
            "emotion_test.csv": {"f1": 0.7},
            "banking77_test.csv": {"f1": 0.7},
            "tweet_test.csv": {"f1": 0.7},
        },
        "pair": {
            "mrpc_test.csv": {"average_precision": 0.7},
            "scitail_test.csv": {"average_precision": 0.7},
            "wic_test.csv": {"average_precision": 0.7},
        },
        "sts": {
            "sick_test.csv": 0.7,
            "sts12_test.csv": 0.7,
            "stsb_test.csv": 0.7,
        },
        "avg_in": 70.0,
        "avg_out": 70.0,
    }
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"run_id": "run-1", "test": test}) + "\n"
    )
    (run_dir / "epochs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "train": {
                    "encoded_texts_cum": 100,
                    "encoded_tokens_cum": 200,
                    "peak_memory_mb": 300,
                },
                "geometry": {
                    "teacher_weighted_distortion": 0.02,
                    "teacher_student_spearman": 0.8,
                },
            }
        )
        + "\n"
    )
    (run_dir / "step_metrics.jsonl").write_text(
        json.dumps({"run_id": "run-1", "step_seconds": 0.4}) + "\n"
    )
    output = compact_run.compact(run_dir, "paper_r1_v2")
    loaded = read_compact_result(output, "paper_r1_v2")

    assert [path.name for path in run_dir.iterdir()] == ["result.csv"]
    assert loaded["avg"] == 70.0
    assert loaded["avg_in"] == 70.0
    assert loaded["emotion"] == 70.0
    assert loaded["mean_step_seconds"] == 0.4
