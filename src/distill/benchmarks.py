"""Benchmark grouping, the paper's domain averages, and the results table.

The split into in-domain and out-of-domain benchmarks and the choice of one
primary metric per task family are protocol, not presentation, so they live in
one place rather than being restated wherever results are printed.
"""

from pathlib import Path
from typing import Any

PRIMARY_EVALUATION_METRICS = {
    "classification": "f1",
    "pair": "average_precision",
    "sts": "spearman",
}
IN_DOMAIN_BENCHMARKS = ("emotion", "wic", "stsb")
OUT_DOMAIN_BENCHMARKS = (
    "banking77",
    "tweet",
    "mrpc",
    "scitail",
    "sick",
    "sts12",
)
ALL_BENCHMARKS = IN_DOMAIN_BENCHMARKS + OUT_DOMAIN_BENCHMARKS


def _strip_split_suffix(path: str) -> str:
    benchmark = Path(path).stem
    for suffix in ("_validation", "_test"):
        benchmark = benchmark.removesuffix(suffix)
    return benchmark


def primary_scores(results: dict[str, Any]) -> dict[str, float]:
    """One primary score per benchmark, on the raw [0, 1] scale.

    Both the domain averages and the printed table read the same numbers from
    here, so the table can never disagree with the averages logged beside it.
    """
    scores: dict[str, float] = {}
    for family, metric_name in PRIMARY_EVALUATION_METRICS.items():
        for path, raw_values in results.get(family, {}).items():
            if family == "sts":
                score = float(raw_values)
            else:
                score = float(raw_values[metric_name])
            scores[_strip_split_suffix(path)] = score
    return scores


def add_domain_averages(results: dict[str, Any]) -> dict[str, Any]:
    """Add paper-protocol averages while preserving all detailed metrics.

    Classification contributes F1, pair classification contributes average
    precision, and STS contributes Spearman. The three aggregate values use the
    paper's [0, 100] percentage-point scale; detailed metrics remain unchanged.
    """
    scores = primary_scores(results)

    missing = sorted(set(ALL_BENCHMARKS) - scores.keys())
    if missing:
        raise ValueError(
            "Cannot compute domain averages; missing primary metrics for: "
            + ", ".join(missing)
        )

    enriched = dict(results)
    enriched["avg_in"] = round(
        100.0
        * sum(scores[name] for name in IN_DOMAIN_BENCHMARKS)
        / len(IN_DOMAIN_BENCHMARKS),
        2,
    )
    enriched["avg_out"] = round(
        100.0
        * sum(scores[name] for name in OUT_DOMAIN_BENCHMARKS)
        / len(OUT_DOMAIN_BENCHMARKS),
        2,
    )
    enriched["avg"] = round(
        100.0 * sum(scores[name] for name in ALL_BENCHMARKS) / len(ALL_BENCHMARKS),
        2,
    )
    return enriched


def _metric_details(values: dict[str, Any]) -> str:
    labels = {
        "accuracy": "Acc",
        "f1": "F1",
        "precision": "P",
        "recall": "R",
        "average_precision": "AP",
        "spearman": "Spearman",
    }
    details = []
    for key, label in labels.items():
        value = values.get(key)
        if isinstance(value, (int, float)):
            details.append(f"{label}={100.0 * float(value):.2f}")
    return " ".join(details)


def _benchmark_rows(results: dict[str, Any]) -> dict[str, tuple[str, str, str]]:
    """(primary metric, score, details) per benchmark, keyed by benchmark name."""
    rows: dict[str, tuple[str, str, str]] = {}
    for family, metric_name in PRIMARY_EVALUATION_METRICS.items():
        for path, raw_values in results.get(family, {}).items():
            values = {"spearman": raw_values} if family == "sts" else dict(raw_values)
            score = float(values[metric_name])
            rows[_strip_split_suffix(path)] = (
                metric_name,
                f"{100.0 * score:.2f}",
                _metric_details(values),
            )
    return rows


def print_evaluation_table(
    current_epoch: int,
    split: str,
    results: dict[str, Any],
    final: bool = False,
) -> None:
    """Print benchmarks grouped by domain, each group closed by its average.

    The grouping the paper reports on is in-domain vs out-of-domain, so that is
    what the table shows: the in-domain benchmarks and AVG IN-DOMAIN, then the
    out-of-domain benchmarks and AVG OUT-OF-DOMAIN, then AVG over all nine. The
    per-family means are gone -- no result is reported per family, and they were
    only inviting the wrong number to be read off the table.
    """
    details_by_benchmark = _benchmark_rows(results)
    scores = primary_scores(results)

    def _group_average(names: tuple[str, ...]) -> float | None:
        present = [scores[name] for name in names if name in scores]
        if len(present) != len(names):
            return None
        return 100.0 * sum(present) / len(present)

    rows: list[tuple[str, str, str, str, str]] = []
    groups = (
        ("in-domain", IN_DOMAIN_BENCHMARKS, "AVG IN-DOMAIN"),
        ("out-of-domain", OUT_DOMAIN_BENCHMARKS, "AVG OUT-OF-DOMAIN"),
    )
    for group_label, names, average_label in groups:
        for name in names:
            entry = details_by_benchmark.get(name)
            if entry is None:
                continue
            metric_name, score, details = entry
            rows.append((group_label, name, metric_name, score, details))
        average = _group_average(names)
        if average is not None:
            rows.append((group_label, average_label, "", f"{average:.2f}", ""))

    reported = {name for _, name, _, _, _ in rows}
    extras = [name for name in sorted(details_by_benchmark) if name not in reported]
    for name in extras:
        metric_name, score, details = details_by_benchmark[name]
        rows.append(("ungrouped", name, metric_name, score, details))

    overall = _group_average(ALL_BENCHMARKS)
    if overall is not None:
        rows.append(("", "AVG", "", f"{overall:.2f}", ""))

    if not rows:
        return

    title = "FINAL TEST" if final else f"{split.upper()} - EPOCH {current_epoch + 1}"
    headers = ("Group", "Benchmark", "Primary metric", "Score", "Details")
    widths = [
        max([len(headers[index]), *(len(row[index]) for row in rows)])
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)

    print("\n" + "=" * len(separator))
    print(title)
    print("=" * len(separator))
    print(
        " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    )
    print(separator)
    for row in rows:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(len(row))))
    print("=" * len(separator) + "\n")
