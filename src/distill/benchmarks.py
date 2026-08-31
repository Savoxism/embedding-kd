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


def add_domain_averages(results: dict[str, Any]) -> dict[str, Any]:
    """Add paper-protocol averages while preserving all detailed metrics.

    Classification contributes F1, pair classification contributes average
    precision, and STS contributes Spearman. The three aggregate values use the
    paper's [0, 100] percentage-point scale; detailed metrics remain unchanged.
    """
    scores: dict[str, float] = {}
    for family, metric_name in PRIMARY_EVALUATION_METRICS.items():
        for path, raw_values in results.get(family, {}).items():
            benchmark = Path(path).stem
            for suffix in ("_validation", "_test"):
                benchmark = benchmark.removesuffix(suffix)
            if family == "sts":
                score = float(raw_values)
            else:
                score = float(raw_values[metric_name])
            scores[benchmark] = score

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


def _benchmark_name(path: str, split: str) -> str:
    name = Path(path).stem
    suffix = f"_{split}"
    return name.removesuffix(suffix)


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


def print_evaluation_table(
    current_epoch: int,
    split: str,
    results: dict[str, Any],
) -> None:
    rows = []
    for family in ("classification", "pair", "sts"):
        family_scores = []
        for path, raw_values in results.get(family, {}).items():
            values = {"spearman": raw_values} if family == "sts" else dict(raw_values)
            metric_name = PRIMARY_EVALUATION_METRICS[family]
            score = float(values[metric_name])
            family_scores.append(score)
            rows.append(
                (
                    family,
                    _benchmark_name(path, split),
                    metric_name,
                    f"{100.0 * score:.2f}",
                    _metric_details(values),
                )
            )
        if family_scores:
            rows.append(
                (
                    family,
                    "MEAN",
                    PRIMARY_EVALUATION_METRICS[family],
                    f"{100.0 * sum(family_scores) / len(family_scores):.2f}",
                    "",
                )
            )

    title = (
        f"VALIDATION - EPOCH {current_epoch + 1}"
        if split == "validation"
        else "FINAL TEST"
    )
    headers = ("Family", "Benchmark", "Primary metric", "Score", "Details")
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
