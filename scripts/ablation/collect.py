"""Aggregate ablation runs into the tidy CSV, the console tables, and LaTeX.

One row per (ablation, arm, seed), read from the files a run already writes:
`arm.json` for the arm's identity and wall-clock, `metrics.jsonl` for the final
test scores, `epochs.jsonl` for the geometry probe and the encoder budget, and
`run.json` for the resolved config and the graph statistics. Nothing is parsed
back out of a directory name.

Three deliberate choices, each answering a way an ablation table can mislead:

*STS and pair-classification are reported separately, never only as `Avg`.* The
method's gain over TALAS is concentrated in STS while pair-cls is down; an arm
that trades one for the other at constant `Avg` looks inert in a single column
and is not. `Avg` is still shown, third.

*Deltas are paired by seed.* Seeds move all arms together, so a difference of
means across independently-averaged arms carries seed noise that the paired
difference does not. `mean_delta` and its std are computed per seed and then
averaged.

*The encoder budget is printed next to every score.* `encoded_texts` and
`encoded_tokens` are what the plan requires the arms to be matched on; an arm
that wins while encoding more texts has not been shown to win.

Usage:
    python scripts/ablation/collect.py                        # console tables
    python scripts/ablation/collect.py --csv runs/ablation/analysis/results.csv
    python scripts/ablation/collect.py --latex latex/tables   # booktabs tables
    python scripts/ablation/collect.py --hubness              # G1 graph diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.distill.benchmarks import (  # noqa: E402
    ALL_BENCHMARKS,
    PRIMARY_EVALUATION_METRICS,
    primary_scores,
)

# The three families, spelled out here because the plan reports two of them
# separately and `benchmarks.py` groups by domain, not by family.
FAMILY_BENCHMARKS = {
    "sts": ("sick", "sts12", "stsb"),
    "pair": ("mrpc", "scitail", "wic"),
    "cls": ("banking77", "tweet", "emotion"),
}
BASELINE_ABLATION = "full"

# Geometry probe fields worth carrying into the table. E_hat is the middle link
# of the causal chain; Spearman is the benchmark-free summary of it.
GEOMETRY_FIELDS = (
    "teacher_weighted_distortion",
    "teacher_student_spearman",
    "cosine_rmse",
    "anisotropy",
    "effective_rank",
    "uniformity",
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _final_test(save_dir: Path) -> dict | None:
    """The last recorded test evaluation, whichever file carried it."""
    for name in ("metrics.jsonl", "epochs.jsonl"):
        for row in reversed(_read_jsonl(save_dir / name)):
            if isinstance(row.get("test"), dict):
                return row["test"]
    return None


def _last_epoch(save_dir: Path) -> dict | None:
    rows = _read_jsonl(save_dir / "epochs.jsonl")
    return rows[-1] if rows else None


def load_run(save_dir: Path) -> dict | None:
    arm_path = save_dir / "arm.json"
    if not arm_path.exists():
        return None
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    test = _final_test(save_dir)
    if test is None:
        print(f"  [warn] no test scores yet, skipping: {save_dir}")
        return None

    scores = primary_scores(test)
    missing = sorted(set(ALL_BENCHMARKS) - scores.keys())
    if missing:
        print(f"  [warn] {save_dir}: missing benchmarks {missing}, skipping")
        return None

    row: dict = {
        "pair": save_dir.parents[2].name,
        "ablation": arm["ablation"],
        "arm": arm["arm"],
        "seed": arm["seed"],
        "graph_key": arm.get("graph_key", ""),
        "wall_clock_s": arm.get("wall_clock_seconds"),
        "extra_args": " ".join(arm.get("extra_args", [])),
        "save_dir": str(save_dir),
    }
    for family, names in FAMILY_BENCHMARKS.items():
        row[f"{family}_avg"] = 100.0 * sum(scores[n] for n in names) / len(names)
    row["avg"] = 100.0 * sum(scores[n] for n in ALL_BENCHMARKS) / len(ALL_BENCHMARKS)
    row["avg_in"] = test.get("avg_in")
    row["avg_out"] = test.get("avg_out")
    for name in ALL_BENCHMARKS:
        row[name] = 100.0 * scores[name]

    epoch = _last_epoch(save_dir) or {}
    geometry = epoch.get("geometry") or {}
    for field in GEOMETRY_FIELDS:
        row[field] = geometry.get(field)
    train = epoch.get("train") or {}
    for field in (
        "encoded_texts_cum",
        "encoded_tokens_cum",
        "peak_memory_mb",
        "loss",
        "loss_rel",
        "loss_row",
        "row_exposed_mass",
    ):
        row[field] = train.get(field)

    manifest_path = save_dir / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config", {})
        row["run_id"] = manifest.get("run_id")
        row["git_sha"] = (manifest.get("git") or {}).get("sha")
        row["git_dirty"] = (manifest.get("git") or {}).get("dirty")
        row["gpu"] = (manifest.get("env") or {}).get("gpu")
        for key in (
            "support_policy",
            "relation_target",
            "use_ambient",
            "knn_mode",
            "row_weight",
            "diffusion_scales",
            "diffusion_quota",
            "hard_neg_k",
            "random_neg_k",
            "direct_temp",
            "learning_rate",
        ):
            row[key] = config.get(key)
        row["graph_stats"] = (manifest.get("artifact") or {}).get("graph_stats", {})
        # One canonical label for "which support regime produced this run".
        # `support_policy` alone is wrong for the two baselines: neither draws
        # support at all, so both leave that field at its default and would be
        # collected -- and plotted -- as the method. Derived here, once, where the
        # whole config is in scope.
        if config.get("batch_local"):
            row["policy"] = "batch_local"
        elif config.get("relation_target") == "ambient_only":
            row["policy"] = "uniform_corpus"
        else:
            row["policy"] = config.get("support_policy")
    return row


def discover(root: Path) -> list[dict]:
    runs = []
    for arm_path in sorted(root.rglob("arm.json")):
        run = load_run(arm_path.parent)
        if run is not None:
            runs.append(run)
    return runs


def _agg(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return float("nan"), float("nan")
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def _fmt(mean: float, std: float, digits: int = 2) -> str:
    if mean != mean:
        return "--"
    return f"{mean:.{digits}f}+-{std:.{digits}f}"


def _paired_delta(arm_rows: list[dict], base_by_seed: dict[int, dict], field: str):
    """Per-seed difference against the full model, then averaged.

    A difference of independently-averaged means carries the seed-to-seed spread
    of both arms; the paired difference cancels the part of it that is common,
    which on 3 seeds is most of it.
    """
    deltas = [
        row[field] - base_by_seed[row["seed"]][field]
        for row in arm_rows
        if row["seed"] in base_by_seed
        and row.get(field) is not None
        and base_by_seed[row["seed"]].get(field) is not None
    ]
    if not deltas:
        return None, None
    return statistics.fmean(deltas), (
        statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    )


def print_tables(runs: list[dict]) -> None:
    by_pair: dict[str, list[dict]] = {}
    for row in runs:
        by_pair.setdefault(row["pair"], []).append(row)

    for pair, pair_rows in sorted(by_pair.items()):
        base_by_seed = {
            row["seed"]: row
            for row in pair_rows
            if row["ablation"] == BASELINE_ABLATION
        }
        ablations = sorted({row["ablation"] for row in pair_rows})
        # The full model is the reference row of every block, so it is printed
        # inside each rather than once at the top.
        ablations = [a for a in ablations if a != BASELINE_ABLATION]

        print(f"\n{'=' * 108}\n{pair}\n{'=' * 108}")
        header = (
            f"{'arm':<22}{'n':>2}  {'STS Avg':>14}{'Pair Avg':>14}{'Avg':>14}"
            f"{'E_hat':>10}{'Spearman':>10}{'enc.texts':>12}{'d Avg':>10}"
        )
        for ablation in [BASELINE_ABLATION] + ablations:
            block = [r for r in pair_rows if r["ablation"] == ablation]
            if not block:
                continue
            print(f"\n-- {ablation} " + "-" * (104 - len(ablation)))
            print(header)
            for arm in sorted({r["arm"] for r in block}):
                arm_rows = [r for r in block if r["arm"] == arm]
                cells = [f"{arm:<22}{len(arm_rows):>2}  "]
                for field in ("sts_avg", "pair_avg", "avg"):
                    cells.append(f"{_fmt(*_agg([r[field] for r in arm_rows])):>14}")
                distortion, _ = _agg(
                    [r["teacher_weighted_distortion"] for r in arm_rows]
                )
                spearman, _ = _agg([r["teacher_student_spearman"] for r in arm_rows])
                texts, _ = _agg([r["encoded_texts_cum"] for r in arm_rows])
                cells.append("      --  " if distortion != distortion else f"{distortion:>10.4f}")
                cells.append("      --  " if spearman != spearman else f"{spearman:>10.4f}")
                cells.append("          --" if texts != texts else f"{texts:>12,.0f}")
                delta, delta_std = _paired_delta(arm_rows, base_by_seed, "avg")
                cells.append(
                    "        --" if delta is None else f"{delta:>+10.2f}"
                )
                print("".join(cells))
                if delta is not None and delta_std:
                    print(f"{'':<24}  (paired delta std {delta_std:.2f})")


def print_hubness(runs: list[dict]) -> None:
    """G1's primary evidence: the indegree tail, per graph build."""
    seen: dict[str, dict] = {}
    for row in runs:
        stats = row.get("graph_stats") or {}
        if stats:
            seen.setdefault(f"{row['pair']}/{row['graph_key']}", stats)
    if not seen:
        print("no graph statistics recorded (needs run.json from a completed run)")
        return
    fields = (
        "avg_degree",
        "max_degree",
        "indegree_max",
        "indegree_p99",
        "indegree_gini",
        "hub_edge_share_top1pct",
        "isolated_indegree_rate",
        "fallback_rate",
    )
    # Truncated to the column width: a header longer than its column runs into
    # its neighbour and the table stops being readable at exactly the moment it
    # has something to say.
    print(f"\n{'graph':<40}" + "".join(f"{f[:19]:>21}" for f in fields))
    for key, stats in sorted(seen.items()):
        cells = "".join(
            f"{stats[f]:>21.4f}"
            if isinstance(stats.get(f), (int, float))
            else f"{'--':>21}"
            for f in fields
        )
        print(f"{key:<40}{cells}")


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_latex(runs: list[dict], out_dir: Path) -> None:
    """One booktabs table per ablation, in the template's existing style.

    Not \\resizebox'd: these tables have few enough columns to set at full size,
    and a resized table renders its numbers at a different point size from the
    body text, which is the most common way an ablation table looks bolted on.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_pair: dict[str, list[dict]] = {}
    for row in runs:
        by_pair.setdefault(row["pair"], []).append(row)

    for pair, pair_rows in sorted(by_pair.items()):
        for ablation in sorted({r["ablation"] for r in pair_rows}):
            if ablation == BASELINE_ABLATION:
                continue
            block = [
                r
                for r in pair_rows
                if r["ablation"] in (ablation, BASELINE_ABLATION)
            ]
            arms = [BASELINE_ABLATION] + sorted(
                {r["arm"] for r in block if r["ablation"] == ablation}
            )
            lines = [
                r"\begin{table}[t]",
                r"\centering",
                rf"\caption{{Ablation {_latex_escape(ablation.upper())} on "
                rf"{_latex_escape(pair)}. Mean $\pm$ std over "
                rf"{len({r['seed'] for r in block})} seeds. $\hat{{E}}$ is the "
                r"teacher-weighted cosine distortion on the held-out geometry probe "
                r"(lower is better); $\rho$ is the teacher--student Spearman "
                r"correlation of pairwise similarity. Encoded texts is the "
                r"cumulative number of student forward passes, the budget the arms "
                r"are matched on.}",
                rf"\label{{tab:ablation_{ablation}_{pair}}}",
                r"\begin{tabular}{lcccccc}",
                r"\toprule",
                r"Arm & STS Avg & Pair-cls Avg & Avg & $\hat{E}$ $\downarrow$ "
                r"& $\rho$ $\uparrow$ & Enc.\ texts \\",
                r"\midrule",
            ]
            for arm in arms:
                arm_rows = [r for r in block if r["arm"] == arm]
                if not arm_rows:
                    continue
                cells = [_latex_escape(arm)]
                for field in ("sts_avg", "pair_avg", "avg"):
                    mean, std = _agg([r[field] for r in arm_rows])
                    cells.append(
                        "--" if mean != mean else f"${mean:.2f} \\pm {std:.2f}$"
                    )
                for field, digits in (
                    ("teacher_weighted_distortion", 4),
                    ("teacher_student_spearman", 4),
                ):
                    mean, std = _agg([r[field] for r in arm_rows])
                    cells.append(
                        "--"
                        if mean != mean
                        else f"${mean:.{digits}f} \\pm {std:.{digits}f}$"
                    )
                texts, _ = _agg([r["encoded_texts_cum"] for r in arm_rows])
                cells.append("--" if texts != texts else f"{texts / 1e6:.2f}M")
                lines.append(" & ".join(cells) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
            path = out_dir / f"ablation_{ablation}_{pair}.tex"
            path.write_text("\n".join(lines), encoding="utf-8")
            print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/ablation", help="ablation output root")
    parser.add_argument("--csv", default=None, help="write the tidy CSV here")
    parser.add_argument("--latex", default=None, help="write booktabs tables here")
    parser.add_argument(
        "--hubness", action="store_true", help="print the kNN-mode graph diagnostics"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"nothing to collect: {root} does not exist")
        return 1
    runs = discover(root)
    if not runs:
        print(f"no completed runs found under {root}")
        return 1
    print(f"collected {len(runs)} runs from {root}")

    print_tables(runs)
    if args.hubness:
        print_hubness(runs)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        # graph_stats is a nested dict; it belongs in --hubness, not in a CSV cell.
        fields = [k for k in runs[0] if k != "graph_stats"]
        for run in runs:
            for key in run:
                if key not in fields and key != "graph_stats":
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(runs)
        print(f"\nwrote {path} ({len(runs)} rows, {len(fields)} columns)")

    if args.latex:
        write_latex(runs, Path(args.latex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
