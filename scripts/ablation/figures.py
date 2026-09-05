"""Generate paper figures from completed GGPKD ablation artifacts.

The current paper uses two figures from this script:

* cumulative teacher-mass coverage for the three graph support policies;
* a four-panel sensitivity plot for Top-K quota and row-loss weight.

The module also retains ``figure_two`` for compatibility with the earlier
coverage-to-downstream diagnostic and its regression test.  That helper is not
included in the current paper.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FULL_WIDTH = 5.5
INK = "#171717"
INK_MUTED = "#77736b"
GRID = "#deddd8"
BLUE = "#2878c8"
ORANGE = "#e96b38"

POLICY_LABEL = {
    "batch_local": "Batch-relational KD",
    "uniform_corpus": "Uniform corpus",
    "uniform": "Uniform within pool",
    "proportional": "Teacher-proportional",
    "topk": r"Teacher top-$K$ (ours)",
}
POLICY_ORDER = (
    "batch_local",
    "uniform_corpus",
    "uniform",
    "proportional",
    "topk",
)
POLICY_COLOR = {
    "batch_local": "#555555",
    "uniform_corpus": "#99958d",
    "uniform": "#4b3fb4",
    "proportional": "#1aa77a",
    "topk": BLUE,
}
POLICY_MARKER = {
    "batch_local": "v",
    "uniform_corpus": "P",
    "uniform": "s",
    "proportional": "D",
    "topk": "o",
}
POLICY_DASH = {
    "uniform": (0, (1, 1.5)),
    "proportional": (0, (4, 1.5, 1, 1.5)),
    "topk": (0, ()),
}
BASELINE_POLICIES = ("batch_local", "uniform_corpus")


def use_paper_style() -> None:
    """Set vector-safe typography at the ICLR template's printed width."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": "#575757",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": "#575757",
            "ytick.color": "#575757",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, path_stem: str | Path) -> None:
    """Write vector PDF for LaTeX and a PNG preview."""
    stem = str(path_stem)
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def figure_coverage(coverage_rows: list[dict], out_dir: Path) -> None:
    """Plot cumulative relation coverage; old hybrid-policy rows are ignored."""
    series: dict[str, dict[int, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in coverage_rows:
        policy = row.get("policy", "")
        if policy not in ("uniform", "proportional", "topk"):
            continue
        series[policy][int(row["epoch"])].append(
            (
                float(row["coverage_mean"]),
                float(row["coverage_p10"]),
                float(row["coverage_p90"]),
            )
        )

    missing = sorted({"uniform", "proportional", "topk"} - set(series))
    if missing:
        raise ValueError(f"coverage CSV is missing policies: {missing}")

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.55))
    handles = []
    for policy in ("uniform", "proportional", "topk"):
        epochs = sorted(series[policy])
        means = [statistics.fmean(v[0] for v in series[policy][e]) for e in epochs]
        lows = [statistics.fmean(v[1] for v in series[policy][e]) for e in epochs]
        highs = [statistics.fmean(v[2] for v in series[policy][e]) for e in epochs]
        color = POLICY_COLOR[policy]
        ax.fill_between(epochs, lows, highs, color=color, alpha=0.09, linewidth=0)
        (line,) = ax.plot(
            epochs,
            means,
            color=color,
            linestyle=POLICY_DASH[policy],
            marker=POLICY_MARKER[policy],
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.7 if policy == "topk" else 1.3,
            label=POLICY_LABEL[policy],
            zorder=3,
        )
        handles.append(line)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Cumulative exposed mass $1-\delta_T$")
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.9, 5.1)
    ax.set_ylim(0, 1.01)
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        borderaxespad=0,
    )
    fig.subplots_adjust(bottom=0.29, left=0.12, right=0.99, top=0.98)
    save(fig, out_dir / "fig_support_coverage")


def figure_one(coverage_rows: list[dict], out_dir: Path) -> None:
    """Backward-compatible name for the coverage figure."""
    figure_coverage(coverage_rows, out_dir)


def _float(value: str | None) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def figure_two(
    result_rows: list[dict], coverage_rows: list[dict], out_dir: Path
) -> None:
    """Legacy coverage-to-geometry diagnostic, isolated to one model pair."""
    s1_pairs = {row.get("pair") for row in result_rows if row.get("ablation") == "s1"}
    if len(s1_pairs) != 1 or not all(s1_pairs):
        raise ValueError(
            "Figure 2 requires S1 results for exactly one model pair; "
            "filter results and coverage to that pair."
        )
    pair = s1_pairs.pop()

    last_epoch = max(int(row["epoch"]) for row in coverage_rows)
    final_coverage: dict[str, list[float]] = defaultdict(list)
    for row in coverage_rows:
        if int(row["epoch"]) == last_epoch:
            final_coverage[row["policy"]].append(float(row["coverage_mean"]))

    points: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in result_rows:
        if row.get("pair") != pair or row.get("ablation") not in ("s1", "full"):
            continue
        policy = row.get("policy") or row.get("support_policy") or ""
        if policy not in POLICY_ORDER:
            continue
        for field in ("teacher_weighted_distortion", "sts_avg"):
            value = _float(row.get(field))
            if value is not None:
                points[policy][field].append(value)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))
    panels = (
        ("coverage", "teacher_weighted_distortion"),
        ("teacher_weighted_distortion", "sts_avg"),
    )
    for ax, (x_field, y_field) in zip(axes, panels, strict=True):
        for policy in POLICY_ORDER:
            if policy not in points or not points[policy].get(y_field):
                continue
            if x_field == "coverage":
                if policy in BASELINE_POLICIES:
                    x_mean, x_std = 0.0, 0.0
                elif policy in final_coverage:
                    x_mean, x_std = _mean_std(final_coverage[policy])
                else:
                    continue
            else:
                x_mean, x_std = _mean_std(points[policy][x_field])
            y_mean, y_std = _mean_std(points[policy][y_field])
            ax.errorbar(
                x_mean,
                y_mean,
                xerr=x_std,
                yerr=y_std,
                color=POLICY_COLOR[policy],
                marker=POLICY_MARKER[policy],
                linestyle="none",
                label=POLICY_LABEL[policy],
            )
    save(fig, out_dir / "fig_causal_chain_legacy")


def _run_metrics(run_dir: Path) -> tuple[float, float]:
    rows = read_csv(run_dir / "result.csv")
    if len(rows) != 1:
        raise ValueError(f"expected one compact result row: {run_dir}")
    return float(rows[0]["avg"]), float(rows[0]["teacher_weighted_distortion"])


def _arm_seed_values(runs_root: Path, relative_arm: str) -> list[tuple[float, float]]:
    arm_dir = runs_root / relative_arm
    seed_dirs = sorted(
        path for path in arm_dir.glob("seed*")
        if path.is_dir() and (path / "result.csv").is_file()
    )
    if not seed_dirs:
        raise ValueError(f"no seed runs found under {arm_dir}")
    return [_run_metrics(path) for path in seed_dirs]


def figure_sensitivity(runs_root: Path, out_dir: Path) -> None:
    """Plot mean and seed spread for both sensitivity axes without dual y-axes."""
    topk = {
        0.5: _arm_seed_values(runs_root, "sensitivity/topk_0_5x"),
        0.75: _arm_seed_values(runs_root, "sensitivity/topk_0_75x"),
        1.0: _arm_seed_values(runs_root, "full/full"),
        1.5: _arm_seed_values(runs_root, "sensitivity/topk_1_5x"),
        2.0: _arm_seed_values(runs_root, "sensitivity/topk_2x"),
        2.5: _arm_seed_values(runs_root, "sensitivity/topk_2_5x"),
        3.0: _arm_seed_values(runs_root, "sensitivity/topk_3x"),
    }
    row = {
        0.0: _arm_seed_values(runs_root, "components/no_row"),
        0.1: _arm_seed_values(runs_root, "sensitivity/row_0_1"),
        0.25: _arm_seed_values(runs_root, "sensitivity/row_0_25"),
        0.5: _arm_seed_values(runs_root, "sensitivity/row_0_5"),
        0.75: _arm_seed_values(runs_root, "sensitivity/row_0_75"),
        1.0: _arm_seed_values(runs_root, "full/full"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 3.45))
    studies = (
        (
            topk,
            r"Support-quota multiplier",
            [
                r"0.5$\times$",
                r"0.75$\times$",
                r"1$\times$",
                r"1.5$\times$",
                r"2$\times$",
                r"2.5$\times$",
                r"3$\times$",
            ],
        ),
        (
            row,
            r"Row-loss weight $\lambda_{\mathrm{row}}$",
            ["0", "0.1", "0.25", "0.5", "0.75", "1"],
        ),
    )
    for column, (study, x_label, tick_labels) in enumerate(studies):
        xs = list(study)
        for metric_index, (metric_name, color) in enumerate(
            (("Avg.", BLUE), (r"$\hat{E}$", ORANGE))
        ):
            ax = axes[metric_index, column]
            value_index = metric_index
            means, stds = zip(
                *[_mean_std([values[value_index] for values in study[x]]) for x in xs],
                strict=True,
            )
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                color=color,
                marker="o",
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=1.45,
                elinewidth=0.9,
                capsize=2,
                zorder=3,
            )
            default_index = xs.index(1.0)
            ax.scatter(
                [1.0],
                [means[default_index]],
                s=52,
                facecolors="none",
                edgecolors=INK,
                linewidths=0.8,
                zorder=4,
            )
            ax.axvline(1.0, color=INK_MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
            direction = r" $\uparrow$" if metric_index == 0 else r" $\downarrow$"
            ax.set_ylabel(metric_name + direction)
            ax.set_xticks(xs, tick_labels)
            ax.margins(x=0.13, y=0.22)
            if metric_index == 0:
                title = "(a) Top-$K$ quota" if column == 0 else "(b) Row objective"
                ax.set_title(title, loc="left", color=INK, pad=4)
            else:
                ax.set_xlabel(x_label)

    fig.text(
        0.5,
        0.005,
        "Open circle marks the canonical setting.",
        ha="center",
        va="bottom",
        color=INK_MUTED,
        fontsize=7,
    )
    fig.tight_layout(w_pad=1.6, h_pad=1.0, rect=(0, 0.045, 1, 1))
    save(fig, out_dir / "fig_sensitivity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path(
            "runs/ablation/qwen3_0_6b_to_minilmv2_h384/"
            "paper_r1_v2/tables/table2_coverage.csv"
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/ablation/qwen3_0_6b_to_minilmv2_h384/paper_r1_v2"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("latex/figures"))
    parser.add_argument("--only", choices=("coverage", "sensitivity"), default=None)
    args = parser.parse_args()

    use_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.only != "sensitivity":
        figure_coverage(read_csv(args.coverage), args.out_dir)
    if args.only != "coverage":
        figure_sensitivity(args.runs_root, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
