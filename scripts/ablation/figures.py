r"""The two main-paper ablation figures.

Figure 1 -- cumulative teacher mass ever exposed, by support policy and epoch.
Figure 2 -- the causal chain in two panels: coverage against geometry distortion,
            and distortion against the downstream STS average.

Both read files produced by the other scripts in this directory:
`replay_coverage.py` for the coverage curves (offline, no training) and
`collect.py --csv` for the trained arms' distortion and scores. Neither figure
re-runs anything, so re-drawing after a style change costs a second.

Drawing conventions, and why each one is not a preference:

*One frameless legend per figure, below the axes.* Direct end-of-line labels were
tried first and are better when series stay apart -- but three of the four
policies converge by design (that convergence is the result), so their labels
collide, and a label placed outside the axes makes the saved width depend on the
text rather than on the template's 5.5 in. The legend keeps identity visible
next to a swatch, which is also what discharges the palette's one contrast
warning: aqua is below 3:1 on white and is legal only when labelled.

*A marker shape and a dash pattern per policy, on top of colour.* Papers get
printed in greyscale and read by colour-blind reviewers. Identity survives both.

*Seed error bars on every point in Figure 2.* The claims are differences of a
tenth of a point on three seeds. A point without its spread is not evidence, and
a figure that hides the spread is arguing rather than reporting.

Usage:
    python scripts/ablation/figures.py \
        --coverage runs/ablation/analysis/coverage.csv \
        --results  runs/ablation/analysis/results.csv \
        --out-dir  latex/figures
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperstyle import (  # noqa: E402
    BASELINE_POLICIES,
    FULL_WIDTH,
    INK,
    INK_MUTED,
    POLICY_COLOR,
    POLICY_DASH,
    POLICY_LABEL,
    POLICY_MARKER,
    POLICY_ORDER,
    save,
    use_paper_style,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value: str | None) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_std(values: list[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def figure_one(coverage_rows: list[dict], out_dir: Path) -> None:
    """Cumulative exposed teacher mass against epoch, one line per policy.

    The shape is the argument. Top-K draws the same columns every epoch, so its
    line is flat after epoch 1: extra epochs buy it no new supervision at all.
    Proportional and uniform sampling can expose new columns over time, but they
    trade away teacher-mass coverage on each update. The figure therefore shows
    the distinction between immediate support quality and cumulative exploration.

    The S1 baselines are absent by construction rather than by omission: neither
    batch-local RKD nor the uniform-corpus control forms a graph relation, so
    each would be a flat line on zero. That belongs in the caption and the table,
    not as two overlapping curves on the axis.
    """
    series: dict[str, dict[int, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in coverage_rows:
        series[row["policy"]][int(row["epoch"])].append(
            (
                float(row["coverage_mean"]),
                float(row["coverage_p10"]),
                float(row["coverage_p90"]),
            )
        )

    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    handles = []
    right = 0
    for policy in POLICY_ORDER:
        if policy not in series:
            continue
        epochs = sorted(series[policy])
        right = max(right, epochs[-1])
        means = [statistics.fmean(v[0] for v in series[policy][e]) for e in epochs]
        low = [statistics.fmean(v[1] for v in series[policy][e]) for e in epochs]
        high = [statistics.fmean(v[2] for v in series[policy][e]) for e in epochs]
        color = POLICY_COLOR[policy]
        # The band is the anchor-to-anchor spread (p10-p90), not the seed spread:
        # coverage barely moves across seeds, and the honest source of variation
        # is that anchors differ enormously in how concentrated their pool is.
        ax.fill_between(epochs, low, high, color=color, alpha=0.09, linewidth=0)
        (line,) = ax.plot(
            epochs,
            means,
            color=color,
            linestyle=POLICY_DASH[policy],
            marker=POLICY_MARKER[policy],
            markeredgecolor="white",
            markeredgewidth=0.5,
            # Ours reads slightly heavier without a second colour channel.
            linewidth=1.6 if policy == "topk" else 1.2,
            label=POLICY_LABEL[policy],
            zorder=3,
        )
        handles.append(line)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Cumulative exposed mass $1-\delta_T$")
    ax.set_xticks(range(1, int(right) + 1))
    ax.set_xlim(0.9, right + 0.1)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        borderaxespad=0.0,
    )
    save(fig, str(out_dir / "fig1_support_coverage"))


def figure_two(result_rows: list[dict], coverage_rows: list[dict], out_dir: Path) -> None:
    """Coverage -> geometry -> downstream, as two scatters sharing a middle axis.

    Read left to right: if higher coverage does not buy lower distortion, or lower
    distortion does not buy a higher STS average, the chain is broken at that
    panel and the paper is entitled to claim an empirical gain but not the
    mechanism. Panel (b) plots on its x-axis exactly the quantity panel (a) ends
    on, so the two compose and a reader can follow one policy across both.

    S1 identifies the model pair. Full runs from transfer experiments must not
    enter its Top-k mean or seed error bars. Coverage must be for the same pair.
    """
    s1_pairs = {row.get("pair") for row in result_rows if row.get("ablation") == "s1"}
    if len(s1_pairs) != 1 or not all(s1_pairs):
        raise ValueError(
            "Figure 2 requires S1 results for exactly one model pair; "
            "filter the results CSV and use that pair's coverage replay."
        )
    pair = s1_pairs.pop()
    print(f"figure 2 model pair: {pair}")

    # Coverage at the last replayed epoch: the total supervision an arm ever saw.
    final_coverage: dict[str, list[float]] = defaultdict(list)
    last_epoch = max(int(r["epoch"]) for r in coverage_rows)
    for row in coverage_rows:
        if int(row["epoch"]) == last_epoch:
            final_coverage[row["policy"]].append(float(row["coverage_mean"]))

    # The support arms only: S1 plus the full model, which is the Top-k arm.
    points: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in result_rows:
        if row.get("pair") != pair or row.get("ablation") not in ("s1", "full"):
            continue
        # `policy` is collect.py's canonical label, which the two baselines need:
        # neither draws support, so both leave `support_policy` at its default and
        # would otherwise be plotted as the method.
        policy = row.get("policy") or row.get("support_policy") or ""
        if policy not in POLICY_ORDER:
            continue
        for field in ("teacher_weighted_distortion", "sts_avg", "avg", "pair_avg"):
            value = _float(row.get(field))
            if value is not None:
                points[policy][field].append(value)

    missing = [p for p in POLICY_ORDER if p not in points]
    if missing:
        print(f"  [warn] figure 2: no trained runs for policy {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))
    panels = (
        (
            axes[0],
            "coverage",
            "teacher_weighted_distortion",
            r"Cumulative exposed mass $1-\delta_T$",
            r"Distortion $\hat{E}$",
            "(a) coverage $\\rightarrow$ geometry",
        ),
        (
            axes[1],
            "teacher_weighted_distortion",
            "sts_avg",
            r"Distortion $\hat{E}$",
            "STS Avg",
            "(b) geometry $\\rightarrow$ downstream",
        ),
    )

    handles = []
    for panel_index, (ax, x_field, y_field, x_label, y_label, title) in enumerate(
        panels
    ):
        drawn = []
        for policy in POLICY_ORDER:
            if policy not in points or not points[policy].get(y_field):
                continue
            if x_field == "coverage":
                if policy in BASELINE_POLICIES:
                    # Zero by construction, not by measurement: neither baseline
                    # supervises a single graph relation, so there is no draw to
                    # replay and nothing that could make this nonzero. Placing
                    # them at 0 is the claim, and it anchors the left end of the
                    # trend the panel is about.
                    x_mean, x_err = 0.0, 0.0
                elif policy not in final_coverage:
                    continue
                else:
                    x_mean, x_err = _mean_std(final_coverage[policy])
            else:
                if not points[policy].get(x_field):
                    continue
                x_mean, x_err = _mean_std(points[policy][x_field])
            y_mean, y_err = _mean_std(points[policy][y_field])
            # Baselines are reference markers, not points on the guide: they
            # differ from the method in more than the support policy, so joining
            # them to the policy trend would assert a comparability the arms do
            # not have. (They would also draw a vertical segment, both sitting at
            # coverage 0.)
            if policy not in BASELINE_POLICIES:
                drawn.append((x_mean, y_mean))
            container = ax.errorbar(
                x_mean,
                y_mean,
                xerr=x_err,
                yerr=y_err,
                color=POLICY_COLOR[policy],
                marker=POLICY_MARKER[policy],
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                elinewidth=0.9,
                capsize=1.8,
                capthick=0.9,
                linestyle="none",
                label=POLICY_LABEL[policy],
                zorder=3,
            )
            if panel_index == 0:
                handles.append(container)
        # A guide through the support policies in x-order, not a fit: with four
        # points a regression line would assert a functional form the data cannot
        # support. The line only makes the ordering readable, which is the claim.
        if len(drawn) > 1:
            drawn.sort()
            ax.plot(
                [p[0] for p in drawn],
                [p[1] for p in drawn],
                color=INK_MUTED,
                linewidth=0.7,
                linestyle=(0, (2, 2)),
                zorder=1,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, loc="left", color=INK, pad=5)
        ax.margins(x=0.16, y=0.22)

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        borderaxespad=0.0,
    )
    fig.tight_layout(w_pad=2.2)
    save(fig, str(out_dir / "fig2_causal_chain"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", default="runs/ablation/analysis/coverage.csv")
    parser.add_argument("--results", default="runs/ablation/analysis/results.csv")
    parser.add_argument("--out-dir", default="latex/figures")
    parser.add_argument(
        "--only",
        choices=["1", "2"],
        default=None,
        help="draw a single figure instead of both",
    )
    args = parser.parse_args()

    use_paper_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_path, results_path = Path(args.coverage), Path(args.results)
    coverage_rows = read_csv(coverage_path) if coverage_path.exists() else []
    result_rows = read_csv(results_path) if results_path.exists() else []

    if args.only != "2":
        if not coverage_rows:
            print(f"skipping figure 1: {coverage_path} not found "
                  "(run replay_coverage.py first)")
        else:
            figure_one(coverage_rows, out_dir)
    if args.only != "1":
        if not coverage_rows or not result_rows:
            print("skipping figure 2: needs both the coverage replay and "
                  "collect.py --csv")
        else:
            figure_two(result_rows, coverage_rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
