r"""Matplotlib styling for the ICLR submission's figures.

Two things a paper figure has to get right that a screen figure does not.

*It is printed at a fixed width and never rescaled.* `latex/iclr2027_conference.sty`
sets `\textwidth` to 5.5 true inches, so a figure drawn at `FULL_WIDTH` and
included with `\includegraphics[width=\textwidth]` is reproduced 1:1 and its 8pt
labels are 8pt on paper. Drawing at some other size and letting LaTeX scale it is
what produces the mismatched, too-small tick labels that mark a figure as
unfinished.

*Its fonts must be embedded and its geometry must stay vector.* `fonttype 42`
embeds TrueType outlines rather than emitting Type-3 bitmaps, which several
camera-ready checkers reject outright; the serif stack matches the template's
`times`, so the figure's text and the body text are the same typeface.

The palette is the categorical set from the data-viz reference palette, in fixed
slot order, validated for all-pairs colour-vision separation (worst CVD dE 9.2,
worst normal-vision dE 16.3 on a light surface). Aqua sits below 3:1 contrast on
white, so every series that uses it is *directly labelled* rather than left to a
legend swatch -- and direct labels plus per-series markers and dashes mean the
figures survive greyscale printing, where colour carries nothing at all.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# latex/iclr2027_conference.sty: \textwidth 5.5 true in
FULL_WIDTH = 5.5
HALF_WIDTH = 2.68  # two side-by-side figures with a gutter

# Fixed slot order. Colour follows the entity, never its rank: `POLICY_COLOR`
# below pins a hue to each policy so a figure that drops an arm does not repaint
# the survivors.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2dd"

# Two neutral steps for the baselines. They are deliberately NOT categorical
# slots: a baseline is a different kind of thing from a policy, grey reads that
# way at a glance, and keeping them off the hue scale means the four policies
# stay the validated all-pairs four rather than becoming an unvalidated six.
# Both clear 3:1 against the white surface and are separated by dE 19 -- checked
# with the palette validator, same as the categorical set. The obvious lighter
# grey does not clear contrast, and a thin 1.3pt line at 2.5:1 disappears in
# print.
BASELINE = ("#52514e", "#8a8880")

# The four support-selection arms of S1, in the order they are argued about:
# the two deterministic-ish controls, then the stochastic control, then ours.
# The baselines lead, because they are where the argument starts.
POLICY_COLOR = {
    "batch_local": BASELINE[0],
    "uniform_corpus": BASELINE[1],
    "uniform": SERIES[3],
    "topk": SERIES[1],
    "proportional": SERIES[2],
    "hybrid": SERIES[0],
}
POLICY_MARKER = {
    "batch_local": "v",
    "uniform_corpus": "P",
    "uniform": "s",
    "topk": "^",
    "proportional": "D",
    "hybrid": "o",
}
POLICY_DASH = {
    "batch_local": (0, (1, 2)),
    "uniform_corpus": (0, (3, 2)),
    "uniform": (0, (1, 1.4)),
    "topk": (0, (4, 1.6)),
    "proportional": (0, (4, 1.4, 1, 1.4)),
    "hybrid": (0, ()),
}
POLICY_LABEL = {
    "batch_local": "Batch-local RKD",
    "uniform_corpus": "Uniform corpus (matched)",
    "uniform": "Uniform",
    "topk": "Teacher top-$K$",
    "proportional": "Teacher-proportional",
    "hybrid": "Head + prop. tail (ours)",
}
POLICY_ORDER = (
    "batch_local",
    "uniform_corpus",
    "uniform",
    "topk",
    "proportional",
    "hybrid",
)
# Arms that form no graph relations at all. Their exposed graph-relation mass is
# zero by construction, not by measurement, so nothing replays them -- Figure 2
# places them at coverage 0 and Figure 1 leaves them out, having no curve to draw.
BASELINE_POLICIES = ("batch_local", "uniform_corpus")


def use_paper_style() -> None:
    """Install the rcParams. Call once, before creating any figure."""
    mpl.rcParams.update(
        {
            # Embed TrueType outlines. Type-3 (the default, 3) is a camera-ready
            # rejection in its own right.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
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
            "figure.titlesize": 9,
            # Recessive frame: the data is the only thing with full-strength ink.
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.6,
            "legend.frameon": False,
            "legend.handlelength": 1.9,
            "legend.columnspacing": 1.1,
            "legend.labelspacing": 0.35,
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, path_stem: str) -> None:
    """Write the PDF the paper includes, plus a PNG for reading in a terminal."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png")
    plt.close(fig)
    print(f"wrote {path_stem}.pdf and {path_stem}.png")
