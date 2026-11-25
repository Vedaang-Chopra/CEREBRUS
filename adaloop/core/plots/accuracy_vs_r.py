"""F05: accuracy against loop passes R with all visual tokens, one panel per benchmark."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

BENCH_LABELS = {"gqa": "GQA", "textvqa": "TextVQA"}
# splits differ in colour, marker, line style and fill
SPLIT_STYLES = {
    "dev": {"color": OKABE_ITO["blue"], "marker": "o", "linestyle": "-"},
    "ctrl_train": {"color": OKABE_ITO["orange"], "marker": "s", "linestyle": "--", "markerfacecolor": "white"},
}


def plot_accuracy_vs_r(summary: pd.DataFrame) -> Figure:
    """F05 figure.

    `summary` columns: benchmark (str), split ("dev" | "ctrl_train"), R (int), accuracy (%).
    One row per benchmark x split x R. Each panel has its own accuracy range.
    """
    require_columns(summary, ["benchmark", "split", "R", "accuracy"], "summary",
                    "nb06 step 3 found no result files (run the full run first)")
    require_unique(summary, ["benchmark", "split", "R"], "summary")
    unknown = sorted(set(summary["split"]) - set(SPLIT_STYLES))
    if unknown:
        raise ValueError(f"unknown splits {unknown}; known: {list(SPLIT_STYLES)}")

    benches = sorted(summary["benchmark"].unique())
    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH_IN, 2.1), layout="constrained")
    axes = fig.subplots(1, len(benches), squeeze=False)[0]
    handles = {}
    for ax, bench in zip(axes, benches):
        rows = summary[summary["benchmark"] == bench]
        for split, style in SPLIT_STYLES.items():
            line = rows[rows["split"] == split].sort_values("R")
            if not line.empty:
                handles[split], = ax.plot(line["R"], line["accuracy"], markersize=4, label=split, **style)
        ax.set_title(BENCH_LABELS.get(bench, bench))
        ax.set_xlabel("loop passes R")
        ax.set_xlim(summary["R"].min() - 0.3, summary["R"].max() + 0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # at least a 5-point span, so noise between nearly equal values does not look like a trend
        low, high = rows["accuracy"].min(), rows["accuracy"].max()
        pad = max(0.1 * (high - low), (5 - (high - low)) / 2)
        ax.set_ylim(low - pad, high + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].set_ylabel("accuracy (%)")
    fig.legend(handles.values(), handles.keys(), loc="outside upper center", ncols=len(handles))
    return fig
