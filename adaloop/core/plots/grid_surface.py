"""F07: accuracy over the k x R grid. F08: accuracy against GFLOPs for the grid cells, Pareto front marked."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from adaloop.core.eval.pareto import pareto_front
from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

BENCH_LABELS = {"gqa": "GQA", "textvqa": "TextVQA"}
K_COLORS = [OKABE_ITO["orange"], OKABE_ITO["bluish_green"], OKABE_ITO["blue"], OKABE_ITO["reddish_purple"]]
K_MARKERS = ["o", "^", "s", "D"]


def _benches(table: pd.DataFrame) -> list[str]:
    return [b for b in BENCH_LABELS if b in set(table["benchmark"])]


def plot_kr_accuracy(table: pd.DataFrame) -> Figure:
    """F07 figure: one heatmap per benchmark, each with its own colour scale.

    `table` columns: benchmark, k, R, accuracy (%) (`cell_table`).
    """
    require_columns(table, ["benchmark", "k", "R", "accuracy"], "table", "run the grid in nb08 first")
    require_unique(table, ["benchmark", "k", "R"], "table")
    benches = _benches(table)
    height = 0.8 + 0.35 * table["k"].nunique()
    fig = plt.figure(figsize=(2 * ONE_COLUMN_WIDTH_IN, height), layout="constrained")
    axes = fig.subplots(1, len(benches), squeeze=False)[0]
    for ax, bench in zip(axes, benches):
        grid = table[table["benchmark"] == bench].pivot(index="k", columns="R", values="accuracy")
        grid = grid.sort_index(ascending=False)  # largest k on top
        if grid.isna().any().any():
            raise ValueError(f"{bench} needs every (k, R) combination; got\n{grid}")
        image = ax.imshow(grid.values, cmap="cividis", aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid.values[i, j]
                dark = image.norm(value) < 0.55  # cividis is dark at the low end
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")
        ax.set_xticks(range(grid.shape[1]), [str(int(R)) for R in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [f"{k:.0%}" for k in grid.index])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("loop passes R")
        ax.set_title(f"{BENCH_LABELS[bench]}: accuracy (%)")
    axes[0].set_ylabel("tokens kept k")
    return fig


def plot_cell_pareto(table: pd.DataFrame) -> Figure:
    """F08 figure: one panel per benchmark; colour = k, the number beside a point = R, ringed = on the Pareto front.

    `table` columns: benchmark, k, R, accuracy (%), gflops_total (`cell_table`).
    """
    require_columns(table, ["benchmark", "k", "R", "accuracy", "gflops_total"], "table", "run the grid in nb08 first")
    require_unique(table, ["benchmark", "k", "R"], "table")
    ks = sorted(table["k"].unique())
    if len(ks) > len(K_COLORS):
        raise ValueError(f"at most {len(K_COLORS)} values of k, got {ks}")
    benches = _benches(table)
    fig = plt.figure(figsize=(2 * ONE_COLUMN_WIDTH_IN, 2.8), layout="constrained")
    axes = fig.subplots(1, len(benches), squeeze=False)[0]
    handles = {}
    for ax, bench in zip(axes, benches):
        panel = table[table["benchmark"] == bench]
        for k, color, marker in zip(ks, K_COLORS, K_MARKERS):
            line = panel[panel["k"] == k].sort_values("R")
            handles[f"k = {k:.0%}"], = ax.plot(line["gflops_total"], line["accuracy"], color=color, marker=marker,
                                               markersize=4, linewidth=0.8)
            for row in line.itertuples():
                ax.annotate(str(int(row.R)), (row.gflops_total, row.accuracy), textcoords="offset points",
                            xytext=(3, 3), fontsize=6, color=color)
        front = panel[pareto_front(panel["gflops_total"].tolist(), panel["accuracy"].tolist())]
        front = front.sort_values("gflops_total")
        handles["Pareto front"], = ax.plot(front["gflops_total"], front["accuracy"], color="black", linestyle=":",
                                           linewidth=0.8, marker="o", markersize=9, markerfacecolor="none",
                                           markeredgewidth=0.8)
        ax.set_title(BENCH_LABELS[bench])
        ax.set_xlabel("GFLOPs per answer (mean)")
    axes[0].set_ylabel("accuracy (%)")
    fig.legend(handles.values(), handles.keys(), loc="outside upper center", ncols=len(handles))
    return fig
