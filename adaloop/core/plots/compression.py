"""F06: accuracy against GFLOPs, kept visual tokens and KV cache bytes, one line per method, Pareto front marked."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from adaloop.core.eval.pareto import pareto_front
from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

BENCH_LABELS = {"gqa": "GQA", "textvqa": "TextVQA", "all": "Both"}
# controls are dashed with hollow markers, methods solid; cls_iqr decides its own count, so it is dash-dotted
CONTROL = {"linestyle": "--", "markerfacecolor": "white"}
STYLES = {
    "random": {"color": "#999999", "marker": "x", **CONTROL},
    "uniform": {"color": OKABE_ITO["sky_blue"], "marker": "s", **CONTROL},
    "text_mean": {"color": OKABE_ITO["reddish_purple"], "marker": "^", **CONTROL},
    "lowres": {"color": OKABE_ITO["yellow"], "marker": "D", **CONTROL},
    "crisp_lite": {"color": OKABE_ITO["orange"], "marker": "o"},
    "rediprune": {"color": OKABE_ITO["blue"], "marker": "s"},
    "divprune": {"color": OKABE_ITO["bluish_green"], "marker": "^"},
    "visionzip": {"color": OKABE_ITO["vermillion"], "marker": "D"},
    "cls_iqr": {"color": OKABE_ITO["black"], "marker": "v", "linestyle": "-."},
}
# column of the table, divisor, axis label
COSTS = (("gflops_total", 1.0, "GFLOPs per answer"), ("n_vis", 1.0, "kept visual tokens (mean)"),
         ("kv_cache_bytes", 1e6, "KV cache after prefill (MB)"))


def plot_pareto(table: pd.DataFrame) -> Figure:
    """F06 figure: one row of panels per benchmark (and "all"), one column per cost.

    `table` columns: cell, method, k (mean kept / full), benchmark, accuracy (%), gflops_total, n_vis, kv_cache_bytes
    (`pareto_table`). A line joins a method's points; the full model (method "none") is a star and a dotted horizontal
    line. Ringed points are on the panel's Pareto front (no point is both cheaper and more accurate).
    """
    columns = ["cell", "method", "k", "benchmark", "accuracy", *[column for column, _, _ in COSTS]]
    require_columns(table, columns, "table", "run the Pareto sweep in nb07 first")
    require_unique(table, ["cell", "benchmark"], "table")
    unknown = sorted(set(table["method"]) - set(STYLES) - {"none"})
    if unknown:
        raise ValueError(f"no style for methods {unknown}")
    benches = [b for b in BENCH_LABELS if b in set(table["benchmark"])]
    fig = plt.figure(figsize=(2 * ONE_COLUMN_WIDTH_IN, 0.6 + 2.0 * len(benches)), layout="constrained")
    axes = fig.subplots(len(benches), len(COSTS), squeeze=False)
    handles = {}
    for bench, row_axes in zip(benches, axes):
        panel = table[table["benchmark"] == bench]
        for ax, (column, scale, label) in zip(row_axes, COSTS):
            for method, style in STYLES.items():
                line = panel[panel["method"] == method].sort_values("k")
                if not line.empty:
                    handles[method], = ax.plot(line[column] / scale, line["accuracy"], markersize=4, linewidth=1,
                                               label=method, **style)
            full = panel[panel["method"] == "none"]
            if not full.empty:
                handles["full tokens"], = ax.plot(full[column] / scale, full["accuracy"], color="black", marker="*",
                                                  markersize=8, linestyle="none", label="full tokens")
                ax.axhline(full["accuracy"].iloc[0], color="black", linewidth=0.5, linestyle=":")
            front = panel[pareto_front(panel[column].tolist(), panel["accuracy"].tolist())]
            handles["Pareto front"], = ax.plot(front[column] / scale, front["accuracy"], marker="o", markersize=9,
                                               markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.8,
                                               linestyle="none", label="Pareto front")
            ax.set_xlabel(label)
        row_axes[0].set_ylabel(f"{BENCH_LABELS[bench]}: accuracy (%)")
    fig.legend(handles.values(), handles.keys(), loc="outside upper center", ncols=4)
    return fig
