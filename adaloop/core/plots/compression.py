"""F06: accuracy against FLOPs (and latency) per compression method, one panel per benchmark."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

BENCH_LABELS = {"gqa": "GQA", "textvqa": "TextVQA"}
# controls are dashed with hollow markers, methods solid
CONTROL_STYLES = {
    "random": {"color": "#999999", "marker": "x"},
    "uniform": {"color": OKABE_ITO["sky_blue"], "marker": "s"},
    "text_mean": {"color": OKABE_ITO["reddish_purple"], "marker": "^"},
    "lowres": {"color": OKABE_ITO["yellow"], "marker": "D"},
}
METHOD_STYLES = {
    "crisp_lite": {"color": OKABE_ITO["orange"], "marker": "o"},
    "rediprune": {"color": OKABE_ITO["blue"], "marker": "s"},
    "divprune": {"color": OKABE_ITO["bluish_green"], "marker": "^"},
    "visionzip": {"color": OKABE_ITO["vermillion"], "marker": "D"},
}


def plot_compression(table: pd.DataFrame) -> Figure:
    """F06 figure.

    `table` columns: method, k, benchmark ("all" lines are ignored), accuracy (%), gflops_total. The full model
    (method "none") is a black star and a horizontal line. With a `latency_s` column that has values, a second row of
    panels shows accuracy against latency (batch 1, one copy).
    """
    require_columns(table, ["method", "k", "benchmark", "accuracy", "gflops_total"], "table",
                    "run the comparison in nb07 first")
    require_unique(table, ["method", "k", "benchmark"], "table")
    rows = table[table["benchmark"] != "all"]
    unknown = sorted(set(rows["method"]) - set(CONTROL_STYLES) - set(METHOD_STYLES) - {"none"})
    if unknown:
        raise ValueError(f"no style for methods {unknown}")
    benches = sorted(rows["benchmark"].unique())
    costs = [("gflops_total", "GFLOPs per answer")]
    if "latency_s" in rows and rows["latency_s"].notna().any():
        costs.append(("latency_s", "latency per answer (s)"))
    fig = plt.figure(figsize=(2 * ONE_COLUMN_WIDTH_IN, 0.5 + 2.1 * len(costs)), layout="constrained")
    axes = fig.subplots(len(costs), len(benches), squeeze=False)
    handles = {}
    for (column, label), row_axes in zip(costs, axes):
        for ax, bench in zip(row_axes, benches):
            panel = rows[rows["benchmark"] == bench].dropna(subset=[column])
            for styles, extra in ((CONTROL_STYLES, {"linestyle": "--", "markerfacecolor": "white"}),
                                  (METHOD_STYLES, {"linestyle": "-"})):
                for method, style in styles.items():
                    line = panel[panel["method"] == method].sort_values("k")
                    if not line.empty:
                        handles[method], = ax.plot(line[column], line["accuracy"], markersize=4, label=method,
                                                   **style, **extra)
            full = panel[panel["method"] == "none"]
            if not full.empty:
                handles["full tokens"], = ax.plot(full[column], full["accuracy"], color="black", marker="*",
                                                  markersize=8, linestyle="none", label="full tokens")
                ax.axhline(full["accuracy"].iloc[0], color="black", linewidth=0.5, linestyle=":")
            ax.set_title(BENCH_LABELS.get(bench, bench))
            ax.set_xlabel(label)
        row_axes[0].set_ylabel("accuracy (%)")
    fig.legend(handles.values(), handles.keys(), loc="outside upper center", ncols=5)
    return fig
