"""F02: parse rate and accuracy per answer-path variant, and optionally TextVQA accuracy per image size."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

BENCH_LABELS = {"gqa": "GQA", "textvqa": "TextVQA"}
# colour and hatch per benchmark, so the bars also differ in greyscale
BENCH_STYLES = [(OKABE_ITO["blue"], ""), (OKABE_ITO["orange"], "///")]


def _label_bars(ax: Axes, bars) -> None:
    ax.bar_label(bars, fmt="%.0f", fontsize=6, padding=1)


def _variant_panel(ax: Axes, variants: pd.DataFrame, metric: str, title: str) -> None:
    """Grouped bars: one group per variant, one bar per benchmark."""
    names = sorted(variants["variant"].unique())
    benches = sorted(variants["benchmark"].unique())
    width = 0.8 / len(benches)
    values = variants.set_index(["variant", "benchmark"])[metric]
    for j, (bench, (color, hatch)) in enumerate(zip(benches, BENCH_STYLES)):
        x = np.arange(len(names)) + (j - (len(benches) - 1) / 2) * width
        heights = [values.get((name, bench), np.nan) for name in names]
        bars = ax.bar(x, heights, width, color=color, hatch=hatch, edgecolor="white", linewidth=0.5,
                      label=BENCH_LABELS.get(bench, bench))
        _label_bars(ax, bars)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlabel("variant")
    ax.set_title(title)
    ax.set_ylim(0, 115)  # headroom for the value labels
    ax.set_yticks(range(0, 101, 25))


def plot_thinking_off(variants: pd.DataFrame, resolution: pd.DataFrame | None = None) -> Figure:
    """F02 figure.

    `variants` columns: variant (str), benchmark ("gqa" | "textvqa"), parse_rate (%), accuracy (%).
    `resolution` columns: res_px (int, long side), accuracy (%), n_vis_mean (visual tokens); TextVQA only.
    Without it (None) the figure has panels (a) and (b) only.
    """
    require_columns(variants, ["variant", "benchmark", "parse_rate", "accuracy"], "variants",
                    "run nb04 with RUN_VARIANTS = True")
    require_unique(variants, ["variant", "benchmark"], "variants")
    if resolution is not None:
        require_columns(resolution, ["res_px", "accuracy", "n_vis_mean"], "resolution",
                        "run nb04 with RUN_RESOLUTION = True")
        require_unique(resolution, ["res_px"], "resolution")
    if variants["benchmark"].nunique() > len(BENCH_STYLES):
        raise ValueError(f"at most {len(BENCH_STYLES)} benchmarks, got {sorted(variants['benchmark'].unique())}")

    rows, ratios = [["parse", "acc"]], [1.25]
    if resolution is not None:
        rows, ratios = [*rows, ["res", "res"]], [*ratios, 1]
    height = 2.0 if resolution is None else 3.4
    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH_IN, height), layout="constrained")
    axes = fig.subplot_mosaic(rows, height_ratios=ratios)
    _variant_panel(axes["parse"], variants, "parse_rate", "(a) parse rate (%)")
    _variant_panel(axes["acc"], variants, "accuracy", "(b) accuracy (%)")
    handles, labels = axes["parse"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels))

    if resolution is None:
        return fig

    res = resolution.sort_values("res_px")
    ax = axes["res"]
    bars = ax.barh(range(len(res)), res["accuracy"], 0.6, color=BENCH_STYLES[1][0], hatch=BENCH_STYLES[1][1],
                   edgecolor="white", linewidth=0.5)
    _label_bars(ax, bars)
    ax.set_yticks(range(len(res)), [f"{int(r.res_px)} px\n{r.n_vis_mean:.0f} tokens" for r in res.itertuples()])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("accuracy (%)")
    ax.set_title("(c) TextVQA by image long side")
    return fig
