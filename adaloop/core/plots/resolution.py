"""F09b: accuracy against visual tokens per image size, all tokens against DivPrune."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

BENCH_LABELS = {"gqa": "GQA", "textvqa": "TextVQA", "all": "Both"}
RES_COLORS = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"], OKABE_ITO["reddish_purple"]]
SETTINGS = {"full": ("o", "all tokens"), "matched": ("^", "DivPrune, matched count"),
            "fixed_k": ("s", "DivPrune, fixed k")}


def plot_resolution(table: pd.DataFrame) -> Figure:
    """F09b figure.

    `table` columns: cell, setting (full | matched | fixed_k), res_px, benchmark, n_vis (mean kept tokens),
    accuracy (%). One panel per benchmark; colour = image size, marker = setting; a grey line joins the
    all-token points.
    """
    require_columns(table, ["cell", "setting", "res_px", "benchmark", "n_vis", "accuracy"], "table",
                    "run nb04 with RUN_RESOLUTION = True")
    require_unique(table, ["cell", "benchmark"], "table")
    sizes = sorted(table["res_px"].unique())
    if len(sizes) > len(RES_COLORS):
        raise ValueError(f"at most {len(RES_COLORS)} image sizes, got {sizes}")
    colors = dict(zip(sizes, RES_COLORS))
    benches = [b for b in BENCH_LABELS if b in set(table["benchmark"])]
    fig = plt.figure(figsize=(2 * ONE_COLUMN_WIDTH_IN, 2.6), layout="constrained")
    axes = fig.subplots(1, len(benches), squeeze=False)[0]
    for ax, bench in zip(axes, benches):
        panel = table[table["benchmark"] == bench]
        full = panel[panel["setting"] == "full"].sort_values("n_vis")
        ax.plot(full["n_vis"], full["accuracy"], color="#999999", linewidth=0.8, zorder=1)
        for row in panel.itertuples():
            filled = row.setting == "full"
            ax.plot(row.n_vis, row.accuracy, marker=SETTINGS[row.setting][0], markersize=6, linestyle="none",
                    color=colors[row.res_px], markerfacecolor=colors[row.res_px] if filled else "none", zorder=2)
        ax.set_title(BENCH_LABELS[bench])
        ax.set_xlabel("visual tokens (mean)")
    axes[0].set_ylabel("accuracy (%)")
    handles = [Line2D([], [], color=colors[px], linewidth=3, label=f"{px} px") for px in sizes]
    handles += [Line2D([], [], color="black", marker=marker, linestyle="none",
                       markerfacecolor="black" if name == "full" else "none", label=label)
                for name, (marker, label) in SETTINGS.items() if name in set(table["setting"])]
    fig.legend(handles=handles, loc="outside upper center", ncols=3)
    return fig
