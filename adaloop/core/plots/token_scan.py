"""F04b: prefill time and decode time per step against the number of visual tokens, per R and batch size."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

R_COLORS = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"], OKABE_ITO["reddish_purple"],
            OKABE_ITO["vermillion"]]
METRICS = {"prefill_ms": "prefill (ms)", "decode_ms_per_step": "decode (ms per step)"}


def plot_token_scan(table: pd.DataFrame) -> Figure:
    """F04b figure.

    `table` columns: method (none | random), R, batch_size, n_vis (mean), prefill_ms, decode_ms_per_step.
    One column per batch size, one row per time; colour = R, line = random, star = all tokens (`none`).
    """
    require_columns(table, ["method", "R", "batch_size", "n_vis", *METRICS], "table",
                    "run nb06 step 7 with RUN_PROFILE = True")
    # `kept` is empty on the `none` lines, so it is not in require_columns
    require_unique(table.assign(kept=table["kept"].fillna(-1)), ["method", "kept", "R", "batch_size"], "table")
    loops, batches = sorted(table["R"].unique()), sorted(table["batch_size"].unique())
    if len(loops) > len(R_COLORS):
        raise ValueError(f"at most {len(R_COLORS)} loop counts, got {loops}")
    colors = dict(zip(loops, R_COLORS))
    fig = plt.figure(figsize=(len(batches) * ONE_COLUMN_WIDTH_IN, 4.2), layout="constrained")
    axes = fig.subplots(len(METRICS), len(batches), squeeze=False, sharex=True)
    for col, batch_size in enumerate(batches):
        for row, (metric, label) in enumerate(METRICS.items()):
            ax = axes[row][col]
            for R in loops:
                cell = table[(table["batch_size"] == batch_size) & (table["R"] == R)]
                kept = cell[cell["method"] == "random"].sort_values("n_vis")
                full = cell[cell["method"] == "none"]
                ax.plot(kept["n_vis"], kept[metric], color=colors[R], marker="o", markersize=3)
                ax.plot(full["n_vis"], full[metric], color=colors[R], marker="*", markersize=8, linestyle="none")
            ax.set_ylim(bottom=0)
            ax.set_ylabel(label if col == 0 else "")
        axes[0][col].set_title(f"batch {batch_size}")
        axes[-1][col].set_xlabel("visual tokens")
    handles = [Line2D([], [], color=colors[R], linewidth=3, label=f"R = {R}") for R in loops]
    handles += [Line2D([], [], color="black", marker="o", markersize=3, label="random"),
                Line2D([], [], color="black", marker="*", linestyle="none", label="all tokens")]
    fig.legend(handles=handles, loc="outside upper center", ncols=3)
    return fig
