"""F09: oracle per cost cap against the best fixed cell and the noise oracle."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import METHOD_COLORS
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

BENCH_LABELS = {"gqa": "GQA", "textvqa": "TextVQA"}


def plot_headroom(table: pd.DataFrame, headline_frac: float) -> Figure:
    """F09 figure: one panel per benchmark, accuracy against the cost cap (as a fraction of cost_ref).

    `table`: `headroom_table`. The grey line is best fixed + noise oracle, so the gap from it up to the oracle is the
    headroom. Caps that no cell fits are left out; the vertical line marks the headline cap.
    """
    require_columns(table, ["benchmark", "cap_frac"], "table", "build it with api.headroom_table")
    require_unique(table, ["benchmark", "cap_frac"], "table")
    benches = [b for b in BENCH_LABELS if b in set(table["benchmark"])]
    fig = plt.figure(figsize=(2 * ONE_COLUMN_WIDTH_IN, 2.6), layout="constrained")
    axes = fig.subplots(1, len(benches), squeeze=False)[0]
    handles = {}
    for ax, bench in zip(axes, benches):
        panel = table[table["benchmark"] == bench].dropna(subset=["best_fixed", "oracle"]).sort_values("cap_frac")
        if panel.empty:
            raise ValueError(f"{bench}: no cost cap fits any cell; raise headroom.cap_fracs")
        x = panel["cap_frac"]
        handles["oracle"], = ax.plot(x, panel["oracle"], color=METHOD_COLORS["oracle"], marker="o", markersize=3.5)
        handles["best fixed"], = ax.plot(x, panel["best_fixed"], color=METHOD_COLORS["best_fixed"], marker="s",
                                         markersize=3.5)
        noise = panel.dropna(subset=["noise_oracle"])
        if not noise.empty:
            handles["best fixed + noise oracle"], = ax.plot(
                noise["cap_frac"], noise["best_fixed"] + noise["noise_oracle"],
                color=METHOD_COLORS["noise_oracle"], linestyle="--", marker="^", markersize=3.5)
        ax.axvline(headline_frac, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(BENCH_LABELS[bench])
        ax.set_xlabel("mean cost cap (x cost_ref)")
    axes[0].set_ylabel("accuracy (%)")
    fig.legend(handles.values(), handles.keys(), loc="outside upper center", ncols=3)
    return fig
