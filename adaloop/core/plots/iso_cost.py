"""F10: accuracy against mean cost per method, with the headline cost marked."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from adaloop.core.plots.checks import require_columns
from adaloop.core.plots.colors import METHOD_COLORS
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

MARKERS = {"controller": "o", "controller_no_h": "o", "heuristic": "D", "best_fixed": "s", "best_fixed_cell": "+",
           "early_exit_only": "^",
           "tokens_only": "v", "upfront_15way": "P", "random": "x", "oracle": "*"}
DASHED = ("controller_no_h", "best_fixed_cell", "random", "oracle")


def plot_iso_cost(points: pd.DataFrame, headline_frac: float, title: str) -> Figure:
    """F10 figure from `iso_cost_points` (columns method, cost_frac, accuracy, chosen); `title` names the split."""
    require_columns(points, ["method", "cost_frac", "accuracy", "chosen"], "points", "build it with api.iso_cost_points")
    fig = plt.figure(figsize=(2 * ONE_COLUMN_WIDTH_IN, 3.2), layout="constrained")
    ax = fig.subplots()
    for method, marker in MARKERS.items():
        curve = points[(points["method"] == method) & points["chosen"]].sort_values("cost_frac")
        if curve.empty:
            continue
        ax.plot(curve["cost_frac"], curve["accuracy"], color=METHOD_COLORS[method], marker=marker, markersize=3.5,
                linestyle="--" if method in DASHED else "-", label=method.replace("_", " "))
    ax.axvline(headline_frac, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("mean FLOPs (x cost_ref)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    fig.legend(loc="outside right center")
    return fig
