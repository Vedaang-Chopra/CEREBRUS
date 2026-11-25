"""F03: pass p of a long run against a fresh R = p run (cosine and answer agreement)."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

# above this cosine, a linear axis would show a flat line at 1: plot the distance on a log-like axis instead
NEAR_ONE = 0.99
LINEAR_BELOW = 1e-7  # symlog: distances under float32 noise are drawn linearly, so exact 0 stays on the axis


def plot_pass_state(table: pd.DataFrame, long_R: int) -> Figure:
    """F03 figure; `long_R` is the loop count of the long run.

    `table` columns: p (int; may be the index), cosine_mean, cosine_min (in [-1, 1]),
    answer_agree (%, over items parsed in both runs).
    """
    if isinstance(table, pd.DataFrame) and table.index.name == "p":
        table = table.reset_index()
    require_columns(table, ["p", "cosine_mean", "cosine_min", "answer_agree"], "table",
                    "build it from the nb05 pass-state rows (answer_agree is NaN if nothing parsed)")
    require_unique(table, ["p"], "table")
    table = table.sort_values("p")
    p = table["p"].astype(int)

    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH_IN, 3.0), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True)

    as_distance = table["cosine_min"].min() >= NEAR_ONE
    for column, label, color, marker, line in [
        ("cosine_mean", "mean over items", OKABE_ITO["blue"], "o", "-"),
        ("cosine_min", "worst item", OKABE_ITO["vermillion"], "s", "--"),
    ]:
        values = (1 - table[column]).clip(lower=0) if as_distance else table[column]
        top.plot(p, values, color=color, marker=marker, linestyle=line, markersize=4, label=label)
    if as_distance:
        top.set_yscale("symlog", linthresh=LINEAR_BELOW)
        highest = max(float((1 - table["cosine_min"]).max()), LINEAR_BELOW)
        top.set_ylim(-0.4 * LINEAR_BELOW, highest * 10)
        top.yaxis.get_major_locator().set_params(numticks=5)
        top.set_ylabel("1 − cosine")
    else:
        top.set_ylabel("cosine")
    top.legend(loc="best")

    bottom.plot(p, table["answer_agree"], color=OKABE_ITO["black"], marker="D", markersize=4)
    bottom.set_ylim(0, 108)
    bottom.set_yticks(range(0, 101, 25))
    bottom.set_ylabel("answer\nagreement (%)")
    bottom.set_xlabel(f"pass p (R = {long_R} run vs fresh R = p run)")
    bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.align_ylabels()
    return fig
