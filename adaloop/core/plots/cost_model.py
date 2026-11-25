"""F04: analytic FLOPs per (k, R) heatmap, and analytic vs measured prefill time ratio."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from adaloop.core.plots.checks import require_columns, require_unique
from adaloop.core.plots.colors import OKABE_ITO
from adaloop.core.plots.style import ONE_COLUMN_WIDTH_IN

CELL_ROW_IN = 0.3  # heatmap height per k row


def _heatmap(ax: Axes, flops: pd.DataFrame) -> None:
    """One annotated cell per (k, R), in GFLOPs; largest k on top."""
    grid = flops.pivot(index="k", columns="R", values="flops_total").sort_index(ascending=False) / 1e9
    if grid.isna().any().any():
        raise ValueError(f"flops needs every (k, R) combination; got\n{grid}")
    image = ax.imshow(grid.values, cmap="cividis", aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j]
            dark = image.norm(value) < 0.55  # cividis is dark at the low end
            ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=7, color="white" if dark else "black")
    ax.set_xticks(range(grid.shape[1]), [str(int(R)) for R in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{k:.0%}" for k in grid.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("loop passes R")
    ax.set_ylabel("tokens kept k")
    ax.set_title("(a) FLOPs per answer (G)")


def _ratio_panel(ax: Axes, ratios: pd.DataFrame, r_ratio: tuple[int, int], tolerance: float) -> None:
    """Per item: measured and analytic time ratio against prompt length, with the accepted band."""
    ratios = ratios.sort_values("n_tokens")
    x, analytic = ratios["n_tokens"], ratios["analytic"]
    ax.fill_between(x, analytic * (1 - tolerance), analytic * (1 + tolerance), color="#999999", alpha=0.25,
                    linewidth=0, label=f"analytic ±{tolerance:.0%}")
    ax.plot(x, analytic, color=OKABE_ITO["black"], linestyle="--", label="analytic")
    ax.plot(x, ratios["measured"], color=OKABE_ITO["vermillion"], marker="o", markersize=3.5, linestyle="none",
            label="measured")
    top = max((analytic * (1 + tolerance)).max(), ratios["measured"].max())
    ax.set_ylim(0, top * 1.45)  # room for the legend above the data
    ax.legend(loc="upper center", ncols=3, handlelength=1.5, columnspacing=0.8, handletextpad=0.4)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel(f"time ratio\nR={r_ratio[1]} / R={r_ratio[0]}")
    ax.set_title("(b) prefill time ratio")


def plot_cost_model(
    flops: pd.DataFrame, ratios: pd.DataFrame, r_ratio: tuple[int, int], tolerance: float
) -> Figure:
    """F04 figure; `r_ratio` is (low R, high R), `tolerance` the accepted relative gap (0.25 = 25%).

    `flops` columns: k (fraction of tokens kept, in (0, 1]), R (int), flops_total (FLOPs); one row per (k, R).
    `ratios` columns: n_tokens, measured, analytic (time at high R / time at low R); one row per item.
    """
    require_columns(flops, ["k", "R", "flops_total"], "flops", "build it in nb06 step 4")
    require_columns(ratios, ["n_tokens", "measured", "analytic"], "ratios", "run the nb06 step 4 timing cell")
    require_unique(flops, ["k", "R"], "flops")

    heat_height = 0.6 + CELL_ROW_IN * flops["k"].nunique()
    ratio_height = 1.7
    fig = plt.figure(figsize=(ONE_COLUMN_WIDTH_IN, heat_height + ratio_height), layout="constrained")
    top, bottom = fig.subplots(2, 1, height_ratios=[heat_height, ratio_height])
    _heatmap(top, flops)
    _ratio_panel(bottom, ratios, r_ratio, tolerance)
    return fig
