"""Shared matplotlib style for paper figures."""
import matplotlib as mpl
from cycler import cycler

from adaloop.core.plots.colors import OKABE_ITO

ONE_COLUMN_WIDTH_IN = 3.3

# Black and yellow are left out of the line cycle (poor contrast on white for lines).
CYCLE_COLORS = [
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["reddish_purple"],
]


def set_paper_style() -> None:
    """Apply paper rcParams: one-column width, 7-8 pt text, Type 42 (TrueType) fonts in PDF, colour-blind-safe cycle."""
    mpl.rcParams.update({
        "figure.figsize": (ONE_COLUMN_WIDTH_IN, 2.4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=CYCLE_COLORS),
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
