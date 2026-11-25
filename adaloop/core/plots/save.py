"""Write paper figures and tables to draft (execution_results/) or final (paper/) folders."""
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from utils.paths import find_repo_root


def _out_dir(kind: str, final: bool) -> Path:
    """Return the folder for `kind` ('figures' or 'tables'): paper/ for finals, execution_results/ for drafts."""
    if final:
        return find_repo_root() / "paper" / kind
    return find_repo_root() / "execution_results" / kind


def _check_name(name: str) -> None:
    """Reject names that would escape the output folder."""
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"name must be a plain file stem, got {name!r}")


def save_figure(fig: Figure, name: str, final: bool = False) -> tuple[Path, Path]:
    """Write `<name>.pdf` and `<name>.png`; return (pdf path, png path)."""
    _check_name(name)
    folder = _out_dir("figures", final)
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / f"{name}.pdf"
    png_path = folder / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"saved figure: {pdf_path}")
    return pdf_path, png_path


def save_table(
    df: pd.DataFrame, name: str, final: bool = False, float_format: str = "%.2f"
) -> tuple[Path, Path]:
    """Write `<name>.csv` and `<name>.tex` (no index); floats in the LaTeX use `float_format`. Returns (csv path, tex path)."""
    _check_name(name)
    folder = _out_dir("tables", final)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{name}.csv"
    tex_path = folder / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, escape=True, float_format=float_format)
    print(f"saved table: {csv_path}")
    return csv_path, tex_path
