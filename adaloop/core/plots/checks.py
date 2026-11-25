"""Input checks shared by the figure functions."""
from collections.abc import Sequence

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Sequence[str], name: str, hint: str) -> None:
    """Raise ValueError unless `df` is a non-empty DataFrame with `columns` and no missing values in them."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{name} is missing or empty: {hint}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} lacks columns {missing}; it has {list(df.columns)}")
    empty = [c for c in columns if df[c].isna().any()]
    if empty:
        raise ValueError(f"{name} has missing values in {empty}: {hint}")


def require_unique(df: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    """Raise ValueError if two rows share the same `keys`."""
    if df.duplicated(list(keys)).any():
        raise ValueError(f"{name} has more than one row for the same {list(keys)}")
