"""S14: iso-cost curves, accuracy at a cost target, paired bootstrap with Holm, and the headroom gate.

All accuracies are the macro-average of the benchmarks and all costs are mean analytic FLOPs / mean cost_ref (equal
item counts per benchmark are required, so both are plain item means). Choices are made on `splits.dev`; the report
split is the last of `splits.eval`. Numbers on the split they were chosen on are labelled as optimistic.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from adaloop.core.controller.curves import interpolation_weights
from adaloop.core.eval.bootstrap import holm_adjust, paired_bootstrap_ci, paired_bootstrap_p
from adaloop.orchestration.tree import check_test_allowed

ITEM = ["benchmark", "item_id"]
CAP_METHODS = ("best_fixed", "oracle")  # defined per cost cap: no curve to interpolate
METHOD_ORDER = ("controller", "controller_no_h", "heuristic", "best_fixed", "early_exit_only", "tokens_only",
                "upfront_15way", "random", "oracle")


def report_split(ctrl: dict) -> str:
    """The split the numbers are reported on: the last of `splits.eval`."""
    return ctrl["splits"]["eval"][-1]


def split_label(ctrl: dict, split: str) -> str:
    """How a number of `split` must be read."""
    return f"{split} (tuned on {split}: optimistic)" if split == ctrl["splits"]["dev"] else split


def _checked(rows: pd.DataFrame) -> pd.DataFrame:
    check_test_allowed(rows["split"].unique())
    counts = rows.drop_duplicates(["split", *ITEM]).groupby(["split", "benchmark"]).size()
    if counts.groupby("split").nunique().gt(1).any():
        raise ValueError(f"the macro-average here needs equal item counts per benchmark:\n{counts}")
    return rows


def _points(rows: pd.DataFrame) -> tuple[pd.MultiIndex, np.ndarray, np.ndarray]:
    """One curve: (items, scores (P, N), cost fraction of each point (P,)) over its lambdas."""
    scores = rows.pivot(index="lam", columns=ITEM, values="correct")
    if scores.isna().any().any():
        raise ValueError("a curve point lacks some items")
    mean = rows.groupby("lam")[["flops_total", "cost_ref"]].mean()
    return scores.columns, scores.to_numpy(dtype=float), (mean["flops_total"] / mean["cost_ref"]).to_numpy()


def _curve_at(rows: pd.DataFrame, frac: float) -> pd.DataFrame | None:
    """Per seed (rows) and item (columns) the score interpolated at cost `frac`; None if any seed does not reach it."""
    out = {}
    for seed, part in rows.groupby("seed"):
        items, scores, costs = _points(part)
        weights = interpolation_weights(costs, frac)
        if weights is None:
            return None
        out[seed] = pd.Series(weights @ scores, index=items)
    return pd.DataFrame(out).T


def _best_within(select: pd.DataFrame, frac: float) -> tuple[float, float] | None:
    """(k0, lambda) with the best mean accuracy on `select` among the points that cost at most `frac`."""
    mean = select.groupby(["k0", "lam"])[["correct", "flops_total", "cost_ref"]].mean()
    fits = mean[mean["flops_total"] <= frac * mean["cost_ref"] * (1 + 1e-12)]
    if fits.empty:
        return None
    return fits.sort_values(["correct", "flops_total"], ascending=[False, True]).index[0]


def method_at(rows: pd.DataFrame, method: str, frac: float, ctrl: dict) -> tuple[dict, pd.Series | None]:
    """One method at one cost: (summary row, per-item scores averaged over seeds, or None).

    Curve methods: k0 is the one with the best interpolated accuracy on dev, then the report split's curve of that k0
    is read at `frac` (its costs use no labels). No extrapolation: a curve that does not reach `frac` gives NaN and a
    note; `acc_within_cap` then still shows the best dev-chosen point that costs no more than `frac`.
    best_fixed and oracle: their rows of this cap. All accuracies in %.
    """
    part = rows[rows["method"] == method]
    split, dev = report_split(ctrl), ctrl["splits"]["dev"]
    out = {"method": method, "cap_frac": frac, "k0": np.nan, "accuracy": np.nan, "seed_std": np.nan, "n_seeds": 0,
           "gqa": np.nan, "textvqa": np.nan, "cost_frac": np.nan, "acc_within_cap": np.nan, "note": ""}
    if part.empty:
        return {**out, "note": "no rows (method not run)"}, None
    if method in CAP_METHODS:
        at_cap = part[(part["split"] == split) & np.isclose(part["cap_frac"], frac)]
        if at_cap.empty:
            return {**out, "note": f"nothing fits {frac} x cost_ref"}, None
        scores = at_cap.set_index(ITEM)["correct"].sort_index().to_frame().T
        cost = at_cap["flops_total"].mean() / at_cap["cost_ref"].mean()
        out.update(k0=at_cap["k0"].iloc[0], cost_frac=cost, acc_within_cap=100.0 * at_cap["correct"].mean())
    else:
        select = part[part["split"] == dev]
        within = _best_within(select, frac)
        if within is not None:
            at = part[(part["split"] == split) & (part["k0"] == within[0]) & (part["lam"] == within[1])]
            out["acc_within_cap"] = 100.0 * at["correct"].mean()
        reached = {}
        for k0, group in select.groupby("k0"):
            curve = _curve_at(group, frac)
            if curve is not None:
                reached[k0] = curve.to_numpy().mean()
        if not reached:
            span = select.groupby(["k0", "lam", "seed"])[["flops_total", "cost_ref"]].mean()
            cost = span["flops_total"] / span["cost_ref"]
            return {**out, "note": f"no k0 curve reaches {frac} x cost_ref in every seed on {dev} "
                                   f"(points span {cost.min():.2f}-{cost.max():.2f})"}, None
        k0 = max(reached, key=reached.get)
        scores = _curve_at(part[(part["split"] == split) & (part["k0"] == k0)], frac)
        out["k0"] = k0
        if scores is None:
            return {**out, "note": f"the {split} curve at k0={k0} does not reach {frac} x cost_ref"}, None
        out["cost_frac"] = frac
    scores = scores.sort_index(axis=1)
    per_seed = 100.0 * scores.to_numpy().mean(axis=1)
    per_item = 100.0 * scores.mean(axis=0)
    out.update(accuracy=per_seed.mean(), n_seeds=len(per_seed),
               seed_std=per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan,
               **per_item.groupby(level="benchmark").mean().to_dict())
    return out, per_item


def headline_table(rows: pd.DataFrame, ctrl: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per method at `headline_frac`, per-item scores in % with one column per method that has a number)."""
    _checked(rows)
    table, items = [], {}
    for method in METHOD_ORDER:
        row, per_item = method_at(rows, method, ctrl["headline_frac"], ctrl)
        table.append({**row, "split": split_label(ctrl, report_split(ctrl))})
        if per_item is not None:
            items[method] = per_item
        if row["note"]:
            print(f"NOTE {method}: {row['note']}")
    return pd.DataFrame(table), pd.DataFrame(items)


def comparison_vs_controller(items: pd.DataFrame, ctrl: dict) -> pd.DataFrame:
    """Controller minus every other method at the headline cost: paired bootstrap over items (points).

    The Holm family is `claims` (config). A claim without a number is listed as not tested and still counts in the
    family size. Every other row is exploratory and has no adjusted p.
    """
    others = [m for m in METHOD_ORDER if m != "controller"]
    n_boot, seed = ctrl["bootstrap"]["n_boot"], ctrl["bootstrap"]["seed"]
    out = []
    for method in others:
        row = {"comparison": f"controller - {method}", "baseline": method, "claim": method in ctrl["claims"],
               "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "p_holm": np.nan, "n_items": 0,
               "split": split_label(ctrl, report_split(ctrl))}
        if "controller" in items and method in items:
            a, b = items["controller"].to_numpy(), items[method].to_numpy()
            if np.isnan(a).any() or np.isnan(b).any():
                raise ValueError(f"controller and {method} were not run on the same items")
            row["diff"], row["ci_low"], row["ci_high"] = paired_bootstrap_ci(a, b, n_boot, seed)
            row.update(p=paired_bootstrap_p(a, b, n_boot, seed), n_items=len(a))
        out.append(row)
    table = pd.DataFrame(out)
    claims = table["claim"]
    table.loc[claims, "p_holm"] = holm_adjust(table.loc[claims, "p"])
    return table


def by_cap_table(rows: pd.DataFrame, ctrl: dict) -> pd.DataFrame:
    """Every method at every cost cap (k0 chosen per cap on dev)."""
    _checked(rows)
    out = [method_at(rows, method, frac, ctrl)[0]
           for method in METHOD_ORDER if method in set(rows["method"]) for frac in ctrl["cap_fracs"]]
    return pd.DataFrame(out).assign(split=split_label(ctrl, report_split(ctrl)))


def iso_cost_points(rows: pd.DataFrame, headline: pd.DataFrame, ctrl: dict) -> pd.DataFrame:
    """F10 data on the report split: every curve point (mean over seeds) and every per-cap point.

    cost_frac = mean FLOPs / mean cost_ref, accuracy in %. `chosen` marks the k0 curve the headline reads (per-cap
    methods: every point).
    """
    part = _checked(rows)
    part = part[part["split"] == report_split(ctrl)].fillna({"lam": -1.0, "cap_frac": -1.0})
    keys = ["method", "k0", "lam", "cap_frac"]
    capped = part["method"].isin(CAP_METHODS)
    part = part.assign(k0=part["k0"].where(~capped, 0.0))  # one best_fixed line over caps, whatever its cell
    table = part.groupby(keys, as_index=False)[["correct", "flops_total", "cost_ref"]].mean()
    table = table.assign(cost_frac=table["flops_total"] / table["cost_ref"], accuracy=100.0 * table["correct"])
    chosen = dict(zip(headline["method"], headline["k0"]))
    fallback = table.groupby("method")["k0"].max().to_dict()  # a method without a headline number still gets drawn
    table["chosen"] = [m in CAP_METHODS or k0 == (fallback[m] if pd.isna(chosen.get(m, np.nan)) else chosen[m])
                       for m, k0 in zip(table["method"], table["k0"])]
    table[["lam", "cap_frac"]] = table[["lam", "cap_frac"]].replace(-1.0, np.nan)
    return table[keys + ["cost_frac", "accuracy", "chosen"]]


def run_headline(rows: pd.DataFrame, ctrl: dict, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Build every S14 policy table, write them under <out_dir>/tables and <out_dir>/figures_data, return them."""
    headline, items = headline_table(rows, ctrl)
    tables = {"T1_headline": headline, "T3_comparisons": comparison_vs_controller(items, ctrl),
              "T3_by_cap": by_cap_table(rows, ctrl)}
    data = {"F10_iso_cost": iso_cost_points(rows, headline, ctrl), "headline_item_scores": items.reset_index()}
    for folder, parts in (("tables", tables), ("figures_data", data)):
        (Path(out_dir) / folder).mkdir(parents=True, exist_ok=True)
        for name, table in parts.items():
            path = Path(out_dir) / folder / name
            if folder == "tables":
                table.to_csv(path.with_suffix(".csv"), index=False)
            else:
                table.to_parquet(path.with_suffix(".parquet"), index=False)
    print(f"S14 tables and figure data -> {out_dir} | {split_label(ctrl, report_split(ctrl))}")
    return {**tables, **data}
