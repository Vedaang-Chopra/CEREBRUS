"""Controller capacity: train every architecture variant, pick one on ctrl_val, report all of them."""
import json
from pathlib import Path

import pandas as pd

from adaloop.orchestration.controller import run_controller
from adaloop.orchestration.diagnostics import collapse_table
from adaloop.orchestration.headline import method_at

ARCH_FILE = "arch.json"


def _report(name: str, rows: pd.DataFrame, log: list[dict], tree: pd.DataFrame, features: dict, ctrl: dict) -> dict:
    """One ablation row: ctrl_val reward (what chooses), then dev numbers (which choose nothing here)."""
    val = pd.DataFrame(log)
    at_cap, _ = method_at(rows, name, ctrl["headline_frac"], ctrl)
    collapsed = collapse_table(rows, tree, features, ctrl, method=name)["collapsed"].sum()
    return {"variant": name, "val_reward": val["val_reward"].mean(), "best_epoch": val["best_epoch"].mean(),
            "dev_acc_at_headline": at_cap["accuracy"], "seed_std": at_cap["seed_std"],
            "collapsed_settings": int(collapsed), "settings": len(val.drop_duplicates(["k0", "lam"])),
            "note": at_cap["note"]}


def run_capacity(tree: pd.DataFrame, features: dict, ctrl: dict,
                 out_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(controller rows of the chosen variant, one report row per variant, the chosen variant's settings).

    `ctrl["capacity"]` lists variants: a name plus the settings it overrides. Each is trained at every k0 x lambda x
    seed; the one with the best mean ctrl_val reward over all of them wins (ties: first listed). The winner is then
    also trained with the other `node_weight_options` (expected_reward only), as "<name>_<option>", and the best mean
    ctrl_val reward of all rows is the controller. With
    `out_dir`: weights/<variant>/, capacity_eval.parquet (rows of every variant), tables/T4_capacity.csv, arch.json.
    """
    parts, report, settings = {}, [], {}

    def train(name: str, overrides: dict) -> None:
        settings[name] = overrides
        run = {**ctrl, **overrides}
        weights = None if out_dir is None else Path(out_dir) / "weights" / name
        log: list[dict] = []
        parts[name] = run_controller(tree, features, run, weights_dir=weights, method=name, log=log)
        report.append(_report(name, parts[name], log, tree, features, run))

    ### Step-1:- Architectures
    for variant in ctrl["capacity"]:
        train(variant["name"], {key: value for key, value in variant.items() if key != "name"})
    ### Step-2:- The other node weights, on the best architecture only
    if ctrl["objective"] == "expected_reward":
        first = max(report, key=lambda row: row["val_reward"])["variant"]
        for option in ctrl["node_weight_options"]:
            if option != ctrl["node_weight"]:
                train(f"{first}_{option}", {**settings[first], "node_weight": option})
    table = pd.DataFrame(report)
    best = table.loc[table["val_reward"].idxmax(), "variant"]
    table.insert(1, "chosen", table["variant"] == best)
    print(f"capacity: {best!r} has the best mean ctrl_val reward of {len(table)} variants")
    if out_dir is not None:
        out = Path(out_dir)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        pd.concat(parts.values(), ignore_index=True).to_parquet(out / "capacity_eval.parquet", index=False)
        table.to_csv(out / "tables" / "T4_capacity.csv", index=False)
        (out / ARCH_FILE).write_text(json.dumps({"variant": best, "settings": settings[best]}, indent=1) + "\n")
    return parts[best].assign(method="controller"), table, settings[best]


def chosen_arch(out_dir: str | Path) -> dict:
    """The settings of the variant `run_capacity` chose: merge them into `ctrl` for runs that must match it."""
    return json.loads((Path(out_dir) / ARCH_FILE).read_text())["settings"]
