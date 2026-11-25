"""Aggregate per-item result rows into one summary.

Adapted from Jiaqi-Gao/adaloop-vlm:adaloop/eval/runner.py::summarize_results @ e870bff.
"""
from collections.abc import Sequence

import numpy as np


def summarize_rows(rows: Sequence[dict]) -> dict:
    """Accuracy, parse rate, token counts, timings and peak memory over `rows` (one run config)."""
    if not rows:
        raise ValueError("no rows to summarize")

    def mean(key: str) -> float:
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "n": len(rows),
        "accuracy": 100.0 * mean("correct"),
        "parse_rate": 100.0 * mean("parse_ok"),
        "hit_limit_rate": 100.0 * mean("hit_limit"),
        "n_vis_mean": mean("n_vis"),
        "n_input_mean": mean("n_input"),
        "n_out_tokens_mean": mean("n_out_tokens"),
        "ttft_s_mean": mean("ttft_s"),
        "generate_s_mean": mean("generate_s"),
        # None for batched rows (not a per-item quantity)
        "peak_mem_gb_max": max((r["peak_mem_gb"] for r in rows if r.get("peak_mem_gb") is not None),
                               default=float("nan")),
    }
    return summary
