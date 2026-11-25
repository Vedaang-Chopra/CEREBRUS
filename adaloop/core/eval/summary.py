"""Aggregate per-item result rows into one summary.

Adapted from Jiaqi-Gao/adaloop-vlm:adaloop/eval/runner.py::summarize_results @ e870bff.
"""
from collections.abc import Sequence

import numpy as np


def summarize_rows(rows: Sequence[dict]) -> dict:
    """Both accuracies, parse/format/length/forced rates, token counts, timings and peak memory over `rows` (one run config).

    A field that no row has (older result files) comes out as nan.
    """
    if not rows:
        raise ValueError("no rows to summarize")

    def mean(key: str) -> float:
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "n": len(rows),
        "accuracy": 100.0 * mean("correct"),
        "accuracy_lmms": 100.0 * mean("correct_lmms"),
        "parse_rate": 100.0 * mean("parse_ok"),
        "format_ok_rate": 100.0 * mean("format_ok"),
        "answer_len_ok_rate": 100.0 * mean("answer_len_ok"),
        "hit_limit_rate": 100.0 * mean("hit_limit"),
        "forced_rate": 100.0 * mean("forced"),
        "n_vis_mean": mean("n_vis"),
        "n_input_mean": mean("n_input"),
        "n_out_tokens_mean": mean("n_out_tokens"),
        "n_think_tokens_mean": mean("n_think_tokens"),
        "n_answer_tokens_mean": mean("n_answer_tokens"),
        "ttft_s_mean": mean("ttft_s"),
        "generate_s_mean": mean("generate_s"),
        # None for batched rows (not a per-item quantity)
        "peak_mem_gb_max": max((r["peak_mem_gb"] for r in rows if r.get("peak_mem_gb") is not None),
                               default=float("nan")),
    }
    return summary
