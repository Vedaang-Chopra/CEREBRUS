"""Public entry points. Notebooks and scripts import only this module."""
from adaloop.core.cost.flops import (
    REF_R,
    cost_params,
    cost_ref,
    flops_decode,
    flops_partial,
    flops_prefill,
    flops_readout,
    flops_total,
    flops_vit,
)
from adaloop.core.eval.extract import extract_answer
from adaloop.core.eval.scorers import score, score_gqa, score_gqa_lmms, score_lmms, score_textvqa
from adaloop.core.eval.summary import summarize_rows
from adaloop.core.model.budget import generate_answers
from adaloop.core.model.hooks import record_pass_states
from adaloop.core.model.inference import generate, generate_batch, last_logits
from adaloop.core.model.load import load_model, load_models, unload_model
from adaloop.core.model.loops import count_layer_calls, expected_layer_calls, set_loops
from adaloop.core.model.prompting import prepare_inputs
from adaloop.core.plots.accuracy_vs_r import plot_accuracy_vs_r
from adaloop.core.plots.colors import METHOD_COLORS, VARIANT_LINESTYLES
from adaloop.core.plots.cost_model import plot_cost_model
from adaloop.core.plots.pass_state import plot_pass_state
from adaloop.core.plots.save import save_figure, save_table
from adaloop.core.plots.style import set_paper_style
from adaloop.core.plots.thinking_off import plot_thinking_off
from adaloop.core.tokens.prune import extract_visual_tokens, prune_inputs
from adaloop.core.tokens.selectors import (
    n_keep_for,
    question_embeddings,
    random_selector,
    text_mean_selector,
    uniform_selector,
)
from adaloop.data.gqa import build_gqa_splits, load_gqa
from adaloop.data.splits import load_manifest, make_splits
from adaloop.data.textvqa import build_textvqa_splits, load_textvqa
from adaloop.orchestration.audit import wrong_predictions
from adaloop.orchestration.compress import METHODS, compress
from adaloop.orchestration.copies import compare_copies, time_copies
from adaloop.orchestration.cost_timing import time_prefill, time_throughput
from adaloop.orchestration.evaluate import (
    evaluate_item,
    read_answer,
    read_results,
    run_evaluation,
)
from adaloop.orchestration.full_run import (
    estimate_full_run,
    full_run_config,
    result_path,
    run_full,
    shard_items,
    summarize_full,
)
from adaloop.orchestration.inputs import item_inputs, item_prompt
from adaloop.orchestration.pass_state import check_pass_state
from adaloop.schemas.cost import CostParams
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.env import setup_notebook
from utils.gpus import select_gpu_group
from utils.hashing import config_hash, git_hash
from utils.images import load_image, resize_long_side
from utils.io import load_config

__all__ = [
    "REF_R",
    "CostParams",
    "GenerationResult",
    "Item",
    "LoadedVLM",
    "METHOD_COLORS",
    "METHODS",
    "VARIANT_LINESTYLES",
    "build_gqa_splits",
    "build_textvqa_splits",
    "check_pass_state",
    "compare_copies",
    "compress",
    "config_hash",
    "cost_params",
    "cost_ref",
    "count_layer_calls",
    "estimate_full_run",
    "evaluate_item",
    "expected_layer_calls",
    "extract_answer",
    "extract_visual_tokens",
    "flops_decode",
    "flops_partial",
    "flops_prefill",
    "flops_readout",
    "flops_total",
    "flops_vit",
    "full_run_config",
    "generate",
    "generate_answers",
    "generate_batch",
    "git_hash",
    "item_inputs",
    "item_prompt",
    "last_logits",
    "load_config",
    "load_gqa",
    "load_image",
    "load_manifest",
    "load_model",
    "load_models",
    "load_textvqa",
    "make_splits",
    "n_keep_for",
    "plot_accuracy_vs_r",
    "plot_cost_model",
    "plot_pass_state",
    "plot_thinking_off",
    "prepare_inputs",
    "prune_inputs",
    "question_embeddings",
    "random_selector",
    "read_answer",
    "read_results",
    "record_pass_states",
    "resize_long_side",
    "result_path",
    "run_evaluation",
    "run_full",
    "save_figure",
    "save_table",
    "score",
    "score_gqa",
    "score_gqa_lmms",
    "score_lmms",
    "score_textvqa",
    "select_gpu_group",
    "set_loops",
    "set_paper_style",
    "setup_notebook",
    "shard_items",
    "summarize_full",
    "summarize_rows",
    "text_mean_selector",
    "time_copies",
    "time_prefill",
    "time_throughput",
    "uniform_selector",
    "unload_model",
    "wrong_predictions",
]
