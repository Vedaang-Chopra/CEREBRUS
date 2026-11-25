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
from adaloop.core.eval.scorers import score, score_gqa, score_textvqa
from adaloop.core.eval.summary import summarize_rows
from adaloop.core.model.hooks import record_pass_states
from adaloop.core.model.inference import generate, generate_batch, last_logits
from adaloop.core.model.load import load_model, unload_model
from adaloop.core.model.loops import count_layer_calls, expected_layer_calls, set_loops
from adaloop.core.model.prompting import prepare_inputs
from adaloop.core.plots.colors import METHOD_COLORS
from adaloop.core.plots.save import save_figure, save_table
from adaloop.core.plots.style import set_paper_style
from adaloop.core.tokens.prune import extract_visual_tokens, prune_inputs
from adaloop.core.tokens.selectors import (
    question_embeddings,
    random_selector,
    text_mean_selector,
    uniform_selector,
)
from adaloop.data.gqa import build_gqa_splits, load_gqa
from adaloop.data.splits import load_manifest, make_splits
from adaloop.data.textvqa import build_textvqa_splits, load_textvqa
from adaloop.orchestration.audit import wrong_predictions
from adaloop.orchestration.cost_timing import time_prefill, time_throughput
from adaloop.orchestration.evaluate import evaluate_item, item_inputs, item_prompt, read_results, run_evaluation
from adaloop.orchestration.full_run import full_run_config, result_path, run_full, shard_items, summarize_full
from adaloop.orchestration.pass_state import check_pass_state
from adaloop.schemas.cost import CostParams
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.env import setup_notebook
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
    "build_gqa_splits",
    "build_textvqa_splits",
    "check_pass_state",
    "config_hash",
    "cost_params",
    "cost_ref",
    "count_layer_calls",
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
    "load_textvqa",
    "make_splits",
    "prepare_inputs",
    "prune_inputs",
    "question_embeddings",
    "random_selector",
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
    "score_textvqa",
    "set_loops",
    "set_paper_style",
    "setup_notebook",
    "shard_items",
    "summarize_full",
    "summarize_rows",
    "text_mean_selector",
    "time_prefill",
    "time_throughput",
    "uniform_selector",
    "unload_model",
    "wrong_predictions",
]
