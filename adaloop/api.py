"""Public entry points. Notebooks and scripts import only this module."""
from adaloop.core.controller.baselines import hull_mix, upper_hull
from adaloop.core.controller.chooser import Chooser, chooser_loss
from adaloop.core.controller.curves import interpolation_weights
from adaloop.core.controller.features import pass_features
from adaloop.core.controller.heuristic import heuristic_actions
from adaloop.core.controller.inputs import fit_normalizer, node_scalars, policy_inputs, tree_data
from adaloop.core.controller.objectives import (
    backward_values,
    expected_reward_loss,
    leaf_rewards,
    supervised_loss,
    supervised_targets,
)
from adaloop.core.controller.policy import Policy
from adaloop.core.controller.rows import leaf_costs, policy_rows, select_on_dev
from adaloop.core.controller.synthetic import SYNTHETIC_PARAMS, synthetic_grid, synthetic_upfront
from adaloop.core.controller.tree import ACTIONS, action_mask, item_tree, path_name, walk
from adaloop.core.cost.flops import (
    REF_R,
    cost_params,
    cost_ref,
    flops_decode,
    flops_features,
    flops_partial,
    flops_prefill,
    flops_readout,
    flops_total,
    flops_vit,
    kv_cache_bytes,
    row_flops,
)
from adaloop.core.cost.meta import cost_params_from_config
from adaloop.core.eval.bootstrap import holm_adjust, paired_bootstrap_ci, paired_bootstrap_p
from adaloop.core.eval.extract import extract_answer
from adaloop.core.eval.pareto import pareto_front
from adaloop.core.eval.scorers import score, score_gqa, score_gqa_lmms, score_lmms, score_textvqa
from adaloop.core.eval.summary import summarize_rows
from adaloop.core.model.budget import generate_answers
from adaloop.core.model.hooks import record_attention_mass, record_first_logits, record_pass_states
from adaloop.core.model.inference import generate, generate_batch, last_logits
from adaloop.core.model.load import load_model, load_models, unload_model
from adaloop.core.model.loops import count_layer_calls, expected_layer_calls, set_loops
from adaloop.core.model.stage_hooks import count_hooks, record_stages
from adaloop.core.model.prompting import prepare_inputs
from adaloop.core.plots.accuracy_vs_r import plot_accuracy_vs_r
from adaloop.core.plots.colors import METHOD_COLORS, VARIANT_LINESTYLES
from adaloop.core.plots.compression import plot_pareto
from adaloop.core.plots.cost_model import plot_cost_model
from adaloop.core.plots.grid_surface import plot_cell_pareto, plot_kr_accuracy
from adaloop.core.plots.headroom import plot_headroom
from adaloop.core.plots.iso_cost import plot_iso_cost
from adaloop.core.plots.pass_state import plot_pass_state
from adaloop.core.plots.resolution import plot_resolution
from adaloop.core.plots.save import save_figure, save_table
from adaloop.core.plots.style import set_paper_style
from adaloop.core.plots.thinking_off import plot_thinking_off
from adaloop.core.plots.token_scan import plot_token_scan
from adaloop.core.tokens.prune import extract_visual_tokens, prune_inputs
from adaloop.core.tokens.selectors import (
    n_keep_for,
    question_embeddings,
    random_selector,
    text_mean_selector,
    uniform_selector,
)
from adaloop.core.train.lora import (
    attach_lora,
    expected_target_count,
    load_lora,
    lora_parameters,
    lora_target_names,
    remove_lora,
)
from adaloop.core.train.schedule import pairing_table, teacher_depth, variant_config, variant_name
from adaloop.core.train.teacher import answer_log_probs, split_top
from adaloop.core.train.teacher_store import load_records, read_index
from adaloop.data.gqa import build_gqa_splits, load_gqa
from adaloop.data.post.images import dhash, near_eval, phash
from adaloop.data.post.store import load_post_data
from adaloop.data.splits import load_manifest, make_splits
from adaloop.data.textvqa import build_textvqa_splits, load_textvqa
from adaloop.orchestration.analysis import batch_regime_table, cell_table, headroom_table, load_grid, noise_rate
from adaloop.orchestration.audit import wrong_predictions
from adaloop.orchestration.baselines import (
    early_exit_rows,
    no_h_rows,
    random_rows,
    save_baseline_rows,
    tokens_only_rows,
    upfront_15way_rows,
)
from adaloop.orchestration.compare import (
    comparison_cells,
    comparison_table,
    latency_table,
    pick_winner,
    run_comparison,
)
from adaloop.orchestration.batch_diagnostic import diagnose_batch
from adaloop.orchestration.batching import batchable_items, item_chunks
from adaloop.orchestration.cell_policies import (
    best_fixed_cell_rows,
    best_fixed_rows,
    cell_arrays,
    grid_cells,
    oracle_rows,
)
from adaloop.orchestration.cells import check_batched
from adaloop.orchestration.compress import METHODS, compress, encode_vision
from adaloop.orchestration.controller import greedy_exits, run_controller, run_heuristic, run_policies, train_policy
from adaloop.orchestration.copies import compare_copies, time_copies
from adaloop.orchestration.cost_timing import (
    cost_grid,
    estimate_grid_hours,
    time_grid,
    time_prefill,
    time_throughput,
)
from adaloop.orchestration.evaluate import (
    evaluate_item,
    read_answer,
    read_results,
    run_evaluation,
)
from adaloop.orchestration.features import load_features, run_item_cells, run_items_cells, save_features
from adaloop.orchestration.full_run import (
    estimate_full_run,
    full_run_config,
    result_path,
    run_full,
    shard_items,
    summarize_full,
)
from adaloop.orchestration.grid import (
    count_done,
    grid_config,
    grid_dir,
    grid_items,
    grid_remaining,
    merge_grid,
    rerun_items,
    run_grid,
)
from adaloop.orchestration.headline import (
    by_cap_table,
    comparison_vs_controller,
    headline_table,
    iso_cost_points,
    report_split,
    run_headline,
    split_label,
)
from adaloop.orchestration.headroom_gate import headroom_gate
from adaloop.orchestration.inputs import item_inputs, item_prompt
from adaloop.orchestration.lora_checks import (
    lora_eval_probe,
    lora_grad_norms,
    one_step_check,
    probe_loss,
    student_layer_calls,
)
from adaloop.orchestration.lora_data import load_targets, schedule_table, student_inputs, item_loss
from adaloop.orchestration.pareto import (
    pareto_cells,
    pareto_seconds_per_answer,
    pareto_status,
    pareto_table,
    run_pareto,
)
from adaloop.orchestration.pass_state import check_pass_state
from adaloop.orchestration.post_data import build_post_data, post_data_dir, post_data_eval_hashes
from adaloop.orchestration.profile import profile_cells, profile_item, profile_items, profile_table
from adaloop.orchestration.resolution import (
    resolution_cells,
    resolution_gain,
    resolution_memory,
    resolution_table,
    resolution_token_counts,
    run_resolution,
)
from adaloop.orchestration.run_check import check_run
from adaloop.orchestration.run_meta import record_launch
from adaloop.orchestration.synthetic_run import write_synthetic_grid
from adaloop.orchestration.teacher_cache import (
    build_teacher_cache,
    check_teacher_live,
    teacher_dir,
    teacher_items,
    teacher_records,
    teacher_run_config,
    teacher_summary,
    teacher_total_records,
)
from adaloop.orchestration.token_scan import run_token_scan, token_scan_table
from adaloop.orchestration.torch_trace import trace_item
from adaloop.orchestration.train_lora import lora_run_dir, lora_run_estimate, require_compressor, train_lora
from adaloop.orchestration.tree import (
    build_tree,
    check_test_allowed,
    controller_dir,
    load_controller_inputs,
    run_tree,
)
from adaloop.orchestration.upfront import load_upfront, run_upfront, upfront_dir, upfront_features
from adaloop.schemas.cost import CostParams
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.env import setup_notebook
from utils.gpus import select_gpu_group
from utils.hashing import config_hash, git_hash
from utils.images import load_image, resize_long_side
from utils.io import load_config
from utils.servers import is_gpu_host, server_name

__all__ = [
    "ACTIONS",
    "Chooser",
    "check_run",
    "chooser_loss",
    "interpolation_weights",
    "holm_adjust",
    "paired_bootstrap_p",
    "plot_iso_cost",
    "early_exit_rows",
    "no_h_rows",
    "random_rows",
    "record_launch",
    "save_baseline_rows",
    "tokens_only_rows",
    "upfront_15way_rows",
    "best_fixed_cell_rows",
    "best_fixed_rows",
    "hull_mix",
    "upper_hull",
    "cell_arrays",
    "grid_cells",
    "oracle_rows",
    "by_cap_table",
    "comparison_vs_controller",
    "headline_table",
    "iso_cost_points",
    "report_split",
    "run_headline",
    "split_label",
    "headroom_gate",
    "write_synthetic_grid",
    "load_upfront",
    "run_upfront",
    "upfront_dir",
    "upfront_features",
    "SYNTHETIC_PARAMS",
    "synthetic_grid",
    "synthetic_upfront",
    "Policy",
    "action_mask",
    "backward_values",
    "build_tree",
    "check_test_allowed",
    "controller_dir",
    "cost_params_from_config",
    "expected_reward_loss",
    "fit_normalizer",
    "greedy_exits",
    "heuristic_actions",
    "item_tree",
    "leaf_costs",
    "leaf_rewards",
    "load_controller_inputs",
    "node_scalars",
    "path_name",
    "policy_inputs",
    "policy_rows",
    "run_controller",
    "run_heuristic",
    "run_policies",
    "run_tree",
    "select_on_dev",
    "supervised_loss",
    "supervised_targets",
    "train_policy",
    "tree_data",
    "walk",
    "build_post_data",
    "dhash",
    "load_post_data",
    "near_eval",
    "phash",
    "post_data_dir",
    "post_data_eval_hashes",
    "count_hooks",
    "plot_token_scan",
    "profile_cells",
    "profile_item",
    "profile_items",
    "profile_table",
    "record_stages",
    "run_token_scan",
    "token_scan_table",
    "trace_item",
    "attach_lora",
    "expected_target_count",
    "item_loss",
    "load_lora",
    "load_targets",
    "lora_eval_probe",
    "lora_grad_norms",
    "lora_parameters",
    "lora_run_dir",
    "lora_run_estimate",
    "lora_target_names",
    "one_step_check",
    "pairing_table",
    "pareto_cells",
    "pareto_front",
    "pareto_seconds_per_answer",
    "pareto_status",
    "pareto_table",
    "probe_loss",
    "REF_R",
    "answer_log_probs",
    "build_teacher_cache",
    "check_teacher_live",
    "CostParams",
    "GenerationResult",
    "Item",
    "LoadedVLM",
    "METHOD_COLORS",
    "METHODS",
    "remove_lora",
    "require_compressor",
    "schedule_table",
    "student_inputs",
    "student_layer_calls",
    "teacher_depth",
    "train_lora",
    "variant_config",
    "VARIANT_LINESTYLES",
    "build_gqa_splits",
    "build_textvqa_splits",
    "check_batched",
    "diagnose_batch",
    "batchable_items",
    "item_chunks",
    "check_pass_state",
    "compare_copies",
    "comparison_cells",
    "comparison_table",
    "compress",
    "config_hash",
    "cost_grid",
    "cost_params",
    "encode_vision",
    "cost_ref",
    "cell_table",
    "count_layer_calls",
    "estimate_full_run",
    "estimate_grid_hours",
    "evaluate_item",
    "expected_layer_calls",
    "extract_answer",
    "extract_visual_tokens",
    "flops_decode",
    "flops_features",
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
    "batch_regime_table",
    "count_done",
    "grid_config",
    "grid_dir",
    "grid_items",
    "grid_remaining",
    "is_gpu_host",
    "server_name",
    "item_inputs",
    "item_prompt",
    "kv_cache_bytes",
    "last_logits",
    "latency_table",
    "load_config",
    "load_features",
    "load_gqa",
    "load_grid",
    "load_image",
    "load_manifest",
    "load_model",
    "load_models",
    "load_records",
    "load_textvqa",
    "make_splits",
    "headroom_table",
    "merge_grid",
    "noise_rate",
    "n_keep_for",
    "paired_bootstrap_ci",
    "pass_features",
    "pick_winner",
    "plot_accuracy_vs_r",
    "plot_cell_pareto",
    "plot_cost_model",
    "plot_headroom",
    "plot_kr_accuracy",
    "plot_pareto",
    "plot_pass_state",
    "plot_resolution",
    "plot_thinking_off",
    "prepare_inputs",
    "prune_inputs",
    "question_embeddings",
    "random_selector",
    "read_index",
    "read_answer",
    "read_results",
    "record_attention_mass",
    "record_first_logits",
    "record_pass_states",
    "rerun_items",
    "resize_long_side",
    "resolution_cells",
    "resolution_gain",
    "resolution_memory",
    "resolution_table",
    "resolution_token_counts",
    "row_flops",
    "result_path",
    "run_comparison",
    "run_evaluation",
    "run_full",
    "run_grid",
    "run_pareto",
    "run_item_cells",
    "run_resolution",
    "run_items_cells",
    "save_figure",
    "save_features",
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
    "split_top",
    "summarize_full",
    "summarize_rows",
    "teacher_dir",
    "teacher_items",
    "teacher_records",
    "teacher_run_config",
    "teacher_summary",
    "teacher_total_records",
    "text_mean_selector",
    "time_grid",
    "time_copies",
    "time_prefill",
    "time_throughput",
    "uniform_selector",
    "unload_model",
    "variant_name",
    "wrong_predictions",
]
