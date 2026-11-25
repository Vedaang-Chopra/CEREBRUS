"""Public entry points. Notebooks and scripts import only this module."""
from adaloop.core.eval.extract import extract_answer
from adaloop.core.eval.scorers import normalize_answer, score, score_gqa
from adaloop.core.eval.summary import summarize_rows
from adaloop.core.model.inference import generate, last_logits
from adaloop.core.model.load import load_model
from adaloop.core.model.loops import count_layer_calls, expected_layer_calls, set_loops
from adaloop.core.model.prompting import prepare_inputs
from adaloop.core.tokens.prune import extract_visual_tokens, prune_inputs
from adaloop.core.tokens.selectors import (
    question_embeddings,
    random_selector,
    text_mean_selector,
    uniform_selector,
)
from adaloop.data.gqa import build_gqa_splits, load_gqa
from adaloop.data.splits import load_manifest, make_splits
from adaloop.orchestration.evaluate import evaluate_item, read_results, run_evaluation
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.hashing import config_hash, git_hash
from utils.images import load_image, resize_long_side
from utils.io import load_config

__all__ = [
    "GenerationResult",
    "Item",
    "LoadedVLM",
    "build_gqa_splits",
    "config_hash",
    "count_layer_calls",
    "evaluate_item",
    "expected_layer_calls",
    "extract_answer",
    "extract_visual_tokens",
    "generate",
    "git_hash",
    "last_logits",
    "load_config",
    "load_gqa",
    "load_image",
    "load_manifest",
    "load_model",
    "make_splits",
    "normalize_answer",
    "prepare_inputs",
    "prune_inputs",
    "question_embeddings",
    "random_selector",
    "read_results",
    "resize_long_side",
    "run_evaluation",
    "score",
    "score_gqa",
    "set_loops",
    "summarize_rows",
    "text_mean_selector",
    "uniform_selector",
]
