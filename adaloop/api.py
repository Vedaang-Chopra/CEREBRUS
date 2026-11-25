"""Public entry points. Notebooks and scripts import only this module."""
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
from adaloop.schemas.vlm import GenerationResult, LoadedVLM
from utils.images import load_image
from utils.io import load_config

__all__ = [
    "GenerationResult",
    "LoadedVLM",
    "count_layer_calls",
    "expected_layer_calls",
    "extract_visual_tokens",
    "generate",
    "last_logits",
    "load_config",
    "load_image",
    "load_model",
    "prepare_inputs",
    "prune_inputs",
    "question_embeddings",
    "random_selector",
    "set_loops",
    "text_mean_selector",
    "uniform_selector",
]
