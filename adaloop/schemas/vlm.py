"""Model handle and generation result."""
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, ProcessorMixin


@dataclass(frozen=True)
class LoadedVLM:
    """A loaded model with its processor."""

    model: PreTrainedModel
    processor: ProcessorMixin
    loading_info: dict


@dataclass(frozen=True)
class GenerationResult:
    """Newly generated tokens (on CPU), decoded text, and cost measurements of the call.

    `raw_text` keeps special tokens (needed for answer extraction); `text` drops them.
    Times are wall-clock seconds with GPUs synchronized; `peak_mem_gb` is summed over GPUs.
    """

    token_ids: torch.Tensor
    text: str
    raw_text: str
    n_input_tokens: int
    n_visual_tokens: int
    hit_limit: bool
    ttft_s: float | None
    generate_s: float
    peak_mem_gb: float
