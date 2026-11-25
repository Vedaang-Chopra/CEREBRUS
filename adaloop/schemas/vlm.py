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
    """Newly generated tokens (on CPU) and their decoded text."""

    token_ids: torch.Tensor
    text: str
