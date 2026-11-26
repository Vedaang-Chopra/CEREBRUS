"""One run of the vision tower on one item, shared by every compression method and k that starts from it."""
from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass(frozen=True)
class VisionRun:
    """Projected tokens (N, d), the tower's full output, the last layer's attention inputs and the wall time."""

    tokens: torch.Tensor
    outputs: ModelOutput
    attention_inputs: dict
    seconds: float
