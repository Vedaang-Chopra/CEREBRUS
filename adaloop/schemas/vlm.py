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

    @property
    def devices(self) -> list[int] | None:
        """Visible GPU indices this model sits on; None without a device map (then: treat as all GPUs)."""
        device_map = getattr(self.model, "hf_device_map", None)
        if not device_map:
            return None
        names = {str(dev) for dev in device_map.values()} - {"cpu", "disk", "meta"}
        return sorted(int(name.removeprefix("cuda:")) for name in names)


@dataclass(frozen=True)
class GenerationResult:
    """Newly generated tokens (on CPU), decoded text, and cost measurements of the call.

    `raw_text` keeps special tokens (needed for answer extraction); `text` drops them.
    Times are wall-clock seconds with the model's GPUs synchronized; `peak_mem_gb` is summed over those GPUs.
    With several copies running in threads, times include contention between the copies (Python GIL).
    From `generate_batch`, `generate_s` is batch time / batch size and `ttft_s`, `peak_mem_gb` are None.
    """

    token_ids: torch.Tensor
    text: str
    raw_text: str
    n_input_tokens: int
    n_visual_tokens: int
    hit_limit: bool
    ttft_s: float | None
    generate_s: float
    peak_mem_gb: float | None
