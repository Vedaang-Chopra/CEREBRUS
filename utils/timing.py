"""GPU timing and memory helpers.

FirstTokenTimer ported from Jiaqi-Gao/adaloop-vlm:adaloop/eval/timing.py @ e870bff.
"""
import logging
import time
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


def cuda_sync() -> None:
    """Wait for every visible GPU, so wall-clock timings are complete."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def reset_peak_memory() -> None:
    """Reset the peak-allocation counter on every visible GPU."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def peak_memory_gb() -> float:
    """Sum over GPUs of peak allocated memory since the last reset, in GiB."""
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3


class FirstTokenTimer:
    """Streamer for `model.generate` that records time to the first generated token.

    `generate` first calls `put` with the prompt, then once per new token.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self.start_time: float | None = None
        self.first_token_time: float | None = None
        self.prompt_received = False

    def start(self) -> None:
        """Start the clock; call right before `generate`."""
        self.start_time = self.clock()
        self.first_token_time = None
        self.prompt_received = False

    def put(self, value: torch.Tensor) -> None:
        """Streamer hook: the first call is the prompt, the second is the first new token."""
        if not self.prompt_received:
            self.prompt_received = True
            return
        if self.first_token_time is None:
            self.first_token_time = self.clock()

    def end(self) -> None:
        """Streamer hook; nothing to do."""

    @property
    def ttft_s(self) -> float | None:
        """Seconds from `start` to the first generated token, or None if no token came."""
        if self.start_time is None or self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time
