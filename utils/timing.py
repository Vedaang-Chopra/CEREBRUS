"""GPU timing and memory helpers.

FirstTokenTimer ported from Jiaqi-Gao/adaloop-vlm:adaloop/eval/timing.py @ e870bff.
"""
import time
from collections.abc import Callable, Sequence

import torch


def _devices(devices: Sequence[int] | None) -> Sequence[int]:
    """The given GPU indices, or every visible GPU."""
    return range(torch.cuda.device_count()) if devices is None else devices


def cuda_sync(devices: Sequence[int] | None = None) -> None:
    """Wait for these GPUs (default: all visible), so wall-clock timings are complete."""
    for i in _devices(devices):
        torch.cuda.synchronize(i)


def reset_peak_memory(devices: Sequence[int] | None = None) -> None:
    """Reset the peak-allocation counter on these GPUs (default: all visible)."""
    for i in _devices(devices):
        torch.cuda.reset_peak_memory_stats(i)


def peak_memory_gb(devices: Sequence[int] | None = None) -> float:
    """Sum over these GPUs (default: all visible) of peak allocated memory since the last reset, in GiB."""
    return sum(torch.cuda.max_memory_allocated(i) for i in _devices(devices)) / 1024**3


class DeviceClock:
    """Time marks per device: CUDA events on a GPU, the wall clock on CPU.

    A duration is only defined between two marks of the same device.
    """

    def mark(self, device: torch.device) -> tuple[torch.device, object]:
        """A mark in the device's queue (GPU) or the current time (CPU)."""
        if device.type != "cuda":
            return device, time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.default_stream(device))
        return device, event

    def sync(self, device: torch.device) -> None:
        """Wait until the device has finished everything queued on it."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def ms(self, start: tuple, end: tuple) -> float:
        """Milliseconds between two marks of one device; both must have been reached (`sync` first)."""
        if start[0] != end[0]:
            raise ValueError(f"marks of two devices cannot be compared: {start[0]} and {end[0]}")
        if start[0].type != "cuda":
            return 1000.0 * (end[1] - start[1])
        return start[1].elapsed_time(end[1])


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
