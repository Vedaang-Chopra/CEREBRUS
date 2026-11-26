"""Per-stage times of the model's own forward, from forward hooks. Nothing is hooked outside `record_stages`."""
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from adaloop.schemas.vlm import LoadedVLM
from utils.timing import DeviceClock


@dataclass
class _Call:
    """One call of one hooked module."""

    stage: str  # vit | projector | prelude | core<p> | coda
    forward: int | None  # index of the model forward it ran in; None outside (vision tower run by itself)
    phase: str | None  # prefill | decode | None
    device: torch.device
    start: object = None
    end: object = None
    arrived: object = None  # after a device change: when the inputs were on this device


class StageTrace:
    """What the hooks saw; read it with `totals` after the context has closed."""

    def __init__(self, clock: DeviceClock) -> None:
        self.clock = clock
        self.calls: list[_Call] = []
        self.phases: list[str] = []  # one entry per model forward

    def totals(self) -> dict:
        """Milliseconds per stage.

        Consecutive calls of one stage on one device form a segment, from the first start to the last end on that
        device (so the work between those modules is inside); durations never span two devices. `prefill` and
        `decode` map stage -> ms over the forwards of that phase (`outside`: the vision tower run by itself).
        `handoff_ms`: input moves at a device change, already inside the stage that received them.
        """
        segments: list[list[_Call]] = []
        for call in self.calls:
            last = segments[-1][-1] if segments else None
            if last and (last.forward, last.stage, last.device) == (call.forward, call.stage, call.device):
                segments[-1].append(call)
            else:
                segments.append([call])
        out = {"prefill": {}, "decode": {}, "outside": {}}
        for segment in segments:
            first = segment[0]
            stages = out[first.phase or "outside"]
            stages[first.stage] = stages.get(first.stage, 0.0) + self.clock.ms(first.start, segment[-1].end)
        handoff = {"prefill": 0.0, "decode": 0.0, "outside": 0.0}
        for call in self.calls:
            if call.arrived is not None:
                handoff[call.phase or "outside"] += self.clock.ms(call.start, call.arrived)
        return {**out, "handoff_ms": handoff, "n_prefill": self.phases.count("prefill"),
                "n_decode": self.phases.count("decode")}


def _device_of(module: nn.Module, args: tuple, kwargs: dict) -> torch.device:
    """Where the module computes: its parameters' device, else its first tensor input's."""
    param = next(module.parameters(), None)
    if param is not None:
        return param.device
    return next(value.device for value in (*args, *kwargs.values()) if torch.is_tensor(value))


def _stage_modules(vlm: LoadedVLM) -> list[tuple[nn.Module, str]]:
    """(module, stage group) of every hooked module; none of them contains another."""
    inner = vlm.model.model
    tower, language = inner.vision_tower, inner.language_model
    start, end = language.config.loop_layer_index_start, language.config.loop_layer_index_end
    modules = [(tower.embeddings, "vit"), (tower.pre_layernorm, "vit"), *((layer, "vit") for layer in tower.layers),
               (tower.merger, "projector"), (language.embed_tokens, "embed"), (language.message_input, "message")]
    for i, layer in enumerate(language.layers):
        modules.append((layer, "prelude" if i < start else "coda" if i >= end else "core"))
    return modules + [(language.final_input, "coda"), (language.norm, "coda"), (vlm.model.lm_head, "coda")]


def count_hooks(vlm: LoadedVLM) -> int:
    """Forward hooks and pre-hooks on the model, to check that profiling left none behind."""
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in vlm.model.modules())


@contextmanager
def record_stages(vlm: LoadedVLM, clock: DeviceClock | None = None) -> Iterator[StageTrace]:
    """Hook the model's modules for the calls made inside the block; yield the `StageTrace` that fills up.

    A model forward with more than one position is `prefill`, with one position `decode`. At a device change the
    device that just worked is synchronized first, so a segment never starts while it waits for the other GPU
    (this makes the hooked run slower than a plain one: take wall times from a plain run). One model copy, one thread.
    """
    clock = clock or DeviceClock()
    trace = StageTrace(clock)
    state = {"forward": None, "phase": None, "open": None, "device": None, "passes": {}, "messages": 0}
    n_core = vlm.model.model.language_model.config.loop_layer_index_end \
        - vlm.model.model.language_model.config.loop_layer_index_start

    def forward_start(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        embeds = kwargs.get("inputs_embeds")
        length = ids.shape[1] if ids is not None else embeds.shape[1]
        state.update(forward=len(trace.phases), phase="prefill" if length > 1 else "decode", passes={}, messages=0)
        trace.phases.append(state["phase"])

    def forward_end(module, args, output):
        state.update(forward=None, phase=None)

    def stage_name(module, group: str) -> str | None:
        if group == "embed":  # outside a forward it is compression reading word embeddings, timed there
            return "prelude" if state["phase"] else None
        if group == "message":  # called before every layer of pass 2 onward
            state["messages"] += 1
            return f"core{(state['messages'] - 1) // n_core + 2}"
        if group == "core":
            state["passes"][module] = state["passes"].get(module, 0) + 1
            return f"core{state['passes'][module]}"
        return group

    def make_pre(group: str):
        def pre(module, args, kwargs):
            stage = stage_name(module, group)
            if stage is None:
                return
            device = _device_of(module, args, kwargs)
            moved = state["device"] is not None and device != state["device"]
            if moved:
                clock.sync(state["device"])
            call = _Call(stage, state["forward"], state["phase"], device, start=clock.mark(device))
            state.update(open=(module, call, moved), device=device)
            trace.calls.append(call)

        return pre

    def post(module, args, output):
        if state["open"] and state["open"][0] is module:
            call = state["open"][1]
            call.end = clock.mark(call.device)
            state["open"] = None

    def child_pre(module, args):
        # the first child to run after a device change: accelerate has moved the parent's inputs by now
        if state["open"] and state["open"][2] and state["open"][1].arrived is None:
            state["open"][1].arrived = clock.mark(state["open"][1].device)

    handles = [vlm.model.register_forward_pre_hook(forward_start, with_kwargs=True),
               vlm.model.register_forward_hook(forward_end)]
    for module, group in _stage_modules(vlm):
        handles.append(module.register_forward_pre_hook(make_pre(group), with_kwargs=True))
        handles.append(module.register_forward_hook(post))
        handles += [child.register_forward_pre_hook(child_pre) for child in module.children()]
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()
        for device in {call.device for call in trace.calls}:  # events must have finished before they are read
            clock.sync(device)
