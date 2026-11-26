"""torch.profiler trace of one item per keep fraction: Chrome trace files, GPU busy vs idle, top kernels."""
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from adaloop.orchestration.cells import generate_cell
from adaloop.orchestration.compress import compress
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.timing import cuda_sync

GPU_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    """Length in ms of the union of (start, end) intervals in microseconds."""
    total, reached = 0.0, float("-inf")
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, reached))
        reached = max(reached, end)
    return total / 1000.0


def trace_item(vlm: LoadedVLM, item: Item, cfg: dict, method: str, ks: Sequence[float], R: int,
               out_dir: str | Path, top: int = 15) -> tuple[list[dict], list[dict]]:
    """Profile compress + generate of one item at each k; return (one summary line per k, `top` kernels per k).

    Each k runs once untimed first. `busy_ms` is the time any GPU ran a kernel or copy, `idle_ms` the rest of the
    wall time (which includes the profiler's own overhead, so compare the two k, not these with unprofiled runs).
    Without a GPU the kernel lines are the top operators by self CPU time instead (`kind` says which).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    summaries, kernels = [], []
    for k in ks:
        cell_cfg = {**cfg, "compressor": method, "k": k}
        inputs = item_inputs(vlm, item, cell_cfg)

        def run():
            packed, _ = compress(vlm, inputs, item, k, cell_cfg)
            return generate_cell(vlm, [packed], cell_cfg, R)[0], packed["input_ids"].shape[1]

        run()  # warmup
        with profile(activities=activities) as profiler:
            cuda_sync(vlm.devices)
            start = time.perf_counter()
            gen, n_input = run()
            cuda_sync(vlm.devices)
            wall_ms = 1000.0 * (time.perf_counter() - start)
        path = out_dir / f"trace_{item.item_id}_{method}_k{round(100 * k)}_R{R}.json"
        profiler.export_chrome_trace(str(path))

        events = [e for e in json.loads(path.read_text())["traceEvents"]
                  if e.get("ph") == "X" and e.get("cat") in GPU_CATEGORIES]
        devices = sorted({e["args"]["device"] for e in events})
        busy = _union_ms([(e["ts"], e["ts"] + e["dur"]) for e in events])
        per_device = {f"busy_gpu{d}_ms": _union_ms([(e["ts"], e["ts"] + e["dur"]) for e in events
                                                    if e["args"]["device"] == d]) for d in devices}
        summaries.append({"method": method, "k": k, "R": R, "n_input": n_input, "n_new": len(gen.token_ids),
                          "wall_ms": wall_ms, "busy_ms": busy if events else float("nan"),
                          "idle_ms": wall_ms - busy if events else float("nan"), **per_device, "trace": str(path)})
        if events:
            by_name: dict[str, list[float]] = {}
            for event in events:
                by_name.setdefault(event["name"], []).append(event["dur"] / 1000.0)
            found = [("gpu_kernel", name, len(v), sum(v)) for name, v in by_name.items()]
        else:
            found = [("cpu_op", op.key, op.count, op.self_cpu_time_total / 1000.0) for op in profiler.key_averages()]
        total = sum(ms for *_, ms in found)
        kernels += [{"k": k, "kind": kind, "name": name[:100], "calls": calls, "total_ms": ms, "pct": 100.0 * ms / total}
                    for kind, name, calls, ms in sorted(found, key=lambda f: -f[3])[:top]]
        print(f"k={k}: wall {wall_ms:.1f} ms | GPU busy {summaries[-1]['busy_ms']:.1f} ms | trace {path}")
    return summaries, kernels
