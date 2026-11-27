"""One left-padded batch of 8 with mixed prompt lengths against batch 1, on the real model, through the grid path.

  ADALOOP_SERVER=pace python scripts/check_padding.py

Asserts what the decoder receives (attention_mask, position_ids), that token counts and FLOPs are each item's own,
and that every item gets R loop passes. Prints batch-8 against batch-1 answers (they may differ: bf16 batch kernels).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.gpus import select_gpu_group  # noqa: E402
from utils.io import load_config  # noqa: E402

K = 0.25
N = 8


def main() -> None:
    """Run the check; any failed assert stops it."""
    cfg = load_config(ROOT / "configs/mvp.yaml")
    select_gpu_group(cfg)

    import torch

    from adaloop import api  # only now: this import initializes CUDA

    cfg = {**api.grid_config(cfg), "k_grid": [K], "batch_grouping": None}
    R_max = max(cfg["R_grid"])
    vlm = api.load_model(cfg)

    ### Step-1:- 8 dev items, half per benchmark, whose compressed prompts differ in length
    pool = api.load_gqa(cfg, "dev", limit=N, root=ROOT) + api.load_textvqa(cfg, "dev", limit=N, root=ROOT)
    length = {}
    for item in pool:
        inputs, _ = api.compress(vlm, api.item_inputs(vlm, item, cfg), item, K, cfg)
        length[item.item_id] = inputs["input_ids"].shape[1]
    items = pool[: N // 2] + pool[N : N + N // 2]
    lengths = [length[item.item_id] for item in items]
    width = max(lengths)
    print("prompt lengths:", lengths)
    assert len(set(lengths)) > 1, "these items have equal lengths: no padding would be tested"

    ### Step-2:- The batch, with what the decoder receives and the state after every pass
    seen, states_by_R = {}, {}

    def watch(module, args, kwargs):
        mask, positions = kwargs.get("attention_mask"), kwargs.get("position_ids")
        if kwargs["inputs_embeds"].shape[1] > 1:  # prompt forwards only, not the cached decode steps
            seen.setdefault("prefill", []).append((None if mask is None else mask.cpu(), positions.cpu()))

    handle = vlm.model.model.language_model.register_forward_pre_hook(watch, with_kwargs=True)
    try:
        batch_rows = api.run_items_cells(vlm, items, cfg)
    finally:
        handle.remove()
    for mask, positions in seen["prefill"]:
        assert mask is not None and mask.shape == (N, width), f"attention_mask {None if mask is None else mask.shape}"
        assert positions.shape == (N, width), f"position_ids {tuple(positions.shape)}"
        for i, T in enumerate(lengths):
            pad = width - T
            assert not mask[i, :pad].any() and mask[i, pad:].eq(1).all(), f"item {i}: mask is not 0 exactly on pads"
            assert torch.equal(positions[i, pad:], torch.arange(T)), f"item {i}: positions of real tokens"
            assert positions[i, :pad].eq(0).all(), f"item {i}: positions of pads"
    print(f"✓ {len(seen['prefill'])} prompt forwards: attention_mask 0 exactly on left pads, positions 0..T-1 per item")

    ### Step-3:- Each item alone: counts, FLOPs, passes and answers
    params = api.cost_params(vlm.model)
    same = total = 0
    for i, item in enumerate(items):
        rows, feats = batch_rows[i]
        alone_rows, alone_feats = api.run_item_cells(vlm, item, cfg)
        for row, alone in zip(rows, alone_rows):
            R = row["R"]
            assert row["batch_size"] == N, f"the batch was split: batch_size {row['batch_size']}"
            assert row["n_input"] == lengths[i], f"{item.item_id}: n_input {row['n_input']} != own {lengths[i]}"
            for key in ("n_input", "n_vis", "n_vis_full"):
                assert row[key] == alone[key], f"{item.item_id} R={R}: {key} {row[key]} != batch-1 {alone[key]}"
            cost, cost_alone = api.row_flops(params, row), api.row_flops(params, alone)
            keys = ["flops_vit", "flops_prefill", "flops_compression", "kv_cache_bytes"]
            if row["n_out_tokens"] == alone["n_out_tokens"]:  # a different answer may have another length
                keys += ["flops_decode", "flops_total"]
            for key in keys:
                assert cost[key] == cost_alone[key], f"{item.item_id} R={R}: {key} {cost[key]} != {cost_alone[key]}"
            batched, single = feats[K, R], alone_feats[K, R]
            for key in ("n_tokens", "n_vis", "flops_readout", "flops_state"):
                assert batched[key] == single[key], f"{item.item_id} R={R}: feature {key}"
            cosine = torch.cosine_similarity(batched["h_last"].float(), single["h_last"].float(), dim=0).item()
            equal = row["raw_output"] == alone["raw_output"]
            same, total = same + equal, total + 1
            print(f"{item.benchmark} {item.item_id} R={R} T={lengths[i]} pad={width - lengths[i]} | answer "
                  f"{'same' if equal else 'DIFFERENT'} {row['pred']!r} / {alone['pred']!r} | h_last cosine {cosine:.5f}")
            states_by_R.setdefault(R, []).append(cosine)
    print(f"✓ token counts and FLOPs of every item equal its batch-1 values | answers equal: {same}/{total}")
    print("lowest h_last cosine per R:", {R: round(min(values), 5) for R, values in states_by_R.items()})

    ### Step-4:- R reaches every item: R states per pass-hooked forward, and the state moves with each pass for each row
    inputs = [api.compress(vlm, api.item_inputs(vlm, item, cfg), item, K, cfg)[0] for item in items]
    api.set_loops(vlm, R_max)
    with api.record_pass_states(vlm, batch=True) as states:
        api.generate_batch(vlm, inputs, 1)
    assert len(states) == R_max and all(state.shape[0] == N for state in states), [tuple(s.shape) for s in states]
    for p in range(1, R_max):
        moved = (states[p] - states[p - 1]).norm(dim=-1)  # (B,)
        assert (moved > 0).all(), f"pass {p + 1} left an item's state unchanged: {moved.tolist()}"
    print(f"✓ R={R_max}: {R_max} core passes in the batched forward, every item's state changes at every pass")


if __name__ == "__main__":
    main()
