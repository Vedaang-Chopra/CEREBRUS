"""The multi-depth teacher cache (S8): the frozen full-token model at every R on every lora_train item.

Per item the vision tower runs once; its output is reused for every R, and each R is generated on its own. A batch
has one R (the loop count is model state). The log-probs come from the logits of the generation itself.
"""
import itertools
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from adaloop.core.eval.scorers import score
from adaloop.core.model.inference import generate_batch
from adaloop.core.model.loops import set_loops
from adaloop.core.tokens.prune import extract_visual_tokens, prune_inputs
from adaloop.core.train.teacher import answer_log_probs, target_tensors
from adaloop.core.train.teacher_store import load_records, read_index, repair_index, write_part
from adaloop.data.splits import load_manifest
from adaloop.orchestration.batching import item_chunks
from adaloop.orchestration.evaluate import read_answer
from adaloop.orchestration.full_run import LOADERS
from adaloop.orchestration.inputs import item_inputs
from adaloop.schemas.items import Item
from adaloop.schemas.vlm import LoadedVLM
from utils.hashing import config_hash, git_hash

TRAIN_SPLIT = "lora_train"
# everything that changes a teacher record; R_list, batch size and placement are not part of it
TEACHER_KEYS = ("model_id", "res_px", "prompts", "answer_format", "no_think_prefill", "reasoning_effort", "code_version")


def teacher_run_config(cfg: dict) -> dict:
    """The config of a teacher generation: the S2 setup with the no-thinking prefill and the teacher's token limit."""
    return {**cfg, "answer_prefill": cfg["no_think_prefill"], "max_new_tokens": cfg["teacher"]["max_new_tokens"]}


def teacher_hash(cfg: dict) -> str:
    """Hash of the settings that change teacher outputs."""
    key = {name: cfg[name] for name in TEACHER_KEYS}
    key["teacher"] = {name: cfg["teacher"][name] for name in ("top_n", "max_new_tokens")}
    return config_hash(key)


def teacher_dir(cfg: dict, root: str | Path) -> Path:
    """execution_results/teacher/<teacher hash>/."""
    return Path(root) / cfg["teacher"]["out_dir"] / teacher_hash(cfg)


def teacher_items(cfg: dict, root: str | Path, limit: int | None = None,
                  shard: tuple[int, int] = (0, 1), lazy: bool = False) -> list[Item]:
    """lora_train items of this shard (only their images are decoded); `limit` per benchmark, None = all.

    `lazy`: images are decoded on each read instead of being held in RAM (see `load_gqa`).
    """
    return [item for benchmark in cfg["teacher"]["benchmarks"]
            for item in LOADERS[benchmark](cfg, TRAIN_SPLIT, limit=limit, root=root, shard=shard, lazy=lazy)]


def teacher_total_records(cfg: dict, root: str | Path) -> int:
    """Records of the full cache, from the frozen split manifests."""
    sizes = [len(load_manifest(Path(root) / cfg["benchmarks"][b]["manifest"])["splits"][TRAIN_SPLIT])
             for b in cfg["teacher"]["benchmarks"]]
    return sum(sizes) * len(cfg["teacher"]["R_list"])


def _packed_inputs(vlm: LoadedVLM, item: Item, cfg: dict) -> dict:
    """Inputs with every visual token and the vision output already computed (reused across R)."""
    inputs = item_inputs(vlm, item, cfg)
    tokens, vision_outputs = extract_visual_tokens(vlm, inputs)
    return prune_inputs(vlm, inputs, vision_outputs, torch.arange(len(tokens)))


@torch.inference_mode()
def _teacher_batch(vlm: LoadedVLM, items: Sequence[Item], inputs: Sequence[dict], cfg: dict, R: int,
                   meta: dict) -> list[tuple[dict, dict]]:
    """(index line, tensors) of each item at loop count R, generated as one batch (`items` come from one chunk)."""
    set_loops(vlm, R)
    results, logits = generate_batch(vlm, inputs, cfg["max_new_tokens"], return_logits=True)
    records = []
    for item, gen, item_logits in zip(items, results, logits):
        pred = read_answer(gen, cfg)
        row = {
            "benchmark": item.benchmark, "item_id": item.item_id, "image_id": item.image_id,
            "question": item.question, "answers": list(item.answers), "R": R, "pred": pred,
            "raw_output": gen.raw_text, "n_answer_tokens": len(gen.token_ids), "parse_ok": pred is not None,
            "hit_limit": gen.hit_limit, "correct": score(item.benchmark, pred, item.answers), **meta,
        }
        records.append((row, target_tensors(gen.token_ids, item_logits, cfg["teacher"]["top_n"])))
    return records


def build_teacher_cache(vlms: Sequence[LoadedVLM], items: Sequence[Item], cfg: dict, root: str | Path,
                        shard: tuple[int, int] = (0, 1)) -> dict:
    """Cache every (item, R) of `items` that is not on disk yet; return {folder, new_records, seconds}.

    `items` are this shard's items (`teacher_items(..., shard=shard)`); `shard` only names the files. With several
    copies each takes every n-th item in its own thread. Resumable.
    """
    cfg = teacher_run_config(cfg)
    teacher, folder = cfg["teacher"], teacher_dir(cfg, root)
    R_list, batch_size = teacher["R_list"], teacher["batch_size"]
    done = {(row["benchmark"], row["item_id"], row["R"]) for row in read_index(folder)}
    pending = [item for item in items if any((item.benchmark, item.item_id, R) not in done for R in R_list)]
    print(f"teacher cache {folder} | shard {shard[0]}/{shard[1]}: {len(items)} items, {len(pending)} to compute, "
          f"R {R_list}, batch {batch_size}, {len(vlms)} copies")
    if not pending:
        return {"folder": folder, "new_records": 0, "seconds": 0.0}
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"config_hash": teacher_hash(cfg), "git_hash": git_hash(), "prompt_hash": config_hash(cfg["prompts"]),
            "model_revision": getattr(vlms[0].model.config, "_commit_hash", None),
            "code_version": cfg["code_version"], "seed": cfg["splits"]["seed"]}
    chunks = item_chunks(vlms[0], pending, cfg, batch_size)
    start = time.perf_counter()

    def work(j: int) -> int:
        vlm, mine = vlms[j], chunks[j :: len(vlms)]
        total, seen = sum(len(chunk) for chunk in mine), 0
        writer = f"shard{shard[0]}of{shard[1]}_copy{j}"
        repair_index(folder / f"index_{writer}.jsonl")
        written = 0
        buffer: dict[int, list] = {R: [] for R in R_list}

        def flush() -> None:
            nonlocal written
            for R, records in buffer.items():
                if records:
                    write_part(folder, writer, R, [row for row, _ in records], [t for _, t in records])
                    written += len(records)
                    records.clear()

        for n, chunk in enumerate(mine):
            inputs = [_packed_inputs(vlm, item, cfg) for item in chunk]  # the vision tower, once per item
            for R in R_list:
                todo = [i for i, item in enumerate(chunk) if (item.benchmark, item.item_id, R) not in done]
                if todo:
                    buffer[R] += _teacher_batch(vlm, [chunk[i] for i in todo], [inputs[i] for i in todo], cfg, R, meta)
            before, seen = seen, seen + len(chunk)
            if seen // teacher["flush_every"] > before // teacher["flush_every"] or n == len(mine) - 1:
                flush()
                print(f"copy {j}: {seen}/{total} items | {seen / (time.perf_counter() - start):.2f} items/s")
        flush()
        return written

    with ThreadPoolExecutor(max_workers=len(vlms)) as pool:
        new_records = sum(pool.map(work, range(len(vlms))))
    return {"folder": folder, "new_records": new_records, "seconds": time.perf_counter() - start}


def teacher_records(cfg: dict, root: str | Path, items: Sequence[Item] | None = None) -> tuple[list[dict], list[dict]]:
    """(index lines, tensors) of the cached records, of `items` only if given; ordered by item and R."""
    folder = teacher_dir(cfg, root)
    rows = read_index(folder)
    if items is not None:
        wanted = {(item.benchmark, item.item_id) for item in items}
        rows = [row for row in rows if (row["benchmark"], row["item_id"]) in wanted]
    rows.sort(key=lambda row: (row["benchmark"], row["item_id"], row["R"]))
    return rows, load_records(folder, rows)


@torch.inference_mode()
def check_teacher_live(vlm: LoadedVLM, items: Sequence[Item], cfg: dict, root: str | Path) -> list[dict]:
    """Compare every cached record of `items` with a fresh teacher-forced forward on prompt + cached answer ids.

    Per record: the largest gap in probability over the cached top tokens, the largest log-prob gap over the top 5,
    and the share of answer tokens that are the teacher-forced argmax. The forward runs the full model (pixel input,
    batch 1, no KV cache); the cache came from batched cached decoding.
    """
    cfg = teacher_run_config(cfg)
    by_key = {(item.benchmark, item.item_id): item for item in items}
    rows, tensors = teacher_records(cfg, root, items)
    out = []
    for row, record in zip(rows, tensors):
        set_loops(vlm, row["R"])
        inputs = item_inputs(vlm, by_key[row["benchmark"], row["item_id"]], cfg)
        answer_ids = record["answer_ids"].long()
        logp = answer_log_probs(vlm, inputs, answer_ids.to(inputs["input_ids"].device)).cpu()
        live = logp.gather(-1, record["top_ids"].long())  # (n, top_n)
        out.append({
            "benchmark": row["benchmark"], "item_id": row["item_id"], "R": row["R"], "n": len(answer_ids),
            "max_prob_diff": float((live.exp() - record["top_logp"].exp()).abs().max()),
            "max_logp_diff_top5": float((live[:, :5] - record["top_logp"][:, :5]).abs().max()),
            "argmax_match": float((logp.argmax(dim=-1) == answer_ids).float().mean()),
        })
    return out


def teacher_summary(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """(per benchmark x R: n, accuracy, parse and limit rates; per benchmark x R pair: share of equal answers)."""
    groups: dict[tuple, list] = {}
    preds: dict[tuple, dict] = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["R"]), []).append(row)
        preds.setdefault((row["benchmark"], row["item_id"]), {})[row["R"]] = row["pred"]

    def percent(group: list, key: str) -> float:
        return 100.0 * sum(float(r[key]) for r in group) / len(group)

    per_R = [{"benchmark": b, "R": R, "n": len(g), "accuracy": percent(g, "correct"),
              "parse_rate": percent(g, "parse_ok"), "hit_limit_rate": percent(g, "hit_limit"),
              "answer_tokens": sum(r["n_answer_tokens"] for r in g) / len(g)}
             for (b, R), g in sorted(groups.items())]
    pairs = []
    for benchmark in sorted({b for b, _ in groups}):
        for a, b in itertools.combinations(sorted({R for bench, R in groups if bench == benchmark}), 2):
            both = [p for (bench, _), p in preds.items() if bench == benchmark and a in p and b in p]
            if both:
                pairs.append({"benchmark": benchmark, "R_a": a, "R_b": b, "n": len(both),
                              "agreement": 100.0 * sum(p[a] == p[b] for p in both) / len(both)})
    return per_R, pairs
