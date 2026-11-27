"""Thin CLI over adaloop.api. Run from the repo root on the GPU host.

  python scripts/mvp.py smoke
  python scripts/mvp.py splits
  python scripts/mvp.py eval --benchmark gqa --split dev --loops 1 3 5 --limit 20
  python scripts/mvp.py full --shard 0/4        # configs/full_baseline.yaml; shard i of n on full_gpu_groups[i] (GPUs 0-7)
  python scripts/mvp.py grid --shard 0/2        # S7: configs/mvp.yaml; shard i of n on gpu_groups[i]
  python scripts/mvp.py grid --limit 20         # nb08's sample: 20 dev items into grid_sample_b8, with an ETA
  python scripts/mvp.py grid --per-split 32     # quick pass into the real grid folder; the full run resumes on it
  python scripts/mvp.py grid-check              # nb08 Step-2b: batch 1 against batch 8 on dev items
  ADALOOP_SERVER=pace python scripts/mvp.py grid   # placement overlay configs/servers/pace.yaml (one H200)
  python scripts/mvp.py pareto --limit 32       # nb07 step 7 (S5b); without --limit the full sweep, resuming
  python scripts/mvp.py upfront                 # S13: upfront features of the grid items (no decoder pass), one copy
  python scripts/mvp.py teacher --shard 0/4     # S8: configs/train_lora.yaml (GPUs 0-7), one process per shard
  python scripts/mvp.py train --variant A       # S9: full LoRA run, configs/train_lora.yaml (a free pair of 0-7)
  python scripts/mvp.py train --variant B --smoke   # S9 smoke run (pairs of smoke.gpu_groups); A and B may run at once

`smoke` and `eval` replace Jiaqi-Gao/adaloop-vlm:scripts/{smoke_test,run_grid}.py @ e870bff.
`smoke`, `eval`, `upfront`, `grid-check` and `pareto` pick a free GPU group; `splits` needs no GPU.
The group is selected before `adaloop.api` is imported (that import initializes CUDA).
`train` claims its GPU pair with a lock, so two runs started together get different pairs.
`full`, `grid`, `grid-check`, `pareto`, `upfront`, `teacher` and `train` are resumable: rerun the same command after an interruption.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.gpus import claim_gpu_group, select_gpu_group  # noqa: E402
from utils.io import load_config  # noqa: E402

BENCHMARKS = ["gqa", "textvqa"]


def cmd_smoke(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Load the model and answer one dev item at R=3."""
    vlm = api.load_model(cfg)
    item = api.load_gqa(cfg, "dev", limit=1, root=ROOT)[0]
    row = api.evaluate_item(vlm, item, cfg, R=3)
    print(json.dumps({k: row[k] for k in ("question_type", "pred", "answers", "correct", "n_vis", "generate_s")}))


def cmd_splits(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Create (or check; --grow: extend to the configured sizes) the split manifests and print their sizes."""
    for benchmark, build in (("gqa", api.build_gqa_splits), ("textvqa", api.build_textvqa_splits)):
        manifest = build(cfg, ROOT, grow=args.grow)
        print(benchmark, {name: len(ids) for name, ids in manifest["splits"].items()}, "hash", manifest["hash"])


def cmd_eval(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Evaluate one benchmark split at each requested R (full tokens), resumable."""
    if args.split == "test":
        raise SystemExit("the test split is frozen until stage S12")
    load_items = {"gqa": api.load_gqa, "textvqa": api.load_textvqa}[args.benchmark]
    items = load_items(cfg, args.split, limit=args.limit, root=ROOT)
    vlm = api.load_model(cfg)
    out_dir = ROOT / cfg["out_dir"] / "eval"
    for R in args.loops:
        out = out_dir / f"{args.benchmark}_{args.split}_n{len(items)}_res{cfg['res_px']}_R{R}_{api.config_hash(cfg)}.jsonl"
        summary = api.run_evaluation(vlm, items, cfg, R, out, resume=True)
        print(json.dumps(summary))


def parse_shard(text: str, groups: list[list[int]]) -> tuple[int, int]:
    """Parse `i/n` (i is 0-based); refuse shards that do not fit the command's GPU groups."""
    try:
        i_text, n_text = text.split("/")
        i, n = int(i_text), int(n_text)
    except ValueError:
        raise SystemExit(f"--shard must look like 0/2, got {text!r}")
    n_groups = len(groups)
    if not 1 <= n <= n_groups:
        raise SystemExit(f"--shard {text}: n must be between 1 and {n_groups} (number of GPU groups)")
    if not 0 <= i < n:
        raise SystemExit(f"--shard {text}: i must be between 0 and {n - 1}")
    return i, n


def cmd_full(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Run shard i of n of the full baseline on GPU group i, resumable."""
    vlm = api.load_model(cfg)
    api.run_full(vlm, cfg, ROOT, shard=args.shard)


def sample_items(api: ModuleType, cfg: dict, limit: int) -> list:
    """The first `limit` dev items, half GQA (the larger half) and half TextVQA, as in nb08's sample."""
    picked = []
    for load, n in ((api.load_gqa, limit - limit // 2), (api.load_textvqa, limit // 2)):
        if n:
            picked += load(cfg, "dev", limit=n, root=ROOT)
    return picked


def quick_items(api: ModuleType, vlm, items: list, cfg: dict, per_split: int) -> list:
    """The first `per_split` items of each benchmark x split, widened to the whole batches of the full run that hold
    them, so the full run later skips those batches and every item is run once."""
    count, first = {}, set()
    for item in items:
        group = (item.benchmark, item.split)
        count[group] = count.get(group, 0) + 1
        if count[group] <= per_split:
            first.add((item.benchmark, item.item_id))
    chunks = api.item_chunks(vlm, items, cfg, cfg["grid_batch"]["size"])
    wide = [item for chunk in chunks if any((i.benchmark, i.item_id) in first for i in chunk) for item in chunk]
    print(f"quick pass: first {per_split} items of {len(count)} benchmark x split groups = {len(first)} items, "
          f"{len(wide)} with their whole batches")
    return wide


def cmd_grid(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Run shard i of n of the grid and of its noise re-run, then merge; resumable.

    --limit writes only the sample items, into the sample folder. --per-split writes the first items of each
    benchmark x split into the real grid folder; the full run continues on top of them. grid.parquet is written only
    when every expected row is there, grid_partial.parquet always (except --limit, which mirrors nb08).
    """
    if any(n is not None and n < 1 for n in (args.limit, args.per_split)):
        raise SystemExit("--limit and --per-split must be at least 1")
    if args.limit is not None and args.per_split is not None:
        raise SystemExit("use --limit or --per-split, not both")
    cfg = api.grid_config(cfg)
    items = api.grid_items(cfg, ROOT)
    full_again = api.rerun_items(items, cfg["n_rerun"], cfg["splits"]["seed"])
    full_dir = api.grid_dir(cfg, ROOT)  # grid batch size in the name, the same folder as nb08
    vlm = api.load_model(cfg)
    only, again, out_dir = items, full_again, full_dir
    if args.limit is not None:
        only = sample_items(api, cfg, args.limit)
        again = api.rerun_items(only, 2, cfg["splits"]["seed"])
        out_dir = api.grid_dir(cfg, ROOT, "grid_sample")
    elif args.per_split is not None:
        only = quick_items(api, vlm, items, cfg, args.per_split)
        keys = {(item.benchmark, item.item_id) for item in only}
        again = [item for item in full_again if (item.benchmark, item.item_id) in keys]  # part of the full re-run
    n_cells = len(cfg["k_grid"]) * len(cfg["R_grid"])
    before = api.count_done(out_dir, cfg)

    ### Step-1:- Main run and noise re-run
    start = time.perf_counter()
    main_n, main_s = api.run_grid([vlm], items, cfg, out_dir, shard=args.shard, only=only)
    again_n, again_s = 0, 0.0
    if again:
        again_n, again_s = api.run_grid([vlm], items, cfg, out_dir, shard=args.shard, rerun_id=1, only=again)
    else:
        print("noise re-run skipped: none of these items is in the full re-run set")
    wall = time.perf_counter() - start
    generated, seconds = main_n + again_n, main_s + again_s
    per_answer = seconds / generated if generated else None
    print(f"this launch: {generated} answers generated, {api.count_done(out_dir, cfg) - before} rows written | "
          f"wall {wall:.1f} s, batches {seconds:.1f} s"
          + (f" | {per_answer:.3f} s per answer" if generated else ""))

    ### Step-2:- Merge and count (shard runs: the counts are complete once every shard has finished)
    if not any(out_dir.glob(f"grid_*_{api.config_hash(cfg)}.jsonl")):
        print(f"no result files yet in {out_dir}")
        return
    params = api.cost_params(vlm.model)
    name = "grid.parquet" if args.limit is not None else "grid_partial.parquet"
    table = api.merge_grid(out_dir, cfg, name=name, params=params)
    expected = {0: len(only) * n_cells, 1: len(again) * n_cells}
    got = table.rerun_id.value_counts().to_dict()
    for rerun_id, n in expected.items():
        print(f"rerun_id={rerun_id}: {got.get(rerun_id, 0)} rows (this run's items: {n})")
    if args.limit is None and args.per_split is None and all(got.get(r, 0) == n for r, n in expected.items()):
        name = "grid.parquet"
        api.merge_grid(out_dir, cfg, name=name, params=params)
    print(f"merged {len(table)} rows -> {out_dir / name}")
    if args.per_split is not None:
        main = table[table.rerun_id == 0]
        accuracy = main.groupby(["benchmark", "split", "k", "R"]).correct.agg(["size", "mean"]).reset_index()
        accuracy["mean"] = (100 * accuracy["mean"]).round(1)
        print(accuracy.rename(columns={"size": "n", "mean": "accuracy"}).to_string(index=False))

    ### Step-3:- How long the rest of the full run takes at this speed (counts unwritten batch neighbours too)
    left = (api.grid_remaining(vlm, items, cfg, full_dir)
            + api.grid_remaining(vlm, items, cfg, full_dir, rerun_id=1, only=full_again))
    eta = "unknown (nothing was generated in this launch)" if per_answer is None else \
        f"{left * per_answer / 3600:.2f} h on one model copy"
    print(f"full run: {left} answers left to generate | ETA {eta}")


def cmd_grid_check(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Batch 1 against the grid batch on dev items (nb08 Step-2b, same folders and csv); resumable."""
    cfg = api.grid_config(cfg)
    check, size = cfg["grid_batch"]["check"], cfg["grid_batch"]["size"]
    check_cfg = {**cfg, "k_grid": check["ks"], "R_grid": [check["R"]]}
    items = api.grid_items(cfg, ROOT)
    check_items = [item for load in (api.load_gqa, api.load_textvqa)
                   for item in load(cfg, "dev", limit=check["n_items"], root=ROOT)]
    vlm = api.load_model(cfg)
    out_dir = api.grid_dir(cfg, ROOT)
    check_dir = out_dir / f"b1_vs_b{size}"
    regimes, generated, seconds = {}, 0, 0.0
    start = time.perf_counter()
    for batch in (1, size):
        print(f"[STAGE] {len(check_items)} dev items at batch {batch}")
        n, s = api.run_grid([vlm], items, check_cfg, check_dir / f"b{batch}", batch_size=batch, only=check_items)
        generated, seconds = generated + n, seconds + s
        regimes[batch] = api.merge_grid(check_dir / f"b{batch}", check_cfg)
    print(f"this launch: {generated} answers generated | wall {time.perf_counter() - start:.1f} s, "
          f"batches {seconds:.1f} s" + (f" | {seconds / generated:.3f} s per answer" if generated else ""))
    for batch, table in regimes.items():
        expected = len(check_items) * len(check["ks"])
        print(f"batch {batch}: {len(table)} rows of {expected} expected")
        if len(table) != expected:
            raise SystemExit(f"batch {batch} is incomplete: rerun the same command")
    regime_table = api.batch_regime_table(regimes[1], regimes[size], check["n_boot"], cfg["splits"]["seed"])
    csv = out_dir / f"b1_vs_b{size}.csv"
    regime_table.to_csv(csv, index=False)
    print(regime_table.round(2).to_string(index=False))
    print(f"saved {csv}")


def cmd_pareto(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """The S5b Pareto sweep of nb07 step 7 (--limit: first N dev items per benchmark), resumable.

    Answers are capped at `pareto.max_new_tokens`, so the rows have other config hashes than nb07's and sit next to
    them in the same folders. Accuracy rows are generated in batches of `eval_batch_size`; the latency pass is batch 1.
    """
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    import pandas as pd

    cfg = {**cfg, "max_new_tokens": cfg["pareto"]["max_new_tokens"]}
    n_full = cfg["pareto"]["n_items"]
    full_items = {"gqa": api.load_gqa(cfg, "dev", limit=n_full, root=ROOT),
                  "textvqa": api.load_textvqa(cfg, "dev", limit=n_full, root=ROOT)}
    n = args.limit or n_full
    items = {benchmark: bench_items[:n] for benchmark, bench_items in full_items.items()}
    base = ROOT / cfg["out_dir"]
    out_dir, latency_dir = base / "nb07" / "pareto", base / "nb07" / "pareto_latency"
    reuse, reuse_latency = base / cfg["pareto"]["reuse_dir"], base / cfg["pareto"]["reuse_latency_dir"]

    def count(run_items: dict, folder: Path, done_dir: Path) -> tuple[int, int]:
        """(rows of the sweep, rows still missing)."""
        status = api.pareto_status(run_items, cfg, folder, done_dir)
        return sum(row["n_items"] for row in status), sum(row["missing"] for row in status)

    total, left = count(items, out_dir, reuse)
    batch = cfg["eval_batch_size"]
    print(f"accuracy rows: total {total} | done {total - left} | left {left} | batch {batch} | {out_dir}")
    latency_items = {benchmark: bench_items[: cfg["latency_items"]] for benchmark, bench_items in items.items()}
    if not args.skip_latency:
        lat_total, lat_left = count(latency_items, latency_dir, reuse_latency)
        print(f"latency rows: total {lat_total} | done {lat_total - lat_left} | left {lat_left} | batch 1")
    vlm = api.load_model(cfg)

    ### Step-1:- Accuracy rows
    start = time.perf_counter()
    rows = api.run_pareto([vlm], items, cfg, out_dir, reuse, batch_size=batch)
    wall = time.perf_counter() - start
    per_answer = wall / left if left else None
    full_left = count(full_items, out_dir, reuse)[1]
    if per_answer is None:
        print(f"nothing to generate | full sweep ({n_full} per benchmark): {full_left} answers left")
    else:
        print(f"{left} answers in {wall:.0f} s: {per_answer:.3f} s per answer | full sweep ({n_full} per benchmark): "
              f"{full_left} answers left, ETA {full_left * per_answer / 3600:.2f} h (latency pass not included)")

    ### Step-2:- Latency pass (batch 1), then the table
    latency_rows = [] if args.skip_latency else api.run_pareto([vlm], latency_items, cfg, latency_dir, reuse_latency)
    table = api.pareto_table(rows, latency_rows, cfg)
    csv = base / "nb07" / f"pareto_cli_n{n}.csv"
    pd.DataFrame(table).to_csv(csv, index=False)
    for line in table:
        if line["benchmark"] == "all":
            print(f"{line['cell']:18s} accuracy {line['accuracy']:5.1f} | kept {line['n_vis']:6.1f} | "
                  f"{line['gflops_total']:8.2f} GFLOPs | parsed {line['parse_rate']:.0f}%")
    print(f"saved {csv}")


def cmd_upfront(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Save the upfront features of the grid items (--limit: only the first ones) next to the grid, resumable."""
    cfg = api.grid_config(cfg)  # the features carry the grid's config hash
    items = api.grid_items(cfg, ROOT)[: args.limit]
    folder = api.upfront_dir(cfg, ROOT)
    written = api.run_upfront(api.load_model(cfg), items, cfg, folder)
    print(f"{written} new of {len(items)} items | {folder}")


def cmd_teacher(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Build shard i of n of the teacher cache on GPU group i, resumable."""
    vlm = api.load_model(cfg)
    items = api.teacher_items(cfg, ROOT, limit=args.limit, shard=args.shard)
    done = api.build_teacher_cache([vlm], items, cfg, ROOT, shard=args.shard)
    print(f"{done['new_records']} new records in {done['seconds']:.0f} s | {done['folder']}")


def cmd_train(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Train the LoRA of one variant on lora_train against the S8 cache (full, or the smoke settings); resumable."""
    api.require_compressor(cfg)  # before the data and the model are loaded
    vcfg = api.variant_config(cfg, args.variant)
    smoke = cfg["smoke"]
    name, limit, steps, eff_batch = "full", None, None, None  # None: cfg["train"]
    if args.smoke:
        name, limit, steps, eff_batch = "smoke", smoke["n_per_benchmark"], smoke["steps"], smoke["eff_batch"]
    items = api.teacher_items(vcfg, ROOT, limit=limit, lazy=True)
    targets = api.load_targets(items, vcfg, ROOT)
    vlm = api.load_model(vcfg)
    peft_model = api.attach_lora(vlm, vcfg)
    run_dir = api.lora_run_dir(vcfg, ROOT, name)
    history = api.train_lora(vlm, peft_model, items, targets, vcfg, run_dir, steps=steps, eff_batch=eff_batch)
    print(f"{len(history)} steps done | {run_dir}")
    if args.smoke:
        print(api.lora_run_estimate(history, vcfg, eff_batch))


def select_gpus(cfg: dict, args: argparse.Namespace) -> None:
    """Pick the GPU group for the command; for `full`, also turn args.shard into (i, n)."""
    if args.command in ("full", "grid", "teacher"):
        # `full` has its own, larger list: notebooks stay on `gpu_groups`
        groups = cfg["full_gpu_groups"] if args.command == "full" else cfg["gpu_groups"]
        args.shard = parse_shard(args.shard, groups)
        i, n = args.shard
        print(f"shard {i}/{n} | GPU group {i}")
        select_gpu_group({**cfg, "gpu_groups": groups}, group=i)
    elif args.command in ("smoke", "eval", "upfront", "grid-check", "pareto"):
        select_gpu_group(cfg)
    elif args.command == "train":
        if args.variant not in cfg["lora_variants"]:
            raise SystemExit(f"train needs --variant, one of {list(cfg['lora_variants'])}; got {args.variant!r}")
        # smoke.gpu_groups is a hashed block, so a server overlay cannot change it: with one set, use its gpu_groups
        server = os.environ.get("ADALOOP_SERVER")
        groups = cfg["smoke"]["gpu_groups"] if args.smoke and server in (None, "", "inara") else cfg["gpu_groups"]
        claim_gpu_group({**cfg, "gpu_groups": groups}, ROOT / cfg["train"]["out_dir"] / "gpu_locks")


def main() -> None:
    """Parse arguments and run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "splits", "eval", "full", "grid", "grid-check", "pareto", "upfront",
                                            "teacher", "train"])
    parser.add_argument("--config", default=None, help="default: per command (full, teacher, train have their own)")
    parser.add_argument("--benchmark", default="gqa", choices=BENCHMARKS)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--loops", type=int, nargs="+", default=[3])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grow", action="store_true", help="splits: extend the manifests to splits.sizes")
    parser.add_argument("--per-split", type=int, default=None, help="grid: quick pass, first N items per benchmark x split")
    parser.add_argument("--skip-latency", action="store_true", help="pareto: no batch-1 latency pass")
    parser.add_argument("--shard", default="0/1", help="full, grid, teacher: i/n, 0-based i, one GPU group per shard")
    parser.add_argument("--variant", help="train: a key of lora_variants (A, B)")
    parser.add_argument("--smoke", action="store_true", help="train: the cfg['smoke'] items, steps and eff_batch")
    args = parser.parse_args()
    if args.config is None:
        defaults = {"full": "configs/full_baseline.yaml", "teacher": "configs/train_lora.yaml",
                    "train": "configs/train_lora.yaml"}
        args.config = defaults.get(args.command, "configs/mvp.yaml")
    cfg = load_config(ROOT / args.config)
    select_gpus(cfg, args)

    from adaloop import api  # only now: this import initializes CUDA

    {"smoke": cmd_smoke, "splits": cmd_splits, "eval": cmd_eval, "full": cmd_full, "grid": cmd_grid,
     "grid-check": cmd_grid_check, "pareto": cmd_pareto, "upfront": cmd_upfront, "teacher": cmd_teacher, "train": cmd_train}[args.command](api, cfg, args)


if __name__ == "__main__":
    main()
