"""Thin CLI over adaloop.api. Run from the repo root on the GPU host.

  python scripts/mvp.py smoke
  python scripts/mvp.py splits
  python scripts/mvp.py eval --benchmark gqa --split dev --loops 1 3 5 --limit 20
  python scripts/mvp.py full --shard 0/2        # configs/full_baseline.yaml; shard i of n on gpu_groups[i]
  python scripts/mvp.py grid --shard 0/2        # S7: configs/mvp.yaml; shard i of n on gpu_groups[i]
  python scripts/mvp.py teacher --shard 0/4     # S8: configs/train_lora.yaml (GPUs 0-7), one process per shard
  python scripts/mvp.py train --variant A       # S9: full LoRA run, configs/train_lora.yaml (a free pair of 0-7)
  python scripts/mvp.py train --variant B --smoke   # S9 smoke run (pairs of smoke.gpu_groups); A and B may run at once

`smoke` and `eval` replace Jiaqi-Gao/adaloop-vlm:scripts/{smoke_test,run_grid}.py @ e870bff.
`smoke` and `eval` pick a free GPU group; `splits` needs no GPU.
The group is selected before `adaloop.api` is imported (that import initializes CUDA).
`train` claims its GPU pair with a lock, so two runs started together get different pairs.
`full`, `grid`, `teacher` and `train` are resumable: rerun the same command after an interruption.
"""
import argparse
import json
import sys
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
    """Create (or check) the split manifests and print their sizes."""
    for benchmark, build in (("gqa", api.build_gqa_splits), ("textvqa", api.build_textvqa_splits)):
        manifest = build(cfg, ROOT)
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


def parse_shard(text: str, cfg: dict) -> tuple[int, int]:
    """Parse `i/n` (i is 0-based); refuse shards that do not fit `gpu_groups`."""
    try:
        i_text, n_text = text.split("/")
        i, n = int(i_text), int(n_text)
    except ValueError:
        raise SystemExit(f"--shard must look like 0/2, got {text!r}")
    n_groups = len(cfg["gpu_groups"])
    if not 1 <= n <= n_groups:
        raise SystemExit(f"--shard {text}: n must be between 1 and {n_groups} (len of gpu_groups)")
    if not 0 <= i < n:
        raise SystemExit(f"--shard {text}: i must be between 0 and {n - 1}")
    return i, n


def cmd_full(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Run shard i of n of the full baseline on GPU group i, resumable."""
    vlm = api.load_model(cfg)
    api.run_full(vlm, cfg, ROOT, shard=args.shard)


def cmd_grid(api: ModuleType, cfg: dict, args: argparse.Namespace) -> None:
    """Run shard i of n of the grid and of its noise re-run on GPU group i, resumable."""
    items = api.grid_items(cfg, ROOT)
    again = api.rerun_items(items, cfg["n_rerun"], cfg["splits"]["seed"])
    vlms = [api.load_model(cfg)]
    out_dir = ROOT / cfg["out_dir"] / "grid"
    api.run_grid(vlms, items, cfg, out_dir, shard=args.shard)
    api.run_grid(vlms, again, cfg, out_dir, shard=args.shard, rerun_id=1)


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
        args.shard = parse_shard(args.shard, cfg)
        i, n = args.shard
        print(f"shard {i}/{n} | GPU group {i}")
        select_gpu_group(cfg, group=i)
    elif args.command in ("smoke", "eval"):
        select_gpu_group(cfg)
    elif args.command == "train":
        if args.variant not in cfg["lora_variants"]:
            raise SystemExit(f"train needs --variant, one of {list(cfg['lora_variants'])}; got {args.variant!r}")
        groups = cfg["smoke"]["gpu_groups"] if args.smoke else cfg["gpu_groups"]
        claim_gpu_group({**cfg, "gpu_groups": groups}, ROOT / cfg["train"]["out_dir"] / "gpu_locks")


def main() -> None:
    """Parse arguments and run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "splits", "eval", "full", "grid", "teacher", "train"])
    parser.add_argument("--config", default=None, help="default: per command (full, teacher, train have their own)")
    parser.add_argument("--benchmark", default="gqa", choices=BENCHMARKS)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--loops", type=int, nargs="+", default=[3])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default="0/1", help="full, grid, teacher: i/n, 0-based i, one gpu_groups entry per shard")
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
     "teacher": cmd_teacher, "train": cmd_train}[args.command](api, cfg, args)


if __name__ == "__main__":
    main()
