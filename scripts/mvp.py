"""Thin CLI over adaloop.api. Run from the repo root on the GPU host.

  python scripts/mvp.py smoke
  python scripts/mvp.py splits
  python scripts/mvp.py eval --benchmark gqa --split dev --loops 1 3 5 --limit 20
  python scripts/mvp.py full --shard 0/2        # configs/full_baseline.yaml; shard i of n on gpu_groups[i]

`smoke` and `eval` replace Jiaqi-Gao/adaloop-vlm:scripts/{smoke_test,run_grid}.py @ e870bff.
`full` is resumable: rerun the same command after an interruption.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaloop import api  # noqa: E402

LOADERS = {"gqa": api.load_gqa, "textvqa": api.load_textvqa}
SPLIT_BUILDERS = {"gqa": api.build_gqa_splits, "textvqa": api.build_textvqa_splits}


def cmd_smoke(cfg: dict, args: argparse.Namespace) -> None:
    """Load the model and answer one dev item at R=3."""
    vlm = api.load_model(cfg)
    item = api.load_gqa(cfg, "dev", limit=1, root=ROOT)[0]
    row = api.evaluate_item(vlm, item, cfg, R=3)
    print(json.dumps({k: row[k] for k in ("question_type", "pred", "answers", "correct", "n_vis", "generate_s")}))


def cmd_splits(cfg: dict, args: argparse.Namespace) -> None:
    """Create (or check) the split manifests and print their sizes."""
    for benchmark, build in SPLIT_BUILDERS.items():
        manifest = build(cfg, ROOT)
        print(benchmark, {name: len(ids) for name, ids in manifest["splits"].items()}, "hash", manifest["hash"])


def cmd_eval(cfg: dict, args: argparse.Namespace) -> None:
    """Evaluate one benchmark split at each requested R (full tokens), resumable."""
    if args.split == "test":
        raise SystemExit("the test split is frozen until stage S12")
    items = LOADERS[args.benchmark](cfg, args.split, limit=args.limit, root=ROOT)
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


def cmd_full(cfg: dict, args: argparse.Namespace) -> None:
    """Run shard i of n of the full baseline on GPU group i, resumable."""
    i, n = parse_shard(args.shard, cfg)
    # set before anything initializes CUDA: this process sees only group i, in PCI bus order
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in cfg["gpu_groups"][i])
    print(f"shard {i}/{n} | physical GPUs {cfg['gpu_groups'][i]} | CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    vlm = api.load_model(cfg)
    api.run_full(vlm, cfg, ROOT, shard=(i, n))


def main() -> None:
    """Parse arguments and run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "splits", "eval", "full"])
    parser.add_argument("--config", default=None, help="default: configs/full_baseline.yaml for full, else configs/mvp.yaml")
    parser.add_argument("--benchmark", default="gqa", choices=sorted(LOADERS))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--loops", type=int, nargs="+", default=[3])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default="0/1", help="full only: i/n, 0-based i, one gpu_groups entry per shard")
    args = parser.parse_args()
    if args.config is None:
        args.config = "configs/full_baseline.yaml" if args.command == "full" else "configs/mvp.yaml"
    cfg = api.load_config(ROOT / args.config)
    {"smoke": cmd_smoke, "splits": cmd_splits, "eval": cmd_eval, "full": cmd_full}[args.command](cfg, args)


if __name__ == "__main__":
    main()
