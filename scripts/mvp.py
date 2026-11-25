"""Thin CLI over adaloop.api. Run from the repo root on the GPU host.

  python scripts/mvp.py smoke
  python scripts/mvp.py splits
  python scripts/mvp.py eval --benchmark gqa --split dev --loops 1 3 5 --limit 20

`smoke` and `eval` replace Jiaqi-Gao/adaloop-vlm:scripts/{smoke_test,run_grid}.py @ e870bff.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaloop import api  # noqa: E402

logger = logging.getLogger("mvp")
LOADERS = {"gqa": api.load_gqa}


def cmd_smoke(cfg: dict, args: argparse.Namespace) -> None:
    """Load the model and answer one dev item at R=3."""
    vlm = api.load_model(cfg)
    item = api.load_gqa(cfg, "dev", limit=1, root=ROOT)[0]
    row = api.evaluate_item(vlm, item, cfg, R=3)
    print(json.dumps({k: row[k] for k in ("question_type", "pred", "answers", "correct", "n_vis", "generate_s")}))


def cmd_splits(cfg: dict, args: argparse.Namespace) -> None:
    """Create (or check) the split manifests and print their sizes."""
    manifest = api.build_gqa_splits(cfg, ROOT)
    print({name: len(ids) for name, ids in manifest["splits"].items()}, "hash", manifest["hash"])


def cmd_eval(cfg: dict, args: argparse.Namespace) -> None:
    """Evaluate one benchmark split at each requested R (full tokens), resumable."""
    if args.split == "test":
        raise SystemExit("the test split is frozen until stage S12")
    items = LOADERS[args.benchmark](cfg, args.split, limit=args.limit, root=ROOT)
    vlm = api.load_model(cfg)
    out_dir = ROOT / cfg["out_dir"] / "eval"
    for R in args.loops:
        out = out_dir / f"{args.benchmark}_{args.split}_n{len(items)}_res{cfg['res_px']}_R{R}.jsonl"
        summary = api.run_evaluation(vlm, items, cfg, R, out, resume=True)
        print(json.dumps(summary))


def main() -> None:
    """Parse arguments and run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "splits", "eval"])
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--benchmark", default="gqa", choices=sorted(LOADERS))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--loops", type=int, nargs="+", default=[3])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = api.load_config(ROOT / args.config)
    {"smoke": cmd_smoke, "splits": cmd_splits, "eval": cmd_eval}[args.command](cfg, args)


if __name__ == "__main__":
    main()
