"""Build the post-training candidate store (data only: no GPU, no model weights). Thin CLI over adaloop.api.

  python scripts/build_post_data.py --config-name vqav2 --rows 50   # smoke: one source into <out>/smoke
  python scripts/build_post_data.py                                 # every source; rerun the same command to resume
  python scripts/build_post_data.py --out /abs/path/post_data       # other output folder

HF_HOME comes from the environment (GQA and TextVQA must be cached or downloadable there).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/post_training_data.yaml", help="relative to the repo root")
    parser.add_argument("--out", help="store folder (default: out_dir of the config)")
    parser.add_argument("--config-name", help="smoke run of this one source")
    parser.add_argument("--rows", type=int, help="candidates in the smoke run (default: smoke_rows)")
    args = parser.parse_args()
    if args.rows and not args.config_name:
        parser.error("--rows needs --config-name")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only; must precede the import of adaloop.api
    from adaloop import api

    cfg = api.load_config(ROOT / args.config)
    folder = api.build_post_data(cfg, ROOT, out_dir=args.out, config_name=args.config_name, rows=args.rows)
    print(f"store: {folder}")


if __name__ == "__main__":
    main()
