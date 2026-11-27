"""Check a grid run folder (rows, features files, meta.json) and write manifest.sha256. No GPU, no model weights.

  python scripts/check_run.py execution_results/mvp/grid_b8              # a finished run
  python scripts/check_run.py execution_results/mvp/grid_b8 --partial    # a running or stopped one

Exit code 1 when a check fails.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never takes a GPU, also next to a running job
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaloop import api  # noqa: E402


def main() -> None:
    """Parse arguments and check one folder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--partial", action="store_true", help="missing rows are reported, not a failure")
    args = parser.parse_args()
    sys.exit(0 if api.check_run(args.run_dir, partial=args.partial) else 1)


if __name__ == "__main__":
    main()
