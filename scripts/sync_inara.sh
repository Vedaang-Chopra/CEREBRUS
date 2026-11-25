#!/usr/bin/env bash
# Send committed `main` to the Inara clone and fast-forward it. Uncommitted work here is not sent.
# Notebooks Vedaang ran on Inara (outputs saved, so "modified") that the incoming commits also change are
# copied to execution_results/notebook_runs/<name>_<time>.ipynb first, then reset, so the merge never blocks.
# Other uncommitted edits on Inara are kept. Needs passwordless `ssh inara` (~/.ssh/config).
set -euo pipefail
REMOTE_DIR=/coc/scratch/vchopra/cerberus_v2
QUIET='^\s+\*|cargo|LFS|lfs|^\s*$'
git remote get-url inara >/dev/null 2>&1 || git remote add inara "inara:$REMOTE_DIR"
git push -q inara main:refs/remotes/mac/main 2>&1 | grep -v -E "$QUIET" || true
ssh inara "cd $REMOTE_DIR && set -e
  stamp=\$(date +%Y%m%d_%H%M%S)
  incoming=\$(git diff --name-only HEAD mac/main)
  for f in \$(git diff --name-only -- '*.ipynb'); do
    if echo \"\$incoming\" | grep -qx \"\$f\"; then
      mkdir -p execution_results/notebook_runs
      cp \"\$f\" \"execution_results/notebook_runs/\$(basename \"\$f\" .ipynb)_\$stamp.ipynb\"
      git checkout -- \"\$f\"
      echo \"backed up run: \$f -> execution_results/notebook_runs/\$(basename \"\$f\" .ipynb)_\$stamp.ipynb\"
    fi
  done
  git merge --ff-only -q mac/main && git log --oneline -1" 2>&1 | grep -v -E "$QUIET"
