#!/usr/bin/env bash
# Send committed `main` to the Inara clone and fast-forward it. Uncommitted work is not sent.
# Needs passwordless `ssh inara` (~/.ssh/config). Inara's own uncommitted edits are kept.
set -euo pipefail
REMOTE_DIR=/coc/scratch/vchopra/cerberus_v2
git remote get-url inara >/dev/null 2>&1 || git remote add inara "inara:$REMOTE_DIR"
git push -q inara main:refs/remotes/mac/main 2>&1 | grep -v -E '^\s+\*|cargo|LFS|lfs' || true
ssh inara "cd $REMOTE_DIR && git merge --ff-only -q mac/main && git log --oneline -1" 2>&1 | grep -v -E '^\s+\*|cargo|^\s*$'
