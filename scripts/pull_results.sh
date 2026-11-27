#!/usr/bin/env bash
# Pull result run dirs from a server into local execution_results/mvp/.
# Usage: scripts/pull_results.sh <inara|pace> [run dirs...]   (default: grid_b8; pace: grid_b8_padded)
# A run dir comes whole: rows, parquet, features*/, meta.json, README.md, manifest.sha256.
# --update: only newer files are copied; no --delete, so local files are never removed.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=${1:?usage: scripts/pull_results.sh <inara|pace> [run dirs...]}
shift
if [ $# -eq 0 ]; then
  if [ "$HOST" = pace ]; then set -- grid_b8_padded; else set -- grid_b8; fi
fi

case "$HOST" in
  pace)
    REMOTE='pace-compute:~/scratch/projects/cerebrus'
    RSYNC_SSH="ssh -S $HOME/.ssh/cm-pace-compute -o ClearAllForwardings=yes"
    ;;
  inara)
    REMOTE='inara:/coc/scratch/vchopra/cerberus_v2'
    RSYNC_SSH=ssh
    ;;
  *)
    echo "ERROR: host must be inara or pace (got $HOST)" >&2
    exit 1
    ;;
esac

for d in "$@"; do
  mkdir -p "execution_results/mvp/$d"
  rsync -a --update -e "$RSYNC_SSH" "$REMOTE/execution_results/mvp/$d/" "execution_results/mvp/$d/"
done
