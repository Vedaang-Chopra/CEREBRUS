#!/usr/bin/env bash
# Sync the committed repo to a server from the Mac. Usage: scripts/sync_server.sh <inara|pace> [rev=main]
# Only committed files are sent; nothing is deleted on the server.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=${1:?usage: scripts/sync_server.sh <inara|pace> [rev]}
REV=${2:-main}

case "$HOST" in
  inara)
    # sync_inara.sh always syncs main, so a different rev cannot be honoured
    [ "$REV" = main ] || { echo "ERROR: sync_inara.sh only syncs main (got rev $REV)" >&2; exit 1; }
    exec scripts/sync_inara.sh
    ;;
  pace)
    CTRL=$HOME/.ssh/cm-pace-compute
    # The control socket must already be open (this script does not log in)
    if ! ssh -S "$CTRL" -O check pace-compute >/dev/null 2>&1; then
      echo "ERROR: no open ssh control socket for pace-compute ($CTRL). Open it first, then rerun." >&2
      exit 1
    fi
    SHORT=$(git rev-parse --short "$REV^{commit}")
    # Server copy has no .git: send the committed tree and extract over it
    git archive "$REV" | ssh -S "$CTRL" -o ClearAllForwardings=yes pace-compute \
      'mkdir -p ~/scratch/projects/cerebrus && tar -x -C ~/scratch/projects/cerebrus'
    printf '%s\n' "$SHORT" | ssh -S "$CTRL" -o ClearAllForwardings=yes pace-compute \
      'cat > ~/scratch/projects/cerebrus/REVISION'
    echo "REVISION on pace-compute:"
    ssh -S "$CTRL" -o ClearAllForwardings=yes pace-compute 'cat ~/scratch/projects/cerebrus/REVISION'
    ;;
  *)
    echo "ERROR: host must be inara or pace (got $HOST)" >&2
    exit 1
    ;;
esac
