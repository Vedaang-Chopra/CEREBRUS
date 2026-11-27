#!/usr/bin/env bash
# Create the AdaLoop env on a Linux GPU server. Run from anywhere; writes only under $HOME/scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_DIR=${ENV_DIR:-$HOME/scratch/envs/adaloop}

# Caches and temp files go under scratch, not $HOME (keep values already set)
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/scratch/cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$HOME/scratch/uv_python}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$HOME/scratch/cache/pip}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/scratch/cache}
export TMPDIR=${TMPDIR:-$HOME/scratch/tmp}
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

# Bootstrap uv in its own venv (skip if already there)
UVBOOT=$HOME/scratch/envs/uvboot
if [ ! -x "$UVBOOT/bin/uv" ]; then
  mkdir -p "$HOME/scratch/envs"
  python3 -m venv "$UVBOOT"
  "$UVBOOT/bin/pip" install uv
fi
UV=$UVBOOT/bin/uv

# pyproject requires Python 3.13; stop if it cannot be fetched
if ! "$UV" python install 3.13; then
  echo "ERROR: could not fetch Python 3.13 (pyproject.toml requires 3.13). Stopping." >&2
  exit 1
fi

# Project venv (skip if it exists)
if [ -x "$ENV_DIR/bin/python" ]; then
  echo "Env exists, skipping venv creation: $ENV_DIR"
else
  mkdir -p "$(dirname "$ENV_DIR")"
  "$UV" venv --python 3.13 "$ENV_DIR"
fi

# Pins live in pyproject.toml (torch, transformers commit, peft)
"$UV" pip install --python "$ENV_DIR/bin/python" -e .

# spaCy model wheel, read from environment.yml so the URL is not duplicated
SPACY_URL=$(grep -o 'https://[^ ]*en_core_web_sm[^ ]*\.whl' environment.yml || true)
if [ -z "$SPACY_URL" ]; then
  echo "ERROR: en_core_web_sm URL not found in environment.yml" >&2
  exit 1
fi
"$UV" pip install --python "$ENV_DIR/bin/python" "$SPACY_URL"

# Quick check of the stack and GPU
"$ENV_DIR/bin/python" - <<'EOF'
import sys

import torch
import transformers

print("python:", sys.executable, sys.version.split()[0])
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
EOF
