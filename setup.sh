#!/usr/bin/env bash
# One-shot setup: create venv, install deps, download model weights, fix tokenizer.
# Usage: ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found — install Python 3.10+ first." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo ">>> Creating venv at .venv"
  python3 -m venv .venv
fi

echo ">>> Installing Python dependencies"
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

MODEL_DIR="${QWEN3_ASR_MODEL:-Qwen3-ASR-1.7B}"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo ">>> Downloading Qwen/Qwen3-ASR-1.7B weights to $MODEL_DIR (~4.4 GB)"
  .venv/bin/pip install --upgrade "huggingface_hub[cli]"
  .venv/bin/huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir "$MODEL_DIR"
else
  echo ">>> Model weights already present at $MODEL_DIR"
fi

if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
  echo ">>> Generating fast tokenizer.json (model ships only the slow tokenizer)"
  .venv/bin/python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL_DIR', use_fast=True)
tok.save_pretrained('$MODEL_DIR')
print('OK — tokenizer.json written')
"
fi

echo
echo ">>> Done. Start the server with:"
echo "    .venv/bin/python main.py"
