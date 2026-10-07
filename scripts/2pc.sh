#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
idx="${1:-40}"
layers="${2:-12}"
gpu="${3:-0}"
export EZPC_PYTHONPATH="$root/third_party/ezpc-sci/build"
export MPC_EZPC_THREADS="${MPC_EZPC_THREADS:-8}"
python -u "$root/scripts/demo.py" --idx "$idx" --layers "$layers" --gpu "$gpu" | grep --line-buffered -v '^connected$'
