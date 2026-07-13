#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
arch="${1:-70}"
build="$root/third_party/phantom-fhe/build"
cmake -S "$root/third_party/phantom-fhe" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch"
cmake --build "$build" --parallel "$(nproc)" --target pipe_ckks_attn_bert_base pipe_ckks_ff1_bert_base pipe_ckks_ff2_bert_base
bash "$root/scripts/build_ezpc_sci.sh"
