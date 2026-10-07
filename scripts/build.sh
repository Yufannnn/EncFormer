#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
arch="${1:-80}"
build="$root/third_party/phantom-fhe/build"
cmake -S "$root/third_party/phantom-fhe" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch"
cmake --build "$build" --parallel "$(nproc)" --target pipe_tp_server_bert_base pipe_tp_client_bert_base
bash "$root/scripts/build_ezpc_sci.sh"
