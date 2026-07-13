#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
idx="${1:-40}"
layers="${2:-12}"
gpu="${3:-0}"
port="${PORT:-36600}"
ckpt="checkpoints/encformer-sst2"
logs="$(mktemp -d)"
export EZPC_PYTHONPATH="$root/third_party/ezpc-sci/build"
trap 'kill ${server:-0} ${client:-0} 2>/dev/null || true' EXIT
MPC_EZPC_MODE=native MPC_EZPC_ROLE=server MPC_EZPC_PORT="$port" NGPU="$gpu" MIDX="$idx" CKPT="$ckpt" NLAYERS="$layers" python -u scripts/demo.py --ckpt "$ckpt" --idx "$idx" --gpu "$gpu" --layers "$layers" >"$logs/server.log" 2>&1 &
server=$!
for _ in $(seq 1 300); do
    ss -ltn 2>/dev/null | grep -q ":$port " && break
    kill -0 "$server" 2>/dev/null || { cat "$logs/server.log"; exit 1; }
    sleep 1
done
MPC_EZPC_PORT="$port" NLAYERS="$layers" python -u scripts/demo_client.py "$port" >"$logs/client.log" 2>&1 &
client=$!
wait "$server"
wait "$client"
cat "$logs/server.log"
tail -5 "$logs/client.log"
