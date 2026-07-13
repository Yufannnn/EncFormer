#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
sci="$root/third_party/ezpc/SCI"
sci_build="$sci/build"
binding="$root/third_party/ezpc-sci"
binding_build="$binding/build"
[ -d "$sci" ] || { echo "missing $sci"; exit 1; }
seal_build="$sci_build/seal-build"
cmake -S "$sci/extern/SEAL/native/src" -B "$seal_build" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$sci_build" -DSEAL_BUILD_TESTS=OFF
cmake --build "$seal_build" --parallel "$(nproc)" --target install
cmake -S "$sci" -B "$sci_build" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$sci_build/install" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_TESTS=OFF -DBUILD_NETWORKS=OFF
cmake --build "$sci_build" --parallel "$(nproc)"
python="${PYTHON:-$(command -v python || command -v python3)}"
pybind11="$($python -m pybind11 --cmakedir)"
cmake -S "$binding" -B "$binding_build" -DCMAKE_BUILD_TYPE=Release -DEZPC_SCI_PYTHON_BINDING=ON -DEZPC_STANDALONE_TEST=ON -DSCI_BUILD_DIR="$sci_build" -Dpybind11_DIR="$pybind11" -DPYTHON_EXECUTABLE="$python"
cmake --build "$binding_build" --parallel "$(nproc)"
"$binding_build/ezpc_sci_test"
PYTHONPATH="$binding_build" "$python" -c 'import ezpc_sci; assert ezpc_sci.HAS_NATIVE_SCI; print("EncFormer EzPC/SCI ready")'
