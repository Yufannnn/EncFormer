// third_party/phantom-fhe/examples/bench_ckks_s_small.cu
//
// S stage (QK^T) — SMALL ring — BSGS eager relin — packed 1-CT output
// FIXED: avoid "scale mismatch" by encoding ALL masks at scale=1.0
//
// Build:
//   cmake -S third_party/phantom-fhe -B third_party/phantom-fhe/build -DCMAKE_BUILD_TYPE=Release
//   cmake --build third_party/phantom-fhe/build --target bench_ckks_s_small -j
// Run:
//   CUDA_VISIBLE_DEVICES=0 ./third_party/phantom-fhe/build/bin/bench_ckks_s_small

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "example.h"     // print_parameters(...)
#include "phantom.h"
#include "util.cuh"
#include "evaluate.cuh"  // rotate_inplace, apply_galois_inplace, add_inplace, multiply_plain_inplace, multiply_inplace, relinearize_inplace, ...

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

using pc64 = cuDoubleComplex;
static inline pc64 pcplx(double r, double i = 0.0) { return make_cuDoubleComplex(r, i); }
static inline double preal(const pc64 &z) { return cuCreal(z); }
static inline double pimag(const pc64 &z) { return cuCimag(z); }

static inline void cuda_sync() {
    auto e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

static inline double ms_since(std::chrono::high_resolution_clock::time_point t0,
                              std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count() * 1e3;
}

static void banner(const std::string &
