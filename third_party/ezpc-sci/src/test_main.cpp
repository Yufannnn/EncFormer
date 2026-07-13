// test_main.cpp — Quick sanity check for the C++ protocol implementations.
// Build with: cmake -DEZPC_STANDALONE_TEST=ON .. && make ezpc_sci_test

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ezpc_sci/config.h"
#include "ezpc_sci/fixedpoint.h"
#include "ezpc_sci/gelu.h"
#include "ezpc_sci/layernorm.h"
#include "ezpc_sci/softmax.h"

using namespace ezpc_sci;

static void test_fixedpoint_roundtrip() {
    ProtocolConfig cfg{43, 13, 2.7};
    double vals[] = {1.5, -0.25, 0.0, 3.14159};
    for (double v : vals) {
        int64_t enc = fp::encode(v, cfg);
        double dec = fp::decode(enc, cfg);
        assert(std::abs(dec - v) < 1e-3);
    }
    printf("  PASS: fixedpoint roundtrip\n");
}

static void test_fixedpoint_mul() {
    ProtocolConfig cfg{43, 13, 2.7};
    int64_t a = fp::encode(2.0, cfg);
    int64_t b = fp::encode(3.0, cfg);
    int64_t c = fp::mul(a, b, cfg);
    double result = fp::decode(c, cfg);
    assert(std::abs(result - 6.0) < 0.05);
    printf("  PASS: fixedpoint multiply\n");
}

static void test_softmax() {
    ProtocolConfig cfg = default_config();
    double in[] = {1.0, 2.0, 3.0, 4.0};
    double out[4];
    softmax_rows(in, out, 1, 4, cfg);
    double sum = 0;
    for (int i = 0; i < 4; i++) {
        assert(out[i] > 0);
        sum += out[i];
    }
    assert(std::abs(sum - 1.0) < 1e-6);
    printf("  PASS: softmax rows sum to 1\n");
}

static void test_layernorm() {
    ProtocolConfig cfg = default_config();
    double in[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double out[8];
    layer_norm(in, out, 1, 8, 1e-5, cfg);

    // Check near-zero mean
    double mean = 0;
    for (int i = 0; i < 8; i++) mean += out[i];
    mean /= 8.0;
    assert(std::abs(mean) < 1e-5);

    // Check unit variance
    double var = 0;
    for (int i = 0; i < 8; i++) var += (out[i] - mean) * (out[i] - mean);
    var /= 8.0;
    assert(std::abs(var - 1.0) < 0.05);
    printf("  PASS: layer norm zero mean unit variance\n");
}

static void test_gelu() {
    ProtocolConfig cfg = default_config();

    // GELU(0) ≈ 0
    double in_zero[] = {0.0};
    double out_zero[1];
    gelu_vec(in_zero, out_zero, 1, cfg);
    assert(std::abs(out_zero[0]) < 0.05);

    // GELU(large positive) > 0
    double in_pos[] = {2.0, 3.0};
    double out_pos[2];
    gelu_vec(in_pos, out_pos, 2, cfg);
    assert(out_pos[0] > 0);
    assert(out_pos[1] > 0);

    printf("  PASS: gelu basic properties\n");
}

int main() {
    printf("Running ezpc_sci C++ tests...\n");
    test_fixedpoint_roundtrip();
    test_fixedpoint_mul();
    test_softmax();
    test_layernorm();
    test_gelu();
    printf("All tests passed.\n");
    return 0;
}
