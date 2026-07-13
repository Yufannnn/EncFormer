// model_config.h — Compile-time model dimensions for EncFormer native CKKS kernels.
//
// All constants are #ifndef-guarded so CMake can override them via -D flags.
// This allows building separate binaries for bert-base, bert-large, gpt2-base.
//
// Usage in CMakeLists.txt:
//   target_compile_definitions(bench_ckks_qkv_full_bert_large PRIVATE
//       ENCFORMER_M=128 ENCFORMER_D=1024 ENCFORMER_H=16
//       ENCFORMER_D_FF=4096 ENCFORMER_NSLOTS=32768)

#pragma once

// --- Primary model dimensions (must be set per config) ---

#ifndef ENCFORMER_NSLOTS
#define ENCFORMER_NSLOTS 16384
#endif

#ifndef ENCFORMER_M
#define ENCFORMER_M 128
#endif

#ifndef ENCFORMER_D
#define ENCFORMER_D 768
#endif

#ifndef ENCFORMER_H
#define ENCFORMER_H 12
#endif

#ifndef ENCFORMER_D_FF
#define ENCFORMER_D_FF 3072
#endif

// --- Derived constants (order matters — no forward references) ---

constexpr int EF_NSLOTS = ENCFORMER_NSLOTS;
constexpr int EF_M      = ENCFORMER_M;
constexpr int EF_D      = ENCFORMER_D;
constexpr int EF_H      = ENCFORMER_H;
constexpr int EF_D_FF   = ENCFORMER_D_FF;
constexpr int EF_DH     = EF_D / EF_H;
constexpr int EF_C      = EF_NSLOTS / EF_M;

// --- Column-usage adjustment for complex-paired packing ---
// When D/C is odd (e.g. gpt2: D=768, C=256 → G=3), complex-paired packing
// needs even groups. Use c_used = C/2 so G doubles and becomes even.
constexpr int EF_C_USED_LIN    = ((EF_D / EF_C) % 2 == 0) ? EF_C : (EF_C / 2);
constexpr int EF_C_USED_FF1_IN = ((EF_D / EF_C) % 2 == 0) ? EF_C : (EF_C / 2);
constexpr int EF_C_USED_FF2_IN = ((EF_D_FF / EF_C) % 2 == 0) ? EF_C : (EF_C / 2);

// --- Baby-step/giant-step decomposition ---
// N1*N2 must equal C (full column count) to cover all output segments.
// N1 ≈ sqrt(C), must divide C.  For C=128: N1=32, N2=4.  For C=256: N1=16, N2=16.
constexpr int EF_N1_DEFAULT = (EF_C <= 128) ? 32 : 16;
constexpr int EF_N2_DEFAULT = EF_C / EF_N1_DEFAULT;

// FF2 uses a smaller N1 (larger input dim → more groups → smaller baby step).
constexpr int EF_N1_FF2 = (EF_C <= 128) ? 8 : 16;
constexpr int EF_N2_FF2 = EF_C / EF_N1_FF2;

// --- QKV / linear projection ---
constexpr int EF_G_QKV    = EF_D / EF_C_USED_LIN;
constexpr int EF_BLOCKS   = (EF_D + EF_C_USED_LIN - 1) / EF_C_USED_LIN;
constexpr int EF_HP_QKV   = EF_G_QKV / 2;

// --- Score FDP constants ---
constexpr int EF_B_FOLD   = (EF_M >= 128) ? 16 : 8;
constexpr int EF_G_FOLD   = EF_M / EF_B_FOLD;
constexpr int EF_HALF_M   = EF_M / 2;
constexpr int EF_BLEN     = EF_H * EF_M;

// FDP packing for Q,K: use fewer columns per ct to reduce depth.
constexpr int EF_C_USED_QK = (EF_C * 3) / 4;
constexpr int EF_BLOCKS_QK = (EF_D + EF_C_USED_QK - 1) / EF_C_USED_QK;
constexpr int EF_C_USED_V  = EF_C;
constexpr int EF_BLOCKS_V  = EF_D / EF_C;

// --- FFN ---
constexpr int EF_G_FF1     = EF_D / EF_C_USED_FF1_IN;
constexpr int EF_B_FF1     = (EF_D_FF + EF_C - 1) / EF_C;
constexpr int EF_G_FF2     = EF_D_FF / EF_C_USED_FF2_IN;
constexpr int EF_B_FF2     = (EF_D + EF_C - 1) / EF_C;
constexpr int EF_HP_FF1    = EF_G_FF1 / 2;
constexpr int EF_HP_FF2    = EF_G_FF2 / 2;

// --- Value+OUT ---
constexpr int EF_OUT_BLOCKS = EF_BLOCKS;
constexpr int EF_G_SV       = EF_D / EF_C;
constexpr int EF_HP_OUT     = EF_G_QKV / 2;
