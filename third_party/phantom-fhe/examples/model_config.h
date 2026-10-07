#pragma once

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

constexpr int EF_NSLOTS = ENCFORMER_NSLOTS;
constexpr int EF_M      = ENCFORMER_M;
constexpr int EF_D      = ENCFORMER_D;
constexpr int EF_H      = ENCFORMER_H;
constexpr int EF_D_FF   = ENCFORMER_D_FF;
constexpr int EF_DH     = EF_D / EF_H;
constexpr int EF_C      = EF_NSLOTS / EF_M;

constexpr int EF_C_USED_LIN    = ((EF_D / EF_C) % 2 == 0) ? EF_C : (EF_C / 2);
constexpr int EF_C_USED_FF1_IN = ((EF_D / EF_C) % 2 == 0) ? EF_C : (EF_C / 2);
constexpr int EF_C_USED_FF2_IN = ((EF_D_FF / EF_C) % 2 == 0) ? EF_C : (EF_C / 2);

constexpr int EF_N1_DEFAULT = (EF_C <= 128) ? 32 : 16;
constexpr int EF_N2_DEFAULT = EF_C / EF_N1_DEFAULT;

constexpr int EF_N1_FF2 = (EF_C <= 128) ? 8 : 16;
constexpr int EF_N2_FF2 = EF_C / EF_N1_FF2;

constexpr int EF_G_QKV    = EF_D / EF_C_USED_LIN;
constexpr int EF_BLOCKS   = (EF_D + EF_C_USED_LIN - 1) / EF_C_USED_LIN;
constexpr int EF_HP_QKV   = EF_G_QKV / 2;

constexpr int EF_B_FOLD   = (EF_M >= 128) ? 16 : 8;
constexpr int EF_G_FOLD   = EF_M / EF_B_FOLD;
constexpr int EF_HALF_M   = EF_M / 2;
constexpr int EF_BLEN     = EF_H * EF_M;

constexpr int EF_C_USED_QK = (EF_C * 3) / 4;
constexpr int EF_BLOCKS_QK = (EF_D + EF_C_USED_QK - 1) / EF_C_USED_QK;
constexpr int EF_C_USED_V  = EF_C;
constexpr int EF_BLOCKS_V  = EF_D / EF_C;

constexpr int EF_G_FF1     = EF_D / EF_C_USED_FF1_IN;
constexpr int EF_B_FF1     = (EF_D_FF + EF_C - 1) / EF_C;
constexpr int EF_G_FF2     = EF_D_FF / EF_C_USED_FF2_IN;
constexpr int EF_B_FF2     = (EF_D + EF_C - 1) / EF_C;
constexpr int EF_HP_FF1    = EF_G_FF1 / 2;
constexpr int EF_HP_FF2    = EF_G_FF2 / 2;

constexpr int EF_OUT_BLOCKS = EF_BLOCKS;
constexpr int EF_G_SV       = EF_D / EF_C;
constexpr int EF_HP_OUT     = EF_G_QKV / 2;
