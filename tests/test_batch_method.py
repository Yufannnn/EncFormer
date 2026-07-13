#!/usr/bin/env python3


from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engines.ckks_engine_plain import CKKSContext
from src.engines.mpc_batch_method import BatchMethodConfig, BatchMethodState
from src.engines.mpc_engine_plain import PlainMpcEngine

SEED = 42
M = 128
D = 768
H = 12


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def test_bpmax_softmax():

    np.random.seed(SEED)
    print("=" * 60)
    print("TEST: BPMax Softmax (EncFormer batch method)")
    print("=" * 60)

    scores = [np.random.randn(M, M).astype(np.float64) for _ in range(H)]

    std_cfg = BatchMethodConfig(mode="off", inference=False, p=5, c=5.0, eps=1e-12, ln_l=1.0, rd_path="")
    std_engine = PlainMpcEngine.__new__(PlainMpcEngine)
    std_engine._batch_cfg = std_cfg
    std_engine._batch_state = BatchMethodState(std_cfg)

    std_results = []
    for h in range(H):
        std_results.append(std_engine.softmax_rows(scores[h], head_index=h))

    bp_train_cfg = BatchMethodConfig(mode="on", inference=False, p=5, c=5.0, eps=1e-12, ln_l=1.0, rd_path="")
    bp_train_engine = PlainMpcEngine.__new__(PlainMpcEngine)
    bp_train_engine._batch_cfg = bp_train_cfg
    bp_train_engine._batch_state = BatchMethodState(bp_train_cfg)

    bp_train_results = []
    for h in range(H):
        bp_train_results.append(bp_train_engine.softmax_rows(scores[h], head_index=h))

    for h in range(H):
        row_sums = bp_train_results[h].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Head {h}: BPMax rows don't sum to 1 (max deviation: {np.max(np.abs(row_sums - 1.0)):.3e})"
        )
        assert np.all(bp_train_results[h] >= 0), f"Head {h}: BPMax has negative values"

    rd_key = "bpmax_running_denominator"
    rd = bp_train_engine._batch_state.buffers.get(rd_key)
    print(f"  Running denominator shape: {rd.shape if rd is not None else 'None'}")
    print(f"  Running denominator range: [{rd.min():.2f}, {rd.max():.2f}]")
    assert rd.shape[0] >= H, f"Expected at least {H} heads, got {rd.shape[0]}"

    bp_inf_cfg = BatchMethodConfig(mode="on", inference=True, p=5, c=5.0, eps=1e-12, ln_l=1.0, rd_path="")
    bp_inf_engine = PlainMpcEngine.__new__(PlainMpcEngine)
    bp_inf_engine._batch_cfg = bp_inf_cfg
    bp_inf_engine._batch_state = BatchMethodState(bp_inf_cfg)

    bp_inf_engine._batch_state.buffers[rd_key] = rd

    bp_inf_results = []
    for h in range(H):
        bp_inf_results.append(bp_inf_engine.softmax_rows(scores[h], head_index=h))

    for h in range(H):
        err_train = rel_err(bp_train_results[h], std_results[h])
        err_inf = rel_err(bp_inf_results[h], std_results[h])
        err_train_vs_inf = rel_err(bp_train_results[h], bp_inf_results[h])
        if h == 0:
            print(f"\n  Head {h} (example):")
            print(f"    std softmax row[0] top-5:  {np.sort(std_results[h][0])[-5:]}")
            print(f"    BPMax train row[0] top-5:  {np.sort(bp_train_results[h][0])[-5:]}")
            print(f"    BPMax infer row[0] top-5:  {np.sort(bp_inf_results[h][0])[-5:]}")
        if h < 3 or h == H - 1:
            print(
                f"  Head {h:2d}: err(BPMax_train vs std)={err_train:.3e}  "
                f"err(BPMax_inf vs std)={err_inf:.3e}  "
                f"err(train vs inf)={err_train_vs_inf:.3e}"
            )

    for h in range(H):
        assert rel_err(bp_train_results[h], bp_inf_results[h]) < 1e-10, (
            f"Head {h}: Training and inference results diverge"
        )

    print("\n  PASS: BPMax softmax works correctly")
    print(f"  Note: BPMax vs standard softmax differ (expected) - this is a different function,")
    print(f"        not an approximation of softmax. Model must be trained with BPMax.")


def test_batchln():

    np.random.seed(SEED + 1)
    print("\n" + "=" * 60)
    print("TEST: BatchLN LayerNorm (EncFormer batch method)")
    print("=" * 60)

    X = np.random.randn(M, D).astype(np.float64) * 5.0
    gamma = np.random.randn(D).astype(np.float64)
    beta = np.random.randn(D).astype(np.float64)

    std_cfg = BatchMethodConfig(mode="off", inference=False, p=5, c=5.0, eps=1e-12, ln_l=1.0, rd_path="")
    std_engine = PlainMpcEngine.__new__(PlainMpcEngine)
    std_engine._batch_cfg = std_cfg
    std_engine._batch_state = BatchMethodState(std_cfg)
    Y_std = std_engine.layer_norm(X, eps=1e-5, gamma=gamma, beta=beta, ln_tag="ln1")

    bln_train_cfg = BatchMethodConfig(mode="on", inference=False, p=5, c=5.0, eps=1e-12, ln_l=1.0, rd_path="")
    bln_train_engine = PlainMpcEngine.__new__(PlainMpcEngine)
    bln_train_engine._batch_cfg = bln_train_cfg
    bln_train_engine._batch_state = BatchMethodState(bln_train_cfg)
    Y_train = bln_train_engine.layer_norm(X, eps=1e-5, gamma=gamma, beta=beta, ln_tag="ln1")

    rd_key = "ln1_running_denominator"
    rd = bln_train_engine._batch_state.buffers.get(rd_key)
    print(f"  Running denominator shape: {rd.shape if rd is not None else 'None'}")
    print(f"  Running denominator range: [{rd.min():.4f}, {rd.max():.4f}]")

    bln_inf_cfg = BatchMethodConfig(mode="on", inference=True, p=5, c=5.0, eps=1e-12, ln_l=1.0, rd_path="")
    bln_inf_engine = PlainMpcEngine.__new__(PlainMpcEngine)
    bln_inf_engine._batch_cfg = bln_inf_cfg
    bln_inf_engine._batch_state = BatchMethodState(bln_inf_cfg)
    bln_inf_engine._batch_state.buffers[rd_key] = rd
    Y_inf = bln_inf_engine.layer_norm(X, eps=1e-5, gamma=gamma, beta=beta, ln_tag="ln1")

    err_train = rel_err(Y_train, Y_std)
    err_inf = rel_err(Y_inf, Y_std)
    err_tv = rel_err(Y_train, Y_inf)

    print(f"\n  err(BatchLN_train vs std_LN) = {err_train:.3e}")
    print(f"  err(BatchLN_inf vs std_LN)   = {err_inf:.3e}")
    print(f"  err(train vs inf)            = {err_tv:.3e}")

    assert err_train < 1e-6, f"BatchLN training mode too far from standard: {err_train:.3e}"
    assert err_tv < 1e-10, f"Training and inference modes diverge: {err_tv:.3e}"

    print("\n  PASS: BatchLN works correctly")
    print(f"  Note: With ln_l=1.0 and same-sample denominator, BatchLN ≈ standard LN")


def test_pipeline_integration():

    np.random.seed(SEED)
    print("\n" + "=" * 60)
    print("TEST: Full pipeline integration with batch method")
    print("=" * 60)

    ctx = CKKSContext(16384)

    A = np.random.randn(M, D).astype(np.float64)
    gamma1 = np.random.randn(D).astype(np.float64)
    beta1 = np.random.randn(D).astype(np.float64)
    gamma2 = np.random.randn(D).astype(np.float64)
    beta2 = np.random.randn(D).astype(np.float64)

    from src.utils import to_ckks_mat, to_mpc_mat

    X_blocks = to_ckks_mat(ctx, A, m=M, use_cc=True)

    os.environ["MPC_BATCH_METHOD"] = "off"
    os.environ["MPC_BATCH_INFERENCE"] = "0"
    from src.engines.mpc_engine_factory import get_mpc_engine

    engine_std = get_mpc_engine("plain")

    from src.mpc.round_b import roundB
    from src.mpc.round_d import roundD

    LN1_std = roundB(ctx, X_blocks, m=M, d_out=D, gamma=gamma1, beta=beta1, mpc_engine=engine_std)
    LN2_std = roundD(ctx, X_blocks, m=M, d_out=D, gamma=gamma2, beta=beta2, mpc_engine=engine_std)

    LN1_std_dec, _ = to_mpc_mat(ctx, LN1_std, m=M, d_out=D, use_cc=True)
    LN2_std_dec, _ = to_mpc_mat(ctx, LN2_std, m=M, d_out=D, use_cc=True)

    os.environ["MPC_BATCH_METHOD"] = "on"
    os.environ["MPC_BATCH_INFERENCE"] = "0"
    engine_batch = get_mpc_engine("plain")

    LN1_batch = roundB(ctx, X_blocks, m=M, d_out=D, gamma=gamma1, beta=beta1, mpc_engine=engine_batch)
    LN2_batch = roundD(ctx, X_blocks, m=M, d_out=D, gamma=gamma2, beta=beta2, mpc_engine=engine_batch)

    LN1_batch_dec, _ = to_mpc_mat(ctx, LN1_batch, m=M, d_out=D, use_cc=True)
    LN2_batch_dec, _ = to_mpc_mat(ctx, LN2_batch, m=M, d_out=D, use_cc=True)

    err_ln1 = rel_err(LN1_batch_dec, LN1_std_dec)
    err_ln2 = rel_err(LN2_batch_dec, LN2_std_dec)

    print(f"  roundB (LN1): err(BatchLN vs std) = {err_ln1:.3e}")
    print(f"  roundD (LN2): err(BatchLN vs std) = {err_ln2:.3e}")

    rd_ln1 = engine_batch._batch_state.buffers.get("ln1_running_denominator")
    rd_ln2 = engine_batch._batch_state.buffers.get("ln2_running_denominator")
    print(f"  ln1 running_denominator: shape={rd_ln1.shape if rd_ln1 is not None else None}")
    print(f"  ln2 running_denominator: shape={rd_ln2.shape if rd_ln2 is not None else None}")

    assert err_ln1 < 1e-6, f"roundB BatchLN too far from standard: {err_ln1:.3e}"
    assert err_ln2 < 1e-6, f"roundD BatchLN too far from standard: {err_ln2:.3e}"

    os.environ["MPC_BATCH_METHOD"] = "on"
    os.environ["MPC_BATCH_INFERENCE"] = "1"
    engine_inf = get_mpc_engine("plain")

    engine_inf._batch_state.buffers["ln1_running_denominator"] = rd_ln1
    engine_inf._batch_state.buffers["ln2_running_denominator"] = rd_ln2

    LN1_inf = roundB(ctx, X_blocks, m=M, d_out=D, gamma=gamma1, beta=beta1, mpc_engine=engine_inf)
    LN1_inf_dec, _ = to_mpc_mat(ctx, LN1_inf, m=M, d_out=D, use_cc=True)
    err_inf = rel_err(LN1_inf_dec, LN1_std_dec)
    print(f"  roundB inference mode: err(BatchLN_inf vs std) = {err_inf:.3e}")
    assert err_inf < 1e-6, f"roundB inference mode too far: {err_inf:.3e}"

    print("\n  PASS: Pipeline integration works correctly")

    for k in ["MPC_BATCH_METHOD", "MPC_BATCH_INFERENCE"]:
        os.environ.pop(k, None)


def test_save_load_denominators():

    np.random.seed(SEED)
    print("\n" + "=" * 60)
    print("TEST: Save/load running denominators")
    print("=" * 60)

    import tempfile

    rd_bpmax = np.random.rand(M, 1).astype(np.float64) * 1000 + 100
    rd_ln1 = np.random.rand(M, 1).astype(np.float64) * 5 + 1
    rd_ln2 = np.random.rand(M, 1).astype(np.float64) * 5 + 1

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmppath = f.name
        np.savez(f, bpmax_running_denominator=rd_bpmax, ln1_running_denominator=rd_ln1, ln2_running_denominator=rd_ln2)

    try:
        cfg = BatchMethodConfig(mode="on", inference=True, p=5, c=5.0, eps=1e-12, ln_l=1.0, rd_path=tmppath)
        state = BatchMethodState(cfg)

        assert "bpmax_running_denominator" in state.buffers
        assert "ln1_running_denominator" in state.buffers
        assert "ln2_running_denominator" in state.buffers

        assert np.allclose(state.buffers["bpmax_running_denominator"], rd_bpmax)
        assert np.allclose(state.buffers["ln1_running_denominator"], rd_ln1)
        assert np.allclose(state.buffers["ln2_running_denominator"], rd_ln2)

        den = state.get_bpmax_den(M)
        assert den is not None
        den_ln1 = state.get_ln_den(M, ln_tag="ln1")
        assert den_ln1 is not None
        den_ln2 = state.get_ln_den(M, ln_tag="ln2")
        assert den_ln2 is not None

        print(f"  Loaded bpmax_rd: shape={state.buffers['bpmax_running_denominator'].shape}")
        print(f"  Loaded ln1_rd:   shape={state.buffers['ln1_running_denominator'].shape}")
        print(f"  Loaded ln2_rd:   shape={state.buffers['ln2_running_denominator'].shape}")
        print("\n  PASS: Save/load works correctly")
    finally:
        os.unlink(tmppath)


if __name__ == "__main__":
    test_bpmax_softmax()
    test_batchln()
    test_pipeline_integration()
    test_save_load_denominators()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
