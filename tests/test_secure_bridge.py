#!/usr/bin/env python3


from __future__ import annotations

import os
import socket
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bridges.channel import InProcessChannelPair, SocketChannel
from src.bridges.in_process_bridge import InProcessBridge
from src.bridges.secure_bridge import Role, SecureBridge
from src.engines.ckks_engine_plain import CKKSContext
from src.engines.mpc_engine_plain import PlainShare
from src.utils_comm import CommStats

SEED = 42
NSLOTS = 4096
REL_TOL = 1e-3


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def test_in_process_channel_basic():

    ch_a, ch_b = InProcessChannelPair.create()
    data = np.array([1.0, 2.0, 3.0])
    ch_a.send("hello", data)
    got = ch_b.recv("hello")
    np.testing.assert_array_equal(got, data)
    print("PASS: test_in_process_channel_basic")


def test_in_process_channel_tagged():

    ch_a, ch_b = InProcessChannelPair.create()
    ch_a.send("x", np.array([1.0]))
    ch_a.send("y", np.array([2.0]))
    got_y = ch_b.recv("y")
    got_x = ch_b.recv("x")
    np.testing.assert_array_equal(got_x, [1.0])
    np.testing.assert_array_equal(got_y, [2.0])
    print("PASS: test_in_process_channel_tagged")


def test_in_process_channel_bidirectional():

    ch_a, ch_b = InProcessChannelPair.create()
    ch_a.send("a2b", np.array([10.0]))
    ch_b.send("b2a", np.array([20.0]))
    np.testing.assert_array_equal(ch_b.recv("a2b"), [10.0])
    np.testing.assert_array_equal(ch_a.recv("b2a"), [20.0])
    print("PASS: test_in_process_channel_bidirectional")


def test_real_shares_sum_to_original():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    x = np.random.randn(NSLOTS)
    ct = ctx.encorypt(x.astype(np.complex128))

    server_ch, client_ch = InProcessChannelPair.create()
    server = SecureBridge(Role.SERVER, server_ch, ctx)
    client = SecureBridge(Role.CLIENT, client_ch)

    s_share = server.ckks_to_mpc(ct)
    c_share = client.ckks_to_mpc(None)
    reconstructed = c_share + s_share

    err = rel_err(reconstructed, x)
    assert err < REL_TOL, f"Reconstruction error {err:.6e} exceeds tolerance {REL_TOL}"
    print(f"PASS: test_real_shares_sum_to_original (rel_err={err:.4e})")


def test_real_server_share_hides_value():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    x = np.random.randn(NSLOTS)
    ct = ctx.encorypt(x.astype(np.complex128))

    server_ch, client_ch = InProcessChannelPair.create()
    server = SecureBridge(Role.SERVER, server_ch, ctx)
    client = SecureBridge(Role.CLIENT, client_ch)

    s_share = server.ckks_to_mpc(ct)
    _ = client.ckks_to_mpc(None)

    corr = abs(np.corrcoef(s_share.ravel(), x.ravel())[0, 1])

    assert corr < 0.05, f"Server share correlation {corr:.4f} is suspiciously high"
    print(f"PASS: test_real_server_share_hides_value (|corr|={corr:.4f})")


def test_real_client_share_hides_value():

    np.random.seed(SEED + 1)
    ctx = CKKSContext(NSLOTS)
    x = np.random.randn(NSLOTS)
    ct = ctx.encorypt(x.astype(np.complex128))

    server_ch, client_ch = InProcessChannelPair.create()
    server = SecureBridge(Role.SERVER, server_ch, ctx)
    client = SecureBridge(Role.CLIENT, client_ch)

    _ = server.ckks_to_mpc(ct)
    c_share = client.ckks_to_mpc(None)

    corr = abs(np.corrcoef(c_share.ravel(), x.ravel())[0, 1])
    assert corr < 0.05, f"Client share correlation {corr:.4f} is suspiciously high"
    print(f"PASS: test_real_client_share_hides_value (|corr|={corr:.4f})")


def test_real_roundtrip():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    x = np.random.randn(NSLOTS)
    ct = ctx.encorypt(x.astype(np.complex128))

    server_ch, client_ch = InProcessChannelPair.create()
    server = SecureBridge(Role.SERVER, server_ch, ctx)
    client = SecureBridge(Role.CLIENT, client_ch)

    s_share = server.ckks_to_mpc(ct)
    c_share = client.ckks_to_mpc(None)

    client.mpc_to_ckks(c_share)
    ct2 = server.mpc_to_ckks(s_share)

    with ctx.decorypt_scope():
        dec = ct2.decorypt(copy=True, readonly=True)
    result = dec.real

    err = rel_err(result, x)
    assert err < REL_TOL, f"Roundtrip error {err:.6e} exceeds tolerance {REL_TOL}"
    print(f"PASS: test_real_roundtrip (rel_err={err:.4e})")


def test_complex_shares_sum_to_original():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    z = np.random.randn(NSLOTS) + 1j * np.random.randn(NSLOTS)
    ct = ctx.encorypt(z)

    server_ch, client_ch = InProcessChannelPair.create()
    server = SecureBridge(Role.SERVER, server_ch, ctx)
    client = SecureBridge(Role.CLIENT, client_ch)

    s_re, s_im = server.complex_ckks_to_mpc(ct)
    c_re, c_im = client.complex_ckks_to_mpc(None)

    rec_re = c_re + s_re
    rec_im = c_im + s_im

    err_re = rel_err(rec_re, z.real)
    err_im = rel_err(rec_im, z.imag)
    assert err_re < REL_TOL, f"Real part error {err_re:.6e}"
    assert err_im < REL_TOL, f"Imag part error {err_im:.6e}"
    print(f"PASS: test_complex_shares_sum_to_original (re={err_re:.4e}, im={err_im:.4e})")


def test_complex_roundtrip():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    z = np.random.randn(NSLOTS) + 1j * np.random.randn(NSLOTS)
    ct = ctx.encorypt(z)

    server_ch, client_ch = InProcessChannelPair.create()
    server = SecureBridge(Role.SERVER, server_ch, ctx)
    client = SecureBridge(Role.CLIENT, client_ch)

    s_re, s_im = server.complex_ckks_to_mpc(ct)
    c_re, c_im = client.complex_ckks_to_mpc(None)

    client.complex_mpc_to_ckks(c_re, c_im)
    ct2 = server.complex_mpc_to_ckks(s_re, s_im)

    with ctx.decorypt_scope():
        dec = ct2.decorypt(copy=True, readonly=True)

    err = rel_err(dec, z)
    assert err < REL_TOL, f"Complex roundtrip error {err:.6e}"
    print(f"PASS: test_complex_roundtrip (rel_err={err:.4e})")


def test_in_process_bridge_real():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    x = np.random.randn(NSLOTS)
    ct = ctx.encorypt(x.astype(np.complex128))

    bridge = InProcessBridge(ctx)
    share = bridge.ckks_to_mpc(ct)

    assert isinstance(share, PlainShare), "Expected PlainShare"
    rec = share.reconstruct()
    err = rel_err(rec, x)
    assert err < REL_TOL, f"InProcessBridge reconstruction error {err:.6e}"
    print(f"PASS: test_in_process_bridge_real (rel_err={err:.4e})")


def test_in_process_bridge_real_roundtrip():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    x = np.random.randn(NSLOTS)
    ct = ctx.encorypt(x.astype(np.complex128))

    bridge = InProcessBridge(ctx)
    share = bridge.ckks_to_mpc(ct)
    ct2 = bridge.mpc_to_ckks(share)

    with ctx.decorypt_scope():
        dec = ct2.decorypt(copy=True, readonly=True)
    err = rel_err(dec.real, x)
    assert err < REL_TOL, f"InProcessBridge roundtrip error {err:.6e}"
    print(f"PASS: test_in_process_bridge_real_roundtrip (rel_err={err:.4e})")


def test_in_process_bridge_complex():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    z = np.random.randn(NSLOTS) + 1j * np.random.randn(NSLOTS)
    ct = ctx.encorypt(z)

    bridge = InProcessBridge(ctx)
    sh_re, sh_im = bridge.complex_ckks_to_mpc(ct)

    assert isinstance(sh_re, PlainShare)
    assert isinstance(sh_im, PlainShare)
    rec = sh_re.reconstruct() + 1j * sh_im.reconstruct()
    err = rel_err(rec, z)
    assert err < REL_TOL, f"Complex reconstruction error {err:.6e}"
    print(f"PASS: test_in_process_bridge_complex (rel_err={err:.4e})")


def test_in_process_bridge_complex_roundtrip():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    z = np.random.randn(NSLOTS) + 1j * np.random.randn(NSLOTS)
    ct = ctx.encorypt(z)

    bridge = InProcessBridge(ctx)
    sh_re, sh_im = bridge.complex_ckks_to_mpc(ct)
    ct2 = bridge.complex_mpc_to_ckks(sh_re, sh_im)

    with ctx.decorypt_scope():
        dec = ct2.decorypt(copy=True, readonly=True)
    err = rel_err(dec, z)
    assert err < REL_TOL, f"Complex roundtrip error {err:.6e}"
    print(f"PASS: test_in_process_bridge_complex_roundtrip (rel_err={err:.4e})")


def test_comm_stats_c2m():

    ctx = CKKSContext(NSLOTS)
    ct = ctx.encorypt(np.zeros(NSLOTS, dtype=np.complex128))
    stats = CommStats()

    bridge = InProcessBridge(ctx)
    bridge.ckks_to_mpc(ct, comm_stats=stats)

    assert stats.ckks_to_mpc_cts == 1, f"Expected 1, got {stats.ckks_to_mpc_cts}"
    print("PASS: test_comm_stats_c2m")


def test_comm_stats_m2c():

    ctx = CKKSContext(NSLOTS)
    stats = CommStats()
    x = np.random.randn(NSLOTS)

    bridge = InProcessBridge(ctx)
    bridge.mpc_to_ckks(x, comm_stats=stats)

    assert stats.mpc_to_ckks_cts == 1, f"Expected 1, got {stats.mpc_to_ckks_cts}"
    print("PASS: test_comm_stats_m2c")


def test_socket_channel_roundtrip():

    np.random.seed(SEED)
    ctx = CKKSContext(NSLOTS)
    x = np.random.randn(NSLOTS)
    ct = ctx.encorypt(x.astype(np.complex128))

    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind(("127.0.0.1", 0))
    port = srv_sock.getsockname()[1]
    srv_sock.listen(1)

    result_holder = {}

    def server_thread():
        conn, _ = srv_sock.accept()
        srv_sock.close()
        ch = SocketChannel(conn)
        bridge = SecureBridge(Role.SERVER, ch, ctx)
        s_share = bridge.ckks_to_mpc(ct)

        ct2 = bridge.mpc_to_ckks(s_share)
        with ctx.decorypt_scope():
            dec = ct2.decorypt(copy=True, readonly=True)
        result_holder["result"] = dec.real
        ch.close()

    def client_thread():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        ch = SocketChannel(sock)
        bridge = SecureBridge(Role.CLIENT, ch)
        c_share = bridge.ckks_to_mpc(None)

        bridge.mpc_to_ckks(c_share)
        ch.close()

    t_srv = threading.Thread(target=server_thread)
    t_cli = threading.Thread(target=client_thread)
    t_srv.start()
    t_cli.start()
    t_srv.join(timeout=10)
    t_cli.join(timeout=10)

    assert "result" in result_holder, "Server thread did not produce result"
    err = rel_err(result_holder["result"], x)
    assert err < REL_TOL, f"Socket roundtrip error {err:.6e}"
    print(f"PASS: test_socket_channel_roundtrip (rel_err={err:.4e})")


def test_run_with_weights_secure_bridge():

    os.environ["ENCFORMER_MODEL"] = "bert-base"
    np.random.seed(SEED)

    from src.encformer import _make_ckks_context, run_with_weights
    from src.engines.mpc_engine_factory import get_mpc_engine

    m, d = 128, 768
    A = np.random.randn(m, d).astype(np.float64)
    weights = {
        "WQ": np.random.randn(d, d).astype(np.float64),
        "WK": np.random.randn(d, d).astype(np.float64),
        "WV": np.random.randn(d, d).astype(np.float64),
        "bQ": np.random.randn(d).astype(np.float64),
        "bK": np.random.randn(d).astype(np.float64),
        "bV": np.random.randn(d).astype(np.float64),
        "WO": np.random.randn(d, d).astype(np.float64),
        "bO": np.random.randn(d).astype(np.float64),
        "ln1_w": np.random.randn(d).astype(np.float64),
        "ln1_b": np.random.randn(d).astype(np.float64),
        "W1": np.random.randn(d, 4 * d).astype(np.float64),
        "b1": np.random.randn(4 * d).astype(np.float64),
        "W2": np.random.randn(4 * d, d).astype(np.float64),
        "b2": np.random.randn(d).astype(np.float64),
        "ln2_w": np.random.randn(d).astype(np.float64),
        "ln2_b": np.random.randn(d).astype(np.float64),
    }

    ctx = _make_ckks_context("plain", 16384)
    mpc = get_mpc_engine("plain")

    np.random.seed(100)
    out_emu = run_with_weights(
        weights=weights,
        input_embeds=A,
        ctx=ctx,
        mpc_backend=mpc,
        verbose=False,
        secure_bridge=False,
    )

    ctx2 = _make_ckks_context("plain", 16384)
    mpc2 = get_mpc_engine("plain")
    np.random.seed(100)
    out_sec = run_with_weights(
        weights=weights,
        input_embeds=A,
        ctx=ctx2,
        mpc_backend=mpc2,
        verbose=False,
        bridge=InProcessBridge(ctx2, seed=77),
    )

    err = rel_err(out_sec, out_emu)

    assert err < 0.10, f"Secure bridge output diverged: rel_err={err:.4e}"
    print(f"PASS: test_run_with_weights_secure_bridge (rel_err={err:.4e})")


def test_two_party_bridge_over_sockets():

    from src.bridges.two_party_bridge import TwoPartyServerBridge, client_bridge_loop

    nslots = 4096
    ctx = CKKSContext(nslots)
    x = np.random.randn(nslots)
    ct = ctx.encorypt(x.astype(np.complex128))

    result_holder: dict = {}

    def _server(port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        result_holder["server_ready"] = True
        conn, _ = srv.accept()
        srv.close()
        ch = SocketChannel(conn)
        bridge = TwoPartyServerBridge(ctx, ch, seed=42)

        sh_re, sh_im = bridge.complex_ckks_to_mpc(ct)
        v_re = sh_re.reconstruct()
        v_im = sh_im.reconstruct()
        ct2 = bridge.complex_mpc_to_ckks(v_re, v_im)
        with ctx.decorypt_scope():
            dec = ct2.decorypt(copy=True, readonly=True).real[:nslots]
        result_holder["roundtrip"] = dec

        ct_real = ctx.encorypt(x.astype(np.complex128))
        sh = bridge.ckks_to_mpc(ct_real)
        v = sh.reconstruct()
        ct3 = bridge.mpc_to_ckks(v)
        with ctx.decorypt_scope():
            dec2 = ct3.decorypt(copy=True, readonly=True).real[:nslots]
        result_holder["roundtrip_real"] = dec2

        bridge.finish()
        ch.close()

    def _client(port):
        import time as _t

        _t.sleep(0.05)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(30):
            try:
                sock.connect(("127.0.0.1", port))
                break
            except ConnectionRefusedError:
                _t.sleep(0.05)
        ch = SocketChannel(sock)
        stats = client_bridge_loop(ch)
        result_holder["client_stats"] = stats
        ch.close()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    t_srv = threading.Thread(target=_server, args=(port,))
    t_cli = threading.Thread(target=_client, args=(port,))
    t_srv.start()
    t_cli.start()
    t_srv.join(timeout=30)
    t_cli.join(timeout=10)

    assert "roundtrip" in result_holder, "Server thread failed"
    err = rel_err(result_holder["roundtrip"], x)
    assert err < REL_TOL, f"Complex roundtrip error {err:.6e}"

    err_real = rel_err(result_holder["roundtrip_real"], x)
    assert err_real < REL_TOL, f"Real roundtrip error {err_real:.6e}"

    stats = result_holder["client_stats"]
    assert stats["complex_c2m"] == 1
    assert stats["complex_m2c"] == 1
    assert stats["real_c2m"] == 1
    assert stats["real_m2c"] == 1
    print(f"PASS: test_two_party_bridge_over_sockets (complex={err:.4e}, real={err_real:.4e})")


def test_run_with_weights_two_party_bridge():

    os.environ["ENCFORMER_MODEL"] = "bert-base"
    np.random.seed(SEED)

    from src.bridges.two_party_bridge import TwoPartyServerBridge, client_bridge_loop
    from src.encformer import _make_ckks_context, run_with_weights
    from src.engines.mpc_engine_factory import get_mpc_engine

    m, d = 128, 768
    A = np.random.randn(m, d).astype(np.float64)
    weights = {
        "WQ": np.random.randn(d, d).astype(np.float64),
        "WK": np.random.randn(d, d).astype(np.float64),
        "WV": np.random.randn(d, d).astype(np.float64),
        "bQ": np.random.randn(d).astype(np.float64),
        "bK": np.random.randn(d).astype(np.float64),
        "bV": np.random.randn(d).astype(np.float64),
        "WO": np.random.randn(d, d).astype(np.float64),
        "bO": np.random.randn(d).astype(np.float64),
        "ln1_w": np.random.randn(d).astype(np.float64),
        "ln1_b": np.random.randn(d).astype(np.float64),
        "W1": np.random.randn(d, 4 * d).astype(np.float64),
        "b1": np.random.randn(4 * d).astype(np.float64),
        "W2": np.random.randn(4 * d, d).astype(np.float64),
        "b2": np.random.randn(d).astype(np.float64),
        "ln2_w": np.random.randn(d).astype(np.float64),
        "ln2_b": np.random.randn(d).astype(np.float64),
    }

    result_holder: dict = {}

    def _server(port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        result_holder["server_ready"] = True
        conn, _ = srv.accept()
        srv.close()
        ch = SocketChannel(conn)

        ctx = _make_ckks_context("plain", 16384)
        mpc = get_mpc_engine("plain")
        bridge = TwoPartyServerBridge(ctx, ch, seed=99)

        np.random.seed(100)
        out = run_with_weights(
            weights=weights,
            input_embeds=A,
            ctx=ctx,
            mpc_backend=mpc,
            verbose=False,
            bridge=bridge,
        )
        bridge.finish()
        result_holder["output"] = out
        ch.close()

    def _client(port):
        import time as _t

        _t.sleep(0.05)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(30):
            try:
                sock.connect(("127.0.0.1", port))
                break
            except ConnectionRefusedError:
                _t.sleep(0.05)
        ch = SocketChannel(sock)
        stats = client_bridge_loop(ch)
        result_holder["client_stats"] = stats
        ch.close()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    ctx_ref = _make_ckks_context("plain", 16384)
    mpc_ref = get_mpc_engine("plain")
    np.random.seed(100)
    out_ref = run_with_weights(
        weights=weights,
        input_embeds=A,
        ctx=ctx_ref,
        mpc_backend=mpc_ref,
        verbose=False,
    )

    t_srv = threading.Thread(target=_server, args=(port,))
    t_cli = threading.Thread(target=_client, args=(port,))
    t_srv.start()
    t_cli.start()
    t_srv.join(timeout=120)
    t_cli.join(timeout=10)

    assert "output" in result_holder, "Server thread failed"
    out_tp = result_holder["output"]
    err = rel_err(out_tp, out_ref)

    assert err < 0.10, f"Two-party bridge output diverged: rel_err={err:.4e}"

    stats = result_holder["client_stats"]
    total_ops = sum(stats.values())
    assert total_ops > 0, "Client performed no bridge operations"
    print(f"PASS: test_run_with_weights_two_party_bridge (rel_err={err:.4e}, client_ops={total_ops})")


ALL_TESTS = [
    test_in_process_channel_basic,
    test_in_process_channel_tagged,
    test_in_process_channel_bidirectional,
    test_real_shares_sum_to_original,
    test_real_server_share_hides_value,
    test_real_client_share_hides_value,
    test_real_roundtrip,
    test_complex_shares_sum_to_original,
    test_complex_roundtrip,
    test_in_process_bridge_real,
    test_in_process_bridge_real_roundtrip,
    test_in_process_bridge_complex,
    test_in_process_bridge_complex_roundtrip,
    test_comm_stats_c2m,
    test_comm_stats_m2c,
    test_socket_channel_roundtrip,
    test_run_with_weights_secure_bridge,
    test_two_party_bridge_over_sockets,
    test_run_with_weights_two_party_bridge,
]


def test_repeated_mpc_ckks_roundtrips_no_drift():

    ctx = CKKSContext(16384)
    bridge = InProcessBridge(ctx, seed=99)

    np.random.seed(42)
    x = np.random.randn(128, 6).astype(np.float64)

    ct = ctx.encorypt(x)
    errors = []

    for trip in range(5):
        shares = bridge.ckks_to_mpc(ct)

        ct = bridge.mpc_to_ckks(shares)

        with ctx.decorypt_scope():
            raw = ct.decorypt(copy=True, readonly=True).real
            dec = raw[: x.size].reshape(x.shape)
        err = np.max(np.abs(dec - x)) / (np.max(np.abs(x)) + 1e-10)
        errors.append(err)

    assert errors[-1] < 0.01, f"Error drifted after 5 roundtrips: {errors}"

    assert errors[-1] < errors[0] * 5, f"Error grew too fast: {errors}"
    print(f"PASS: test_repeated_mpc_ckks_roundtrips_no_drift (errors={[f'{e:.2e}' for e in errors]})")


if __name__ == "__main__":
    passed = 0
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
