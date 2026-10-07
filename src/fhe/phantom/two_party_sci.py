from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.mpc_batch_method import BatchMethodState, load_batch_method_config
from src.engines.mpc_gelu_secure import load_secure_gelu_config

F = 13
PRESCALE_A = 3
BPMAX_CMP_BITS = 21
GELU_CMP_BITS = 22
MASK_VALUE = -32.0


def _import_sci():
    p = os.environ.get("EZPC_PYTHONPATH", "").strip()
    if not p:
        here = os.path.dirname(os.path.abspath(__file__))
        p = os.path.normpath(os.path.join(here, "..", "..", "..", "third_party", "ezpc-sci", "build"))
    if p not in sys.path:
        sys.path.insert(0, p)
    import ezpc_sci  # type: ignore[import-untyped]

    if not ezpc_sci.HAS_NATIVE_SCI or not getattr(ezpc_sci, "HAS_OT_POOL", False):
        raise RuntimeError("ezpc_sci is not built with SCI; run scripts/build_ezpc_sci.sh")
    return ezpc_sci


def _ports_free(base: int, threads: int) -> bool:
    for off in (0, 50, 100):
        for t in range(threads):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", base + off + t))
                except OSError:
                    return False
    return True


def _pick_port(threads: int) -> int:
    try:
        low = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
    except Exception:
        low = 32768
    rng = np.random.default_rng()
    for _ in range(200):
        base = int(rng.integers(10000, min(low, 65536) - 100 - threads - 1))
        if _ports_free(base, threads):
            return base
    raise RuntimeError("no free port range found")


def _install(state: BatchMethodState, denoms: dict) -> None:
    for k, v in (denoms or {}).items():
        if v is None:
            state.buffers.pop(k, None)
        else:
            state.buffers[k] = np.asarray(v, dtype=np.float64)


def bert_base_plan(layers: int, layer_params: list, *, m: int = 128, d: int = 768, h: int = 12,
                   d_ff: int = 3072) -> list:
    plan = []
    for P in layer_params[:layers]:
        lp1 = {"denoms": P["denoms"], "gamma": P["ln1_w"], "beta": P["ln1_b"]}
        lp2 = {"denoms": P["denoms"], "gamma": P["ln2_w"], "beta": P["ln2_b"]}
        plan += [("softmax", (h, m, m)), ("layer_norm", (m, d), "ln1", lp1), ("gelu_select", m * d_ff),
                 ("layer_norm", (m, d), "ln2", lp2)]
    return plan


def _online(sci, ctx, e: dict, x=None, f0=None, f1=None):
    if e["op"] == "bpmax":
        x = np.zeros((e["rows"], e["cols"]), dtype=np.uint64) if x is None else x
        return sci.bpmax_rows_shares(ctx, x, None, c=e["c"], p=e["p"], prescaled=True, cmp_bits=BPMAX_CMP_BITS)
    if e["op"] == "ln":
        x = np.zeros((e["rows"], e["cols"]), dtype=np.uint64) if x is None else x
        return sci.layernorm_shares(ctx, x, e["rd"], gamma=e["gamma"], beta=e["beta"], eps=e["eps"],
                                    ln_l=e["ln_l"])
    if x is None:
        x = f0 = f1 = np.zeros(e["n"], dtype=np.uint64)
    return sci.gelu_select_shares(ctx, x, f0, f1, cmp_bits=GELU_CMP_BITS, threshold=e["threshold"])


def _prepare(sci, ctx, e: dict) -> None:
    if e["op"] == "bpmax":
        sci.prepare_bpmax(ctx, e["rows"], e["cols"], None, c=e["c"], p=e["p"], prescaled=True,
                          cmp_bits=BPMAX_CMP_BITS)
    elif e["op"] == "ln":
        sci.prepare_layernorm(ctx, e["rows"], e["cols"])
    else:
        sci.prepare_gelu_select(ctx, e["n"], cmp_bits=GELU_CMP_BITS, threshold=e["threshold"])


def _key(e: dict, threads: int) -> tuple:
    return (e["op"], e.get("rows"), e.get("cols"), e.get("n"), threads)


class SciShares:
    name = "sci"
    prescaled = True

    def __init__(self, role: str, conn: Any, *, address: str | None = None, threads: int | None = None):
        self.role = role
        self._sci = _import_sci()
        self._cfg = load_batch_method_config()
        self._threshold = float(load_secure_gelu_config().threshold)
        self._state = BatchMethodState(self._cfg)
        self._ot_counts: dict[tuple, list[int]] = {}
        params = {"F": F, "c": self._cfg.c, "p": self._cfg.p, "eps": self._cfg.eps, "ln_l": self._cfg.ln_l,
                  "threshold": self._threshold}
        address = address or os.environ.get("MPC_EZPC_ADDRESS", "127.0.0.1")
        if role == "server":
            threads = int(threads or os.environ.get("MPC_EZPC_THREADS", "4"))
            port = int(os.environ.get("MPC_EZPC_PORT", "0")) or _pick_port(threads)
            conn.send({"sci_port": port, "threads": threads, "params": params})
            if conn.recv() != "ok":
                raise RuntimeError("client rejected the SCI setup")
        else:
            msg = conn.recv()
            port, threads = int(msg["sci_port"]), int(msg["threads"])
            if msg["params"] != params:
                conn.send("parameter mismatch")
                raise RuntimeError(f"parameter mismatch: {msg['params']} vs {params}")
            conn.send("ok")
        self.ctx = self._sci.SCIContext(role=0 if role == "server" else 1, address=address, port=port,
                                        threads=threads)

    def set_running_denominators(self, denoms: dict) -> None:
        _install(self._state, denoms)

    def _bpmax_entry(self, H: int, m: int, n: int) -> dict:
        return {"op": "bpmax", "rows": H * m, "cols": n, "c": float(self._cfg.c), "p": int(self._cfg.p)}

    def _ln_entry(self, rows: int, cols: int, gamma, beta, tag, state) -> dict:
        den = state.get_ln_den(rows, ln_tag=tag)
        if den is None:
            raise RuntimeError(f"layer_norm: no {tag} running denominator")
        return {"op": "ln", "rows": rows, "cols": cols, "rd": np.asarray(den, dtype=np.float64).reshape(-1),
                "gamma": np.asarray(gamma, dtype=np.float64).reshape(-1),
                "beta": np.asarray(beta, dtype=np.float64).reshape(-1),
                "eps": float(self._cfg.eps), "ln_l": float(self._cfg.ln_l)}

    def _gelu_entry(self, n: int) -> dict:
        return {"op": "gelu_select", "n": int(n), "threshold": self._threshold}

    def offline(self, plan: list) -> None:
        entries = []
        for call in plan:
            if call[0] == "softmax":
                entries.append(self._bpmax_entry(*call[1]))
            elif call[0] == "layer_norm":
                lp = call[3]
                st = BatchMethodState(self._cfg)
                _install(st, lp["denoms"])
                entries.append(self._ln_entry(*call[1], lp["gamma"], lp["beta"], call[2], st))
            else:
                entries.append(self._gelu_entry(call[1]))
        if self.ctx.planned or self.ctx.plan_queue_len:
            self.ctx.end_plan()
        threads = self.ctx.threads
        for e in entries:
            k = _key(e, threads)
            if k not in self._ot_counts:
                _online(self._sci, self.ctx, e)
                self._ot_counts[k] = list(self.ctx.last_stats["online"]["ot_counts"])
        for e in entries:
            _prepare(self._sci, self.ctx, e)
        counts = np.zeros(threads * self._sci.SCIContext.OT_INSTANCES, dtype=np.uint64)
        for e in entries:
            counts += np.asarray(self._ot_counts[_key(e, threads)], dtype=np.uint64)
        self.ctx.pool_fill([int(v) for v in counts])
        self.ctx.begin_online()

    def softmax(self, s_share: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(s_share, dtype=np.uint64)
        H, m, n = x.shape
        y = _online(self._sci, self.ctx, self._bpmax_entry(H, m, n), x=x.reshape(H * m, n))
        return y.reshape(s_share.shape)

    def layer_norm(self, x_share: np.ndarray, gamma, beta, tag: str) -> np.ndarray:
        x = np.ascontiguousarray(x_share, dtype=np.uint64)
        return _online(self._sci, self.ctx, self._ln_entry(x.shape[0], x.shape[1], gamma, beta, tag, self._state), x=x)

    def gelu_select(self, x_share: np.ndarray, f0_share: np.ndarray, f1_share: np.ndarray) -> np.ndarray:
        u = lambda a: np.ascontiguousarray(a, dtype=np.uint64).reshape(-1)  # noqa: E731
        y = _online(self._sci, self.ctx, self._gelu_entry(np.size(x_share)), x=u(x_share), f0=u(f0_share),
                    f1=u(f1_share))
        return y.reshape(np.shape(x_share))

    def prescale_rows(self, H: int, m: int) -> tuple[np.ndarray, float]:
        s = np.zeros((H, m))
        for h in range(H):
            den = self._state.get_bpmax_den(m, head_index=h)
            if den is None:
                raise RuntimeError("no bpmax running denominator")
            s[h] = np.power(np.asarray(den, dtype=np.float64).reshape(-1) + float(self._cfg.eps),
                            -1.0 / int(self._cfg.p)) * 2.0 ** PRESCALE_A
        return s, float(self._cfg.c)
