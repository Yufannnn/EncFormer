from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

F_BITS = 13
_ONE = float(1 << F_BITS)
ROOT = Path(__file__).resolve().parents[3]


def to_fixed(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x, dtype=np.float64) * _ONE).astype(np.int64).view(np.uint64)


def from_fixed(s: np.ndarray) -> np.ndarray:
    return np.asarray(s, dtype=np.uint64).view(np.int64).astype(np.float64) / _ONE


@dataclass(frozen=True)
class Geom:
    m: int = 128
    d: int = 768
    h: int = 12
    d_ff: int = 3072
    nslots: int = 16384

    @property
    def c(self) -> int:
        return self.nslots // self.m

    @property
    def d_h(self) -> int:
        return self.d // self.h


def pack_pairs(x: np.ndarray, g: Geom, c_in: int) -> np.ndarray:
    m, d = x.shape
    hp = d // (2 * c_in)
    out = np.zeros((hp, 2, g.nslots), dtype=np.uint64)
    for h in range(hp):
        out[h, 0, : c_in * m] = x[:, (2 * h) * c_in:(2 * h + 1) * c_in].T.reshape(-1)
        out[h, 1, : c_in * m] = x[:, (2 * h + 1) * c_in:(2 * h + 2) * c_in].T.reshape(-1)
    return out


def unpack_blocks(s: np.ndarray, g: Geom, d_out: int, c_used: int) -> np.ndarray:
    y = np.zeros((g.m, d_out), dtype=np.uint64)
    for b in range(s.shape[0]):
        used = min(c_used, d_out - b * c_used)
        if used <= 0:
            break
        y[:, b * c_used:b * c_used + used] = s[b, 0, : used * g.m].reshape(used, g.m).T
    return y


def split_paired(s: np.ndarray) -> np.ndarray:
    out = np.zeros((2 * s.shape[0], 2, s.shape[2]), dtype=np.uint64)
    out[0::2, 0] = s[:, 0]
    out[1::2, 0] = s[:, 1]
    return out


def unpack_scores(s: np.ndarray, g: Geom, index: np.ndarray) -> np.ndarray:
    out = np.zeros((g.h, g.m, g.m), dtype=np.uint64)
    ct, slot, h, row, cre, cim = (index[:, k] for k in range(6))
    out[h, row, cre] = s[ct, 0, slot]
    out[h, row, cim] = s[ct, 1, slot]
    return out


def pack_pfd(a: np.ndarray, g: Geom) -> np.ndarray:
    m, half = g.m, g.m // 2
    hpc = g.c // g.d_h
    nb = g.h // hpc
    out = np.zeros((nb, 2, g.nslots), dtype=np.uint64)
    r = np.arange(m)
    for b in range(nb):
        for hl in range(hpc):
            ah = a[b * hpc + hl]
            for t in range(half):
                seg = (hl * half + t) * m
                out[b, 0, seg:seg + m] = ah[r, (r - t) % m]
                out[b, 1, seg:seg + m] = ah[r, (r - t - half) % m]
    return out


@dataclass
class Binary:
    exe: str
    tp_dir: Path
    side: str
    gpu: str = "0"
    env: Dict[str, str] = field(default_factory=dict)
    proc: subprocess.Popen | None = None
    k: int = 0

    def start(self) -> None:
        env = dict(os.environ, TP_DIR=str(self.tp_dir), CUDA_VISIBLE_DEVICES=self.gpu, **self.env)
        self.out = open(self.tp_dir / self.side / "stdout.log", "w")
        self.proc = subprocess.Popen([self.exe], env=env, stdout=self.out, stderr=subprocess.STDOUT)

    def wait_ready(self, name: str, timeout: float = 900.0) -> Dict[str, Any]:
        return _parse_kv(_wait(self.tp_dir / self.side / name, self.proc, timeout).read_text())

    def request(self, line: str, timeout: float = 900.0) -> Dict[str, Any]:
        req = self.tp_dir / self.side / f"req_{self.k}"
        resp = self.tp_dir / self.side / f"resp_{self.k}"
        tmp = req.with_suffix(".tmp")
        tmp.write_text(line + "\n")
        tmp.rename(req)
        kv = _parse_kv(_wait(resp, self.proc, timeout).read_text())
        resp.unlink()
        self.k += 1
        return kv

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.request("shutdown", timeout=30)
            except Exception:  # noqa: BLE001
                pass
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()


def _parse_kv(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) == 2:
            try:
                out[p[0]] = float(p[1])
            except ValueError:
                out[p[0]] = p[1]
    return out


def _wait(path: Path, proc: subprocess.Popen | None, timeout: float) -> Path:
    t0 = time.monotonic()
    abort = path.parent.parent / "ABORT"
    while not path.exists():
        if abort.exists():
            raise RuntimeError(f"other party aborted:\n{abort.read_text()[-2000:]}")
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{proc.args[0]} exited (rc={proc.returncode}) while waiting for {path.name}")
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.0005)
    return path


def _write_u64(path: Path, a: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.ascontiguousarray(a, dtype=np.uint64).tofile(tmp)
    tmp.rename(path)


def _read_u64(path: Path, shape) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint64).reshape(shape)


@dataclass
class LayerPublic:
    bO: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    ln1_w: np.ndarray
    ln1_b: np.ndarray
    ln2_w: np.ndarray
    ln2_b: np.ndarray
    denoms: dict


def party_main(role: str, conn, *, tp_dir: str, exe: str, layers: List[Dict[str, np.ndarray]] | None,
               pub: List[LayerPublic], x0: np.ndarray | None, mask: np.ndarray, g: Geom, gpu: str,
               result_path: str, c0: int = 2, cf1: int = 3, cf2: int = 6) -> None:
    from src.fhe.phantom.two_party_sci import MASK_VALUE, SciShares, bert_base_plan

    tp = Path(tp_dir)
    side = "srv" if role == "server" else "cli"
    wire, own = tp / "wire", tp / side
    bd = {"x": 7, "pfd": 7, "ff1": 7, "ff2": 7, "score": 7, "z": 7, "h1": 7, "h2": 7, "f": 15}
    binr = Binary(exe, tp, side, gpu, env={"TP_C0": str(c0), "TP_CF1": str(cf1), "TP_CF2": str(cf2),
                                           **{f"TP_BOUND_{k.upper()}": str(bd[k]) for k in ("score", "z", "h1", "h2", "f")}})
    binr.start()
    try:
        mpc = SciShares(role, conn)
        ready = binr.wait_ready("server_ready" if role == "server" else "client_ready")
        if str(ready.get("conversion")) != "c2m_m2c_rerandomized":
            raise RuntimeError("unexpected conversion backend")
        if role == "server":
            index = np.fromfile(own / "score_index.i32", dtype=np.int32).reshape(-1, 6)
            conn.send(index)
        else:
            index = conn.recv()
        nslots = g.nslots
        mask_add = np.where(np.asarray(mask).reshape(-1) > 0, 0.0, MASK_VALUE)

        def m2c(name: str, slot_share: np.ndarray, chain: int) -> None:
            n = slot_share.shape[0]
            if role == "server":
                _write_u64(own / f"{name}_s1.u64", slot_share)
                binr.request(f"m2c_mask {name} {n} {bd[name]}")
            else:
                _write_u64(own / f"{name}_s0.u64", slot_share)
                _wait(wire / f"{name}_mask.u64", None, 900.0)
                binr.request(f"m2c {name} {n} {chain}")
                (wire / f"{name}_mask.u64").unlink()

        def c2m(name: str, ncts: int) -> np.ndarray:
            if role == "server":
                return _read_u64(own / f"{name}_s1.u64", (ncts, 2, nslots))
            _wait(wire / f"{name}_c2m.bin", None, 900.0)
            binr.request(f"c2m {name}")
            (wire / f"{name}_c2m.bin").unlink()
            return _read_u64(own / f"{name}_s0.u64", (ncts, 2, nslots))

        def server_stage(cmd: str, consumed: str) -> None:
            _wait(wire / consumed, None, 900.0)
            binr.request(cmd)
            (wire / consumed).unlink()

        lp = [{"denoms": P_.denoms, "ln1_w": P_.ln1_w, "ln1_b": P_.ln1_b, "ln2_w": P_.ln2_w, "ln2_b": P_.ln2_b}
              for P_ in pub]
        mpc.offline(bert_base_plan(len(pub), lp, m=g.m, d=g.d, h=g.h, d_ff=g.d_ff))

        x_sh = to_fixed(x0) if role == "client" else np.zeros((g.m, g.d), dtype=np.uint64)
        n_score = int(index[:, 0].max()) + 1
        layer_s = []
        for li, P in enumerate(pub):
            mpc.set_running_denominators(P.denoms)
            if role == "server":
                wpath = own / "layer.bin"
                W = layers[li]
                rs, cc = mpc.prescale_rows(g.h, g.m)
                np.concatenate([W[k].reshape(-1) for k in ("WQ", "WK", "WV", "bQ", "bK", "bV", "WO", "W1", "W2", "b1")]
                               + [rs.reshape(-1), np.array([cc, 1.0])]).astype(np.float64).tofile(wpath)
                binr.request(f"load_layer {wpath}")
                conn.send("go")
                conn.recv()
            else:
                conn.recv()
                conn.send("go")
            t_layer = time.perf_counter()

            m2c("x", pack_pairs(x_sh, g, g.c), c0)
            if role == "server":
                server_stage("qkv_score", "x_ct.bin")
            s = unpack_scores(c2m("score", n_score), g, index)
            if role == "server":
                s = s + to_fixed(np.broadcast_to(mask_add, (g.h, g.m, g.m)))
            a = mpc.softmax(s)

            m2c("pfd", pack_pfd(a, g), c0 + 2)
            if role == "server":
                server_stage("value_out", "pfd_ct.bin")
            z = unpack_blocks(split_paired(c2m("z", g.d // g.c // 2)), g, g.d, g.c) + x_sh
            if role == "server":
                z = z + to_fixed(np.broadcast_to(P.bO, (g.m, g.d)))
            z_ln = mpc.layer_norm(z, P.ln1_w, P.ln1_b, "ln1")

            m2c("ff1", pack_pairs(z_ln, g, g.c), cf1)
            if role == "server":
                server_stage("ff1", "ff1_ct.bin")
            nb = g.d_ff // g.c
            h1 = unpack_blocks(split_paired(c2m("h1x", nb // 2)), g, g.d_ff, g.c)
            f01 = c2m("h1f", nb)
            f0 = unpack_blocks(f01, g, g.d_ff, g.c)
            f1 = unpack_blocks(f01[:, ::-1], g, g.d_ff, g.c)
            gl = mpc.gelu_select(h1, f0, f1)
            m2c("ff2", pack_pairs(gl, g, g.c), cf2)
            if role == "server":
                server_stage("ff2", "ff2_ct.bin")
            h2 = unpack_blocks(split_paired(c2m("h2", g.d // g.c // 2)), g, g.d, g.c) + z_ln
            if role == "server":
                h2 = h2 + to_fixed(np.broadcast_to(P.b2, (g.m, g.d)))
            x_sh = mpc.layer_norm(h2, P.ln2_w, P.ln2_b, "ln2")
            layer_s.append(time.perf_counter() - t_layer)
            if role == "client":
                print(f"  layer {li + 1:2d}/{len(pub)}  {layer_s[-1]:5.2f} s", flush=True)

        if role == "server":
            conn.send(x_sh)
        else:
            np.save(result_path + ".hidden.npy", from_fixed(x_sh + conn.recv()))
            np.save(result_path + ".layer_s.npy", np.array(layer_s))
    except BaseException:
        import traceback
        (tp / "ABORT").write_text(f"{role}: " + traceback.format_exc())
        raise
    finally:
        binr.stop()


def run(layers: List[Dict[str, np.ndarray]], pub: List[LayerPublic], x0: np.ndarray, mask: np.ndarray, g: Geom,
        gpu: str = "0", bin_dir: str | None = None, tp_dir: str | None = None) -> tuple[np.ndarray, list]:
    bin_dir = bin_dir or str(ROOT / "third_party/phantom-fhe/build/bin")
    tp = Path(tp_dir or tempfile.mkdtemp(prefix="encformer_"))
    if tp.exists():
        shutil.rmtree(tp)
    for sub in ("wire", "srv", "cli"):
        (tp / sub).mkdir(parents=True)
    result = str(tp / "result")
    common = dict(tp_dir=str(tp), pub=pub, mask=mask, g=g, gpu=gpu, result_path=result)
    srv_kw = dict(common, exe=os.path.join(bin_dir, "pipe_tp_server_bert_base"), layers=layers, x0=None)
    cli_kw = dict(common, exe=os.path.join(bin_dir, "pipe_tp_client_bert_base"), layers=None, x0=x0)
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    ps = [ctx.Process(target=party_main, args=("server", a), kwargs=srv_kw),
          ctx.Process(target=party_main, args=("client", b), kwargs=cli_kw)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    if any(p.exitcode != 0 for p in ps):
        raise RuntimeError(f"party failed (exit codes {[p.exitcode for p in ps]}); logs in {tp}")
    out = np.load(result + ".hidden.npy")
    layer_s = np.load(result + ".layer_s.npy").tolist()
    shutil.rmtree(tp, ignore_errors=True)
    return out, layer_s
