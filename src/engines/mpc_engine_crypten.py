from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from typing import Optional

import numpy as np

from src.engines.mpc_batch_method import BatchMethodState, load_batch_method_config
from src.engines.mpc_gelu_secure import (
    build_quantized_real_algorithm4_constants,
    load_secure_gelu_config,
    quantize_real_array,
    secure_gelu_algorithm4_split,
)

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Optional[Exception] = exc
else:
    _TORCH_IMPORT_ERROR = None


def _install_crypten_torch_onnx_compat() -> None:

    if torch is None:
        return
    try:
        importlib.import_module("torch.onnx._internal.registration")
        return
    except Exception:
        pass

    try:
        importlib.import_module("torch.onnx._internal")
    except Exception:
        internal_name = "torch.onnx._internal"
        internal_mod = sys.modules.get(internal_name)
        if internal_mod is None:
            internal_mod = types.ModuleType(internal_name)
            internal_mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[internal_name] = internal_mod
        torch_onnx = getattr(torch, "onnx", None)
        if torch_onnx is not None:
            setattr(torch_onnx, "_internal", internal_mod)

    reg_name = "torch.onnx._internal.registration"
    if reg_name not in sys.modules:
        reg_mod = types.ModuleType(reg_name)
        reg_mod.registry = {}
        sys.modules[reg_name] = reg_mod
    internal_mod = sys.modules.get("torch.onnx._internal")
    reg_mod = sys.modules.get(reg_name)
    if internal_mod is not None and reg_mod is not None:
        setattr(internal_mod, "registration", reg_mod)


_install_crypten_torch_onnx_compat()

try:
    import crypten
except Exception as exc:  # pragma: no cover
    crypten = None  # type: ignore[assignment]
    _CRYPTEN_IMPORT_ERROR: Optional[Exception] = exc
else:
    _CRYPTEN_IMPORT_ERROR = None


class _CrypTenSecureGeluOps:
    def ewmulcc(self, u, v):
        return u * v

    def ewmulcp(self, u, c):
        return u * float(c)

    def cmp(self, a, b):
        if hasattr(a, "lt"):
            return a.lt(b)
        if hasattr(b, "gt"):
            return b.gt(float(a))
        raise TypeError("cmp expects at least one CrypTen secret operand.")

    def mux(self, val, bit):
        return val * bit

    def add(self, a, b):
        return a + b

    def addc(self, a, c):
        return a + float(c)

    def neg(self, a):
        return -a

    def xor_bits(self, a, b):
        return a + b - 2 * a * b


class CrypTenMpcEngine:
    name = "crypten"

    def __init__(self, *, device: str = "auto"):
        if torch is None:
            raise ImportError(f"torch is required for CrypTenMpcEngine: {_TORCH_IMPORT_ERROR}")
        if crypten is None:
            raise ImportError(f"crypten is required for CrypTenMpcEngine: {_CRYPTEN_IMPORT_ERROR}")
        self._device = self._resolve_device(device)
        self._batch_cfg = load_batch_method_config()
        self._batch_state = BatchMethodState(self._batch_cfg)
        self.comm_stats = None
        self._init_crypten()

    def set_running_denominators(self, denoms: dict) -> None:

        for key, val in denoms.items():
            if val is not None:
                self._batch_state.buffers[key] = val
            else:
                self._batch_state.buffers.pop(key, None)

    @property
    def device(self) -> str:
        return str(self._device)

    def _resolve_device(self, device: str) -> "torch.device":
        mode = device.lower().strip()
        if mode in {"", "auto"}:
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        want_cuda = mode in {"cuda", "gpu"}
        allow_unsafe_windows_cuda = os.environ.get("CRYPTEN_WINDOWS_CUDA_UNSAFE", "0") == "1"
        if want_cuda and os.name == "nt" and not allow_unsafe_windows_cuda:
            raise RuntimeError(
                "CrypTen CUDA on native Windows is unstable in this setup (stack overflow). "
                "Use WSL/Linux for GPU, or set MPC_DEVICE=cpu. "
                "To force native Windows CUDA anyway, set CRYPTEN_WINDOWS_CUDA_UNSAFE=1."
            )
        if want_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if want_cuda:
            raise RuntimeError("MPC_DEVICE=cuda requested but CUDA is not available.")
        return torch.device("cpu")

    def _init_crypten(self) -> None:
        is_initialized = getattr(crypten, "is_initialized", None)
        if callable(is_initialized) and bool(is_initialized()):
            return
        init_thread = getattr(crypten, "init_thread", None)

        if os.name == "nt" and callable(init_thread):
            init_thread(0, 1)
            self._patch_inprocess_compat()
            return
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        try:
            crypten.init()
        except Exception as exc:
            if callable(init_thread) and "Multiprocessing is not supported on Windows" in str(exc):
                init_thread(0, 1)
                self._patch_inprocess_compat()
                return
            raise

    def _patch_inprocess_compat(self) -> None:
        comm = crypten.communicator.get()
        all_reduce = getattr(comm.__class__, "all_reduce", None)
        if all_reduce is None:
            return
        sig = inspect.signature(all_reduce)
        if "batched" in sig.parameters:
            return
        op_default = sig.parameters["op"].default

        def _all_reduce_with_batched(self_comm, tensor, op=op_default, async_op=False, batched=False):
            if batched:
                if int(self_comm.get_world_size()) != 1:
                    raise RuntimeError("CrypTen Windows thread mode batched all_reduce supports only world_size=1.")
                return tensor
            return all_reduce(self_comm, tensor, op=op, async_op=async_op)

        setattr(comm.__class__, "all_reduce", _all_reduce_with_batched)

    def _to_crypten(self, x: np.ndarray) -> "crypten.CrypTensor":
        t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self._device)
        return crypten.cryptensor(t)

    def _to_numpy(self, x_enc: "crypten.CrypTensor") -> np.ndarray:
        pt = x_enc.get_plain_text()
        if pt.device.type != "cpu":
            pt = pt.cpu()
        return pt.numpy().astype(np.float64, copy=False)

    def _as_public_tensor(self, x: np.ndarray):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self._device)

    def _record_rounds(self, op: str, n: int) -> None:
        if self.comm_stats is not None:
            self.comm_stats.add_mpc_rounds(op, n)

    def _pow_secure_int(self, base, p: int):
        if p <= 0:
            return base * 0 + 1.0
        out = base
        for _ in range(1, p):
            out = out * base
        return out

    def softmax_rows(self, x: np.ndarray, *, head_index: int | None = None) -> np.ndarray:
        if self._batch_cfg.enabled and self._batch_cfg.enable_bpmax:
            x_arr = np.asarray(x, dtype=np.float64)
            p = int(self._batch_cfg.p)

            z_arr = np.maximum(x_arr + float(self._batch_cfg.c), 0.0)

            row_max = np.max(np.abs(z_arr), axis=1, keepdims=True)
            target = float(os.environ.get("MPC_CRYPTEN_BP_TARGET_ABS", "2.0"))
            row_scale = np.maximum(1.0, row_max / max(target, 1.0))
            z_scaled = z_arr / row_scale
            z_enc = self._to_crypten(z_scaled)
            pw = self._pow_secure_int(z_enc, p)
            den_cur_scaled = np.power(z_scaled, p).sum(axis=1, keepdims=True)
            if self._batch_cfg.inference:
                den = self._batch_state.get_bpmax_den(x_arr.shape[0], head_index=head_index)
                if den is not None:
                    inv = self._as_public_tensor((row_scale**p) / (den + float(self._batch_cfg.eps)))
                else:
                    inv = self._as_public_tensor(1.0 / (den_cur_scaled + float(self._batch_cfg.eps)))
            else:
                den_plain = np.power(z_arr, p).sum(axis=1, keepdims=True)
                h = head_index if head_index is not None else 0
                self._batch_state.update_bpmax_head(h, den_plain)
                inv = self._as_public_tensor(1.0 / (den_cur_scaled + float(self._batch_cfg.eps)))
            y_enc = pw * inv
            self._record_rounds("bpmax", 3)
            return self._to_numpy(y_enc)
        x_enc = self._to_crypten(x)
        y_enc = x_enc.softmax(dim=1)
        self._record_rounds("softmax", 3)
        return self._to_numpy(y_enc)

    def layer_norm(
        self,
        x: np.ndarray,
        *,
        eps: float = 1e-5,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        ln_tag: str | None = None,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        key = "ln2_running_denominator" if ln_tag == "ln2" else "ln1_running_denominator"
        if self._batch_cfg.enabled and self._batch_cfg.enable_batchln:
            row_max = np.max(np.abs(x_arr), axis=1, keepdims=True)
            target = float(os.environ.get("MPC_CRYPTEN_LN_TARGET_ABS", "32.0"))
            row_scale = np.maximum(1.0, row_max / max(target, 1.0))
            x_scaled = x_arr / row_scale
            x_enc = self._to_crypten(x_scaled)
            mean = x_enc.mean(dim=1, keepdim=True)
            centered = x_enc - mean
            den = self._batch_state.get_ln_den(x_arr.shape[0], ln_tag=ln_tag)
            if self._batch_cfg.inference and den is not None:
                den_eff = float(self._batch_cfg.ln_l) * (den / row_scale)
                inv = self._as_public_tensor(1.0 / (den_eff + float(self._batch_cfg.eps)))
                y_enc = centered * inv
            else:
                var = (centered * centered).mean(dim=1, keepdim=True)
                y_enc = centered / (
                    float(self._batch_cfg.ln_l) * (var + float(eps)).sqrt() + float(self._batch_cfg.eps)
                )
                if not self._batch_cfg.inference:
                    xc_plain = x_arr - x_arr.mean(axis=1, keepdims=True)
                    rms_plain = np.sqrt((xc_plain * xc_plain).mean(axis=1, keepdims=True) + float(eps))
                    self._batch_state.update_running(key, rms_plain)
        else:
            row_max = np.max(np.abs(x_arr), axis=1, keepdims=True)
            target = float(os.environ.get("MPC_CRYPTEN_LN_TARGET_ABS", "32.0"))
            row_scale = np.maximum(1.0, row_max / max(target, 1.0))
            x_scaled = x_arr / row_scale
            x_enc = self._to_crypten(x_scaled)
            mean = x_enc.mean(dim=1, keepdim=True)
            centered = x_enc - mean
            var = (centered * centered).mean(dim=1, keepdim=True)
            y_enc = centered / (var + float(eps)).sqrt()
        if gamma is not None:
            g_t = torch.as_tensor(np.asarray(gamma, dtype=np.float32), device=self._device)
            y_enc = y_enc * g_t
        if beta is not None:
            b_t = torch.as_tensor(np.asarray(beta, dtype=np.float32), device=self._device)
            y_enc = y_enc + b_t

        is_batch = self._batch_cfg.enabled and self._batch_cfg.enable_batchln
        self._record_rounds("batchln" if is_batch else "layernorm", 0 if is_batch else 2)
        return self._to_numpy(y_enc)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        cfg = load_secure_gelu_config()
        xq = quantize_real_array(np.asarray(x, dtype=np.float64), cfg)
        x_enc = self._to_crypten(xq)
        consts = build_quantized_real_algorithm4_constants(cfg)
        clip_lo = float(consts.neg_thresh)
        clip_hi = float(consts.pos_thresh)
        xq_poly = np.clip(xq, clip_lo, clip_hi)
        x_poly_enc = self._to_crypten(xq_poly)
        ops = _CrypTenSecureGeluOps()
        y_enc = secure_gelu_algorithm4_split(x_enc, x_poly_enc, x_enc, ops=ops, consts=consts)
        y = self._to_numpy(y_enc)
        self._record_rounds("gelu", 4)
        return quantize_real_array(y, cfg)

    def softmax_rows_shares(self, shares, *, head_index=None):
        from src.share_types import ShareMatrix

        x = shares.reconstruct()
        y = self.softmax_rows(x, head_index=head_index)
        return ShareMatrix(y, np.zeros_like(y))

    def layer_norm_shares(self, shares, *, eps=1e-5, gamma=None, beta=None, ln_tag=None):
        from src.share_types import ShareMatrix

        x = shares.reconstruct()
        try:
            y = self.layer_norm(x, eps=eps, gamma=gamma, beta=beta, ln_tag=ln_tag)
        except TypeError:
            y = self.layer_norm(x, eps=eps, gamma=gamma, beta=beta)
        return ShareMatrix(y, np.zeros_like(y))

    def gelu_shares(self, shares):
        from src.share_types import ShareMatrix

        x = shares.reconstruct()
        y = self.gelu(x)
        return ShareMatrix(y, np.zeros_like(y))
