#!/usr/bin/env python3
import os
import sys

import numpy as np

ROOT = os.environ.get("ENCFORMER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["EZPC_PYTHONPATH"] = f"{ROOT}/third_party/ezpc-sci/build"
os.environ["MPC_BATCH_METHOD"] = "on"
os.environ["MPC_BATCH_INFERENCE"] = "1"
os.environ["MPC_EZPC_MODE"] = "native"
os.environ["MPC_EZPC_ROLE"] = "client"
sys.path.insert(0, ROOT)
from src.engines.mpc_engine_ezpc import EzPCMpcEngine
from src.models.model_config import get_config

port = int(sys.argv[1]) if len(sys.argv) > 1 else 36600
NL = int(os.environ.get("NLAYERS", "12"))
cfg = get_config("bert-base")
m, d, dff, H = cfg.m, cfg.d_model, cfg.d_ff, cfg.H

eng = EzPCMpcEngine(mode="native", role="client", port=port)
zc = np.zeros((m, m))
zd = np.zeros((m, d))
zf = np.zeros((m, dff))
zg = np.zeros(d)
print(f"[client] connected on port {port}, mirroring {NL} layer(s)", flush=True)
for li in range(NL):
    for h in range(H):
        eng.softmax_rows(zc, head_index=h)
    eng.layer_norm(zd, eps=1e-5, gamma=zg, beta=zg, ln_tag="ln1")
    eng.gelu(zf)
    eng.layer_norm(zd, eps=1e-5, gamma=zg, beta=zg, ln_tag="ln2")
    print(f"  [client] layer {li + 1}/{NL} mirrored", flush=True)
print("[client] done", flush=True)
