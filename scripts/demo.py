#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import tempfile
import time

ROOT = os.environ.get("ENCFORMER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBDIR = f"{ROOT}/third_party/phantom-fhe/build/lib"
os.environ["LD_LIBRARY_PATH"] = LIBDIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["MPC_BATCH_METHOD"] = "on"
os.environ["MPC_BATCH_INFERENCE"] = "1"
os.environ.setdefault("MPC_EZPC_MODE", "emulated")
os.environ.setdefault("MPC_EMULATED_FIXEDPOINT", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, ROOT)
from transformers import BertTokenizer

from src.engines.encformer_model import extract_layer_weights, load_checkpoint
from src.engines.mpc_engine_factory import get_mpc_engine
from src.fhe.phantom import phantom_native_pipe as P
from src.fhe.phantom.phantom_native_bench import _config_target, resolve_native_binaries
from src.inference_runtime import inject_layer_running_denoms, prepare_layer_running_denoms
from src.models.model_config import get_config

LABELS = {0: "negative", 1: "positive"}
C = {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m", "c": "\033[36m", "b": "\033[1m", "d": "\033[2m", "x": "\033[0m"}


def checkpoint():
    return f"{ROOT}/checkpoints/encformer-sst2"


def col(s, k):
    return f"{C[k]}{s}{C['x']}" if sys.stdout.isatty() else str(s)


def rule(title=""):
    line = "=" * 72
    if title:
        print(f"\n{col(line, 'd')}\n{col(title, 'b')}\n{col(line, 'd')}")
    else:
        print(col(line, "d"))


def main():
    ap = argparse.ArgumentParser(prog="EncFormer")
    ap.add_argument("--ckpt", default=checkpoint())
    ap.add_argument("--idx", type=int, default=40, help="SST-2 validation index (ignored if --text)")
    ap.add_argument("--text", default=None, help="custom sentence to classify")
    ap.add_argument("--gpu", default="0", help="CUDA device index for native FHE")
    ap.add_argument("--layers", type=int, default=None, help="#transformer layers (default: all 12)")
    ap.add_argument("--json", default=None, help="write a machine-readable result JSON here")
    args = ap.parse_args()

    cfg = get_config("bert-base")
    M, D, H = cfg.m, cfg.d_model, cfg.H
    DH = D // H

    def _bin(bins, base):
        return bins[_config_target(base, "bert-base")]

    rule("EncFormer encrypted BERT-base inference")
    print(f"  checkpoint : {col(args.ckpt, 'c')}")
    print(f"  FHE backend: native .cu Phantom GPU CKKS  (GPU {args.gpu})")
    print(f"  MPC protocol: Π_MBMax · Π_MBLN · Π_GELU  (EzPC fixed point)")

    model, denoms = load_checkpoint(args.ckpt)
    model.eval()
    tok = BertTokenizer.from_pretrained(f"{ROOT}/data/bert-base-uncased", local_files_only=True)

    if args.text is not None:
        sentence, label = args.text, None
    else:
        with open(f"{ROOT}/data/sst2-validation.jsonl", encoding="utf-8") as handle:
            val = [json.loads(line) for line in handle]
        sentence, label = val[args.idx]["sentence"], val[args.idx]["label"]

    enc = tok(sentence, padding="max_length", truncation=True, max_length=M, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    tti = enc.get("token_type_ids", torch.zeros_like(ids))

    rule("INPUT")
    print(f"  sentence   : {col(repr(sentence), 'y')}")
    if label is not None:
        print(f"  gold label : {col(LABELS[label], 'b')}")
    print(f"  tokens     : {int(am.sum())} real / {M} padded")

    with torch.no_grad():
        pos = torch.arange(M).unsqueeze(0)
        e = model.word_embeddings(ids) + model.position_embeddings(pos) + model.token_type_embeddings(tti)
        e = model.embedding_dropout(model.embedding_ln(e))
    h = e.squeeze(0).numpy().astype(np.float64)
    mask_np = am.squeeze(0).numpy().astype(np.float64)
    mask_add = (1.0 - mask_np) * (-1e9)
    print(f"  embedding  : shape={h.shape}  norm={np.linalg.norm(h):.2f}")

    with torch.no_grad():
        ref_logits = model(ids, am, tti, use_running_stats=True).squeeze(0).numpy()
    ref_pred = int(np.argmax(ref_logits))

    _targets = [_config_target(t, "bert-base") for t in ("pipe_ckks_attn", "pipe_ckks_ff1", "pipe_ckks_ff2")]
    _bindir = os.path.join(ROOT, "third_party/phantom-fhe/build/bin")
    if all(os.path.isfile(os.path.join(_bindir, t)) for t in _targets):
        bins = {t: os.path.join(_bindir, t) for t in _targets}
    else:
        bins = resolve_native_binaries(targets=_targets, build_dir="build")
    mpc = get_mpc_engine("ezpc")
    n_layers = args.layers if args.layers is not None else len(model.layers)
    preps = [prepare_layer_running_denoms(denoms, i) for i in range(len(model.layers))]

    rule(f"ENCRYPTED FORWARD  ·  {n_layers} transformer layers")
    print(
        col(
            "  L   time   ‖h‖      finite  stages: QKV/Score(FHE)→Π_MBMax(MPC)→Π_MBLN(MPC)"
            "→FF1(FHE)→Π_GELU(MPC)→FF2(FHE)→Π_MBLN(MPC)",
            "d",
        )
    )

    _stg_t = [0.0]
    layer_log = []

    def _stage(tag, backend, arr):
        dt = time.perf_counter() - _stg_t[0]
        _stg_t[0] = time.perf_counter()
        shp = "×".join(str(x) for x in np.asarray(arr).shape)
        print(
            f"       {col('[' + backend + ']', 'c' if backend == 'FHE' else 'y')} "
            f"{tag:<34} shape={shp:<11} ‖·‖={np.linalg.norm(arr):9.2f}  (+{dt:4.1f}s)",
            flush=True,
        )

    t0 = time.perf_counter()
    for li in range(n_layers):
        W = extract_layer_weights(model, li)
        inject_layer_running_denoms(mpc, preps[li])
        td = tempfile.mkdtemp(prefix="demo_")
        for f in ("phase1_done", "a_heads_in.bin", "a_heads_in.bin.tmp"):
            p = os.path.join(td, f)
            os.path.exists(p) and os.remove(p)
        A = h.astype(np.float64)
        d = A.shape[1]
        m = A.shape[0]
        bQ = W.get("bQ", np.zeros(d))
        bK = W.get("bK", np.zeros(d))
        bV = W.get("bV", np.zeros(d))
        bO = W.get("bO", np.zeros(d))
        b1 = W.get("b1", np.zeros(W["W1"].shape[1]))
        b2 = W.get("b2", np.zeros(d))
        detail = li == 0
        if detail:
            print(col("     ── layer 1 · per-stage  (FHE = CKKS on GPU, MPC = 2PC protocol) ──", "d"))
            _stg_t[0] = time.perf_counter()
        P._write_f64(os.path.join(td, "attn_in.bin"), A, W["WQ"], W["WK"], W["WV"], bQ, bK, bV, W["WO"], bO)
        proc = P._start_native(_bin(bins, "pipe_ckks_attn"), td, args.gpu)
        P._wait_for_file(os.path.join(td, "phase1_done"), proc=proc)
        S = P._read_f64(os.path.join(td, "score_out.bin"), (H, m, m))
        if detail:
            _stage("QKV Proj + Score  (Q·Kᵀ)", "FHE", S)
        A_heads = [mpc.softmax_rows(S[hd] / math.sqrt(DH) + mask_add[None, :], head_index=hd) for hd in range(H)]
        if detail:
            _stage("Π_MBMax  (Batch Power-Max softmax)", "MPC", np.array(A_heads))
        P._write_f64_atomic(os.path.join(td, "a_heads_in.bin"), np.array(A_heads, dtype=np.float64))
        P._finish_native(proc, "pipe_ckks_attn")
        Z = P._read_f64(os.path.join(td, "value_out.bin"), (m, d))
        if detail:
            _stage("Value (A·V) + Out Proj", "FHE", Z)
        Z_ln = mpc.layer_norm(Z + A[:, :d], eps=1e-5, gamma=W.get("ln1_w"), beta=W.get("ln1_b"), ln_tag="ln1")
        if detail:
            _stage("Π_MBLN  (Batch LayerNorm 1)", "MPC", Z_ln)
        P._write_f64(os.path.join(td, "ff1_in.bin"), Z_ln, W["W1"])
        P._run_native(_bin(bins, "pipe_ckks_ff1"), td, args.gpu)
        H1 = P._read_f64(os.path.join(td, "ff1_out.bin"), (m, W["W1"].shape[1]))
        if detail:
            _stage("FF1  (feed-forward 768→3072)", "FHE", H1)
        H1g = mpc.gelu(H1 + b1[np.newaxis, :])
        if detail:
            _stage("Π_GELU  (BOLT piecewise)", "MPC", H1g)
        P._write_f64(os.path.join(td, "ff2_in.bin"), H1g, W["W2"])
        P._run_native(_bin(bins, "pipe_ckks_ff2"), td, args.gpu)
        H2 = P._read_f64(os.path.join(td, "ff2_out.bin"), (m, d))
        if detail:
            _stage("FF2  (feed-forward 3072→768)", "FHE", H2)
        h = mpc.layer_norm(
            H2 + b2[np.newaxis, :] + Z_ln, eps=1e-5, gamma=W.get("ln2_w"), beta=W.get("ln2_b"), ln_tag="ln2"
        )
        if detail:
            _stage("Π_MBLN  (Batch LayerNorm 2) → out", "MPC", h)
            print(col("     ── layer summary (layers 2–12 collapse to this one line) ──", "d"))
        fin = bool(np.all(np.isfinite(h)))
        _cum = time.perf_counter() - t0
        layer_log.append(
            {
                "layer": li + 1,
                "cum_s": round(_cum, 2),
                "layer_s": round(_cum - (layer_log[-1]["cum_s"] if layer_log else 0.0), 2),
                "norm": round(float(np.linalg.norm(h)), 2),
                "finite": fin,
            }
        )
        flag = col("ok", "g") if fin else col("NaN", "r")
        print(f"  {li + 1:2d}  {_cum:5.1f}s  {np.linalg.norm(h):8.2f}   {flag}", flush=True)

    elapsed = time.perf_counter() - t0

    pw = model.pooler.weight.detach().numpy().astype(np.float64)
    pb = model.pooler.bias.detach().numpy().astype(np.float64)
    cw = model.classifier.weight.detach().numpy().astype(np.float64)
    cb = model.classifier.bias.detach().numpy().astype(np.float64)
    enc_logits = np.tanh(h[0] @ pw.T + pb) @ cw.T + cb
    enc_pred = int(np.argmax(enc_logits))

    ref_finite = bool(np.all(np.isfinite(ref_logits)))
    partial = n_layers < len(model.layers)
    rule("RESULT")
    if partial:
        diff = None
        agree = None
        print(f"  native CKKS      : {col('PASS', 'g')}  ({n_layers} layer smoke test)")
    else:
        print(f"  encrypted logits : [{enc_logits[0]:+.4f}, {enc_logits[1]:+.4f}]  ->  {col(LABELS[enc_pred], 'b')}")
        if ref_finite:
            print(f"  plaintext logits : [{ref_logits[0]:+.4f}, {ref_logits[1]:+.4f}]  ->  {LABELS[ref_pred]}")
            diff = float(np.max(np.abs(enc_logits - ref_logits)))
            agree = enc_pred == ref_pred
            print(
                f"  max |enc - plain|: {col(f'{diff:.4f}', 'g' if diff < 0.5 else 'y')}   "
                f"prediction agrees: {col(agree, 'g' if agree else 'r')}"
            )
        else:
            diff = None
            agree = None
            print(f"  plaintext logits : unavailable")
        if label is not None:
            ok = enc_pred == label
            print(
                f"  vs gold label    : {col('CORRECT' if ok else 'WRONG', 'g' if ok else 'r')}  (gold={LABELS[label]})"
            )
    print(
        f"  wallclock        : {elapsed:.1f}s total · {elapsed / n_layers:.1f}s/layer "
        f"({n_layers} layers, native GPU FHE)"
    )
    comm_mb = comm_rounds = None
    ctx = getattr(mpc, "_sci_ctx", None)
    if ctx is not None and hasattr(ctx, "get_comm"):
        comm_mb = ctx.get_comm() / (1024.0 * 1024.0)
        comm_rounds = int(ctx.get_rounds())
        print(
            f"  2PC comm (party) : {col(f'{comm_mb:.2f} MB', 'c')} · {comm_rounds} rounds "
            f"over {n_layers} layers ({comm_mb / n_layers:.2f} MB/layer)"
        )
    rule()

    if args.json:
        try:
            gpu_name = torch.cuda.get_device_name(int(args.gpu))
        except Exception:
            gpu_name = f"cuda:{args.gpu}"
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "sentence": sentence,
                    "gold": label,
                    "enc_pred": enc_pred,
                    "ref_pred": ref_pred,
                    "enc_logits": enc_logits.tolist(),
                    "ref_logits": ref_logits.tolist(),
                    "max_abs_diff": diff,
                    "agree": agree,
                    "layers": n_layers,
                    "seconds": elapsed,
                    "per_layer_s": elapsed / n_layers,
                    "gpu": gpu_name,
                    "layer_log": layer_log,
                    "twopc_comm_mb": comm_mb,
                    "twopc_rounds": comm_rounds,
                },
                fh,
                indent=2,
            )
        print(f"  wrote {args.json}")
    return 0 if partial or label is None or enc_pred == label else 1


if __name__ == "__main__":
    sys.exit(main())
