#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

ROOT = os.environ.get("ENCFORMER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["LD_LIBRARY_PATH"] = f"{ROOT}/third_party/phantom-fhe/build/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("MPC_EZPC_THREADS", "8")
os.environ.setdefault("EZPC_PYTHONPATH", f"{ROOT}/third_party/ezpc-sci/build")
os.environ["MPC_BATCH_METHOD"] = "encformer"
os.environ["MPC_BATCH_INFERENCE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, ROOT)
from transformers import BertTokenizer

from src.engines.encformer_model import extract_layer_weights, load_checkpoint
from src.fhe.phantom import two_party_native as tp
from src.inference_runtime import prepare_layer_running_denoms
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
    ap.add_argument("--gpu", default="0", help="CUDA device index")
    ap.add_argument("--layers", type=int, default=None, help="#transformer layers (default: all 12)")
    ap.add_argument("--json", default=None, help="write a machine-readable result JSON here")
    args = ap.parse_args()

    cfg = get_config("bert-base")
    M = cfg.m
    rule("EncFormer two-party encrypted BERT-base inference")
    print(f"  checkpoint : {col(args.ckpt, 'c')}")
    print(f"  server     : Phantom GPU CKKS, evaluation keys only  (GPU {args.gpu})")
    print(f"  client     : secret key")
    print(f"  MPC        : EzPC/SCI two-party  (Π_MBMax · Π_MBLN · Π_GELU)")

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

    n_layers = args.layers if args.layers is not None else len(model.layers)
    with torch.no_grad():
        pos = torch.arange(M).unsqueeze(0)
        e = model.word_embeddings(ids) + model.position_embeddings(pos) + model.token_type_embeddings(tti)
        e = model.embedding_ln(e)
        ref_logits = model(ids, am, tti, use_running_stats=True).squeeze(0).numpy().astype(np.float64)
        ext = (1.0 - am[:, None, None, :].float()) * torch.finfo(torch.float32).min
        ref_h = e
        for layer in model.layers[:n_layers]:
            ref_h = layer(ref_h, ext, use_running_stats=True)
    x0 = e.squeeze(0).numpy().astype(np.float64)
    ref_h = ref_h.squeeze(0).numpy().astype(np.float64)
    mask = am.squeeze(0).numpy().astype(np.float64)

    W = [extract_layer_weights(model, i) for i in range(n_layers)]
    pub = [tp.LayerPublic(bO=w["bO"], b1=w["b1"], b2=w["b2"], ln1_w=w["ln1_w"], ln1_b=w["ln1_b"], ln2_w=w["ln2_w"],
                          ln2_b=w["ln2_b"], denoms=prepare_layer_running_denoms(denoms, i)) for i, w in enumerate(W)]
    g = tp.Geom(m=cfg.m, d=cfg.d_model, h=cfg.H, d_ff=cfg.d_ff, nslots=cfg.nslots)

    rule(f"ENCRYPTED FORWARD  ·  {n_layers} transformer layers")
    print(col("  per layer: M2C → QKV/Score (CKKS) → C2M → Π_MBMax → M2C → Value/OUT (CKKS) → C2M → Π_MBLN", "d"))
    print(col("             → M2C → FF1 + GELU candidates (CKKS) → C2M → Π_GELU → M2C → FF2 (CKKS) → C2M → Π_MBLN", "d"))
    t0 = time.perf_counter()
    h, layer_s = tp.run(W, pub, x0, mask, g, gpu=args.gpu)
    wall = time.perf_counter() - t0
    online = float(sum(layer_s))

    finite = bool(np.all(np.isfinite(h)))
    rel = float(np.linalg.norm(h - ref_h) / np.linalg.norm(ref_h))
    partial = n_layers < len(model.layers)
    enc_logits = enc_pred = diff = agree = None
    rule("RESULT")
    if partial:
        ok = finite and rel < 0.01
        print(f"  {n_layers}-layer output vs plaintext : rel. error {col(f'{rel:.2e}', 'g' if ok else 'r')}")
    else:
        pw = model.pooler.weight.detach().numpy().astype(np.float64)
        pb = model.pooler.bias.detach().numpy().astype(np.float64)
        cw = model.classifier.weight.detach().numpy().astype(np.float64)
        cb = model.classifier.bias.detach().numpy().astype(np.float64)
        enc_logits = np.tanh(h[0] @ pw.T + pb) @ cw.T + cb
        enc_pred = int(np.argmax(enc_logits))
        ref_pred = int(np.argmax(ref_logits))
        diff = float(np.max(np.abs(enc_logits - ref_logits)))
        agree = enc_pred == ref_pred
        print(f"  encrypted logits : [{enc_logits[0]:+.4f}, {enc_logits[1]:+.4f}]  ->  {col(LABELS[enc_pred], 'b')}")
        print(f"  plaintext logits : [{ref_logits[0]:+.4f}, {ref_logits[1]:+.4f}]  ->  {LABELS[ref_pred]}")
        print(f"  max |enc - plain|: {diff:.4f}   prediction agrees: {col(agree, 'g' if agree else 'r')}")
        if label is not None:
            ok = enc_pred == label
            print(f"  vs gold label    : {col('CORRECT' if ok else 'WRONG', 'g' if ok else 'r')}  (gold={LABELS[label]})")
    print(f"  online time      : {online:.1f}s · {online / n_layers:.2f}s/layer  (setup and offline excluded)")
    print(f"  wallclock        : {wall:.1f}s")
    rule()

    if args.json:
        try:
            gpu_name = torch.cuda.get_device_name(int(args.gpu))
        except Exception:
            gpu_name = f"cuda:{args.gpu}"
        with open(args.json, "w") as fh:
            json.dump({"sentence": sentence, "gold": label, "enc_pred": enc_pred, "ref_pred": int(np.argmax(ref_logits)),
                       "enc_logits": None if enc_logits is None else enc_logits.tolist(),
                       "ref_logits": ref_logits.tolist(), "max_abs_diff": diff, "agree": agree, "finite": finite,
                       "hidden_rel_err": rel, "layers": n_layers, "online_s": online, "per_layer_s": layer_s,
                       "wall_s": wall, "gpu": gpu_name}, fh, indent=2)
        print(f"  wrote {args.json}")
    if partial:
        return 0 if finite and rel < 0.01 else 1
    return 0 if label is None or enc_pred == label else 1


if __name__ == "__main__":
    sys.exit(main())
