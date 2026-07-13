#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def show(name, ok, value=""):
    state = "PASS" if ok else "FAIL"
    tail = f"  {value}" if value else ""
    print(f"[{state}] {name}{tail}")
    return ok


def checkpoint():
    return f"{ROOT}/checkpoints/encformer-sst2"


def main():
    ok = True
    try:
        import torch

        gpu = torch.cuda.is_available()
        ok &= show("CUDA", gpu, torch.cuda.get_device_name(0) if gpu else "unavailable")
        if gpu:
            cc = torch.cuda.get_device_capability(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            ok &= show("GPU architecture", cc[0] >= 7, f"sm_{cc[0]}{cc[1]}")
            ok &= show("GPU memory", mem >= 24, f"{mem:.0f} GB")
    except Exception as exc:
        ok &= show("CUDA", False, str(exc))
    base = f"{ROOT}/third_party/phantom-fhe/build"
    bins = [f"{base}/bin/pipe_ckks_{name}_bert_base" for name in ("attn", "ff1", "ff2")]
    ok &= show("PhantomFHE", all(map(os.path.isfile, bins)) and os.path.isfile(f"{base}/lib/libPhantom.so"))
    ckpt = checkpoint()
    ok &= show("EncFormer checkpoint", os.path.isfile(f"{ckpt}/model.pt"), ckpt)
    data = f"{ROOT}/data"
    ok &= show("BERT tokenizer", os.path.isfile(f"{data}/bert-base-uncased/vocab.txt"))
    with open(f"{data}/sst2-validation.jsonl", encoding="utf-8") as handle:
        samples = sum(1 for line in handle if line.strip())
    ok &= show("SST-2 validation", samples == 872, f"{samples} samples")
    path = os.environ.get("EZPC_PYTHONPATH", f"{ROOT}/third_party/ezpc-sci/build")
    sys.path.insert(0, path)
    try:
        import ezpc_sci

        native = bool(ezpc_sci.HAS_NATIVE_SCI and hasattr(ezpc_sci, "bpmax_2pc"))
        ok &= show("EzPC/SCI", native)
    except Exception as exc:
        ok &= show("EzPC/SCI", False, str(exc))
    with open(f"{ROOT}/src/engines/mpc_engine_ezpc.py", encoding="utf-8") as handle:
        code = handle.read()
    ok &= show("MPC parameters", "ring_bits: int = 43" in code and "scale_bits: int = 13" in code)
    with open(f"{ROOT}/third_party/phantom-fhe/src/prng.cu", encoding="utf-8") as handle:
        code = handle.read()
    ok &= show("CKKS secret", "sample_ternary_poly" in code and "hamming_weight" in code)
    print("EncFormer ready" if ok else "EncFormer check failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
