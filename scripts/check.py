#!/usr/bin/env python3
import os
import subprocess
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
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    try:
        query = "name,compute_cap,memory.total"
        line = subprocess.run(["nvidia-smi", "-i", gpu, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
        name, cc, mem = [v.strip() for v in line.split(",")]
        ok &= show("CUDA", True, name)
        ok &= show("GPU architecture", float(cc) >= 7.0, f"sm_{cc.replace('.', '')}")
        ok &= show("GPU memory", float(mem) / 1024 >= 40, f"{float(mem) / 1024:.0f} GB")
    except Exception as exc:
        ok &= show("CUDA", False, str(exc).splitlines()[0] if str(exc) else "nvidia-smi unavailable")
    base = f"{ROOT}/third_party/phantom-fhe/build"
    bins = [f"{base}/bin/pipe_tp_{role}_bert_base" for role in ("server", "client")]
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

        native = bool(ezpc_sci.HAS_NATIVE_SCI and getattr(ezpc_sci, "HAS_OT_POOL", False))
        ok &= show("EzPC/SCI", native)
    except Exception as exc:
        ok &= show("EzPC/SCI", False, str(exc))
    print("EncFormer ready" if ok else "EncFormer check failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
