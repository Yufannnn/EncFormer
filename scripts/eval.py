#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def checkpoint():
    return f"{ROOT}/checkpoints/encformer-sst2"


def validation():
    with open(f"{ROOT}/data/sst2-validation.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def accuracy(limit, gpu):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, ROOT)
    import torch
    from transformers import BertTokenizer

    from src.engines.encformer_model import load_checkpoint

    data = validation()
    tok = BertTokenizer.from_pretrained(f"{ROOT}/data/bert-base-uncased", local_files_only=True)
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(checkpoint())
    model.to(dev).eval()
    total = min(limit, len(data))
    correct = 0
    finite = 0
    with torch.no_grad():
        for start in range(0, total, 64):
            ids = list(range(start, min(start + 64, total)))
            batch = tok(
                [data[i]["sentence"] for i in ids],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            logits = model(
                batch["input_ids"].to(dev),
                batch["attention_mask"].to(dev),
                batch.get("token_type_ids").to(dev),
                use_running_stats=True,
            )
            labels = torch.tensor([data[i]["label"] for i in ids], device=dev)
            valid = torch.isfinite(logits).all(-1)
            finite += int(valid.sum())
            correct += int(((logits.argmax(-1) == labels) & valid).sum())
    return correct / total, finite / total


def inference(layers, gpu):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        path = handle.name
    cmd = [
        sys.executable,
        f"{ROOT}/scripts/demo.py",
        "--ckpt",
        checkpoint(),
        "--idx",
        "40",
        "--gpu",
        str(gpu),
        "--layers",
        str(layers),
        "--json",
        path,
    ]
    run = subprocess.run(cmd, env=dict(os.environ, HF_HUB_OFFLINE="1"), capture_output=True, text=True, timeout=1800)
    if run.returncode:
        raise RuntimeError(run.stderr or run.stdout)
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    os.unlink(path)
    return result


def main():
    parser = argparse.ArgumentParser(prog="EncFormer evaluation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--json")
    args = parser.parse_args()
    full = args.full
    checks = []
    start = time.perf_counter()
    env = subprocess.run([sys.executable, f"{ROOT}/scripts/check.py"], capture_output=True).returncode == 0
    checks.append(("runtime", env, "ready"))
    acc, finite = accuracy(872 if full else 200, args.gpu)
    checks.append(("SST-2 accuracy", 0.884 <= acc <= 0.944, f"{acc:.4f}"))
    checks.append(("finite outputs", finite >= 0.95, f"{finite:.4f}"))
    result = inference(12 if full else 1, args.gpu)
    values = result.get("enc_logits") or []
    checks.append(("native CKKS", bool(values) and all(value == value for value in values), "finite"))
    if full:
        checks.append(("prediction", result["enc_pred"] == result["gold"], str(result["enc_pred"])))
        delta = result.get("max_abs_diff")
        checks.append(
            ("logit difference", delta is not None and delta < 0.01, "none" if delta is None else f"{delta:.6f}")
        )
    passed = all(item[1] for item in checks)
    print("EncFormer evaluation")
    for name, ok, value in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {value}")
    print(
        f"{'PASS' if passed else 'FAIL'}  {sum(item[1] for item in checks)}/{len(checks)}  {time.perf_counter() - start:.1f}s"
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"pass": passed, "accuracy": acc, "finite": finite, "checks": checks}, handle, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
