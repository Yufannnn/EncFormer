#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engines.encformer_model import (
    EncFormerBertForSequenceClassification,
    EncFormerGPT2LM,
    extract_layer_weights,
    infer_model_config_name,
    load_checkpoint,
)
from src.inference_runtime import prepare_layer_running_denoms
from src.models.model_config import get_config

TASK_CONFIG = {
    "sst2": {
        "dataset": "stanfordnlp/sst2",
        "num_labels": 2,
        "metric": "accuracy",
        "text_cols": ("sentence",),
        "label_col": "label",
    },
    "mrpc": {
        "dataset": "nyu-mll/glue",
        "subset": "mrpc",
        "num_labels": 2,
        "metric": "accuracy_f1",
        "text_cols": ("sentence1", "sentence2"),
        "label_col": "label",
    },
    "rte": {
        "dataset": "nyu-mll/glue",
        "subset": "rte",
        "num_labels": 2,
        "metric": "accuracy",
        "text_cols": ("sentence1", "sentence2"),
        "label_col": "label",
    },
}


def load_eval_data(task: str, tokenizer, max_len: int = 128):
    from datasets import load_dataset

    cfg = TASK_CONFIG[task]
    subset = cfg.get("subset")
    if subset:
        ds = load_dataset(cfg["dataset"], subset)
    else:
        ds = load_dataset(cfg["dataset"])

    text_cols = cfg["text_cols"]

    def tokenize(examples):
        if len(text_cols) == 1:
            return tokenizer(
                examples[text_cols[0]],
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
        else:
            return tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )

    val_key = "validation" if "validation" in ds else "test"
    val_ds = ds[val_key].map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "token_type_ids"]
    if "token_type_ids" not in val_ds.column_names:
        cols = ["input_ids", "attention_mask"]

    val_ds.set_format("torch", columns=cols + [cfg["label_col"]])
    return val_ds, cfg


def validate_model_matches_config(model, model_config_name: str) -> None:

    cfg = get_config(model_config_name)
    is_gpt2 = isinstance(model, EncFormerGPT2LM)
    expected_model_type = "gpt2" if cfg.name.startswith("gpt2") else "bert"
    actual_model_type = "gpt2" if is_gpt2 else "bert"

    mismatches = []
    if actual_model_type != expected_model_type:
        mismatches.append(f"type={actual_model_type} (expected {expected_model_type})")
    if model.hidden_size != cfg.d_model:
        mismatches.append(f"hidden_size={model.hidden_size} (expected {cfg.d_model})")
    if len(model.layers) != cfg.num_layers:
        mismatches.append(f"num_layers={len(model.layers)} (expected {cfg.num_layers})")
    if model.seq_len != cfg.m:
        mismatches.append(f"seq_len={model.seq_len} (expected {cfg.m})")

    if mismatches:
        raise ValueError(
            "Checkpoint/model_config mismatch: "
            + ", ".join(mismatches)
            + f". Requested model_config={model_config_name!r}."
        )


def resolve_model_config_name(model, requested_name: str | None) -> str:

    inferred_name = infer_model_config_name(model)
    if requested_name is None:
        if inferred_name is None:
            raise ValueError("Could not infer model_config from the checkpoint. Please pass --model_config explicitly.")
        return inferred_name

    validate_model_matches_config(model, requested_name)
    if inferred_name is not None and inferred_name != requested_name:
        raise ValueError(
            f"Requested --model_config {requested_name!r} does not match the loaded checkpoint ({inferred_name!r})."
        )
    return requested_name


def validate_eval_support(model, task: str, ckks_backend: str, model_config_name: str) -> None:
    pass


def eval_pytorch_reference(model, val_ds, device, max_samples=None):

    is_gpt2 = isinstance(model, EncFormerGPT2LM)
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    n = len(val_ds) if max_samples is None else min(max_samples, len(val_ds))
    with torch.no_grad():
        for i in range(n):
            sample = val_ds[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attn_mask = sample["attention_mask"].unsqueeze(0).to(device)

            if is_gpt2:
                logits = model(input_ids, attn_mask, use_running_stats=True)

                seq_len = attn_mask.sum(dim=-1).int().item()
                pred = logits[0, seq_len - 1].argmax(dim=-1).item()
            else:
                token_type_ids = sample.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.unsqueeze(0).to(device)
                logits = model(input_ids, attn_mask, token_type_ids, use_running_stats=True)
                pred = logits.argmax(dim=-1).item()

            label = sample["label"].item()

            all_preds.append(pred)
            all_labels.append(label)
            if pred == label:
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy, all_preds, all_labels


def eval_encformer_pipeline(
    model,
    val_ds,
    denoms,
    engine_name,
    max_samples=None,
    ckks_backend="plain",
    gpu=None,
    two_party=False,
    secure_head=False,
):

    os.environ["MPC_BATCH_METHOD"] = "on"
    os.environ["MPC_BATCH_INFERENCE"] = "1"

    from src.encformer import _make_ckks_context, run_with_weights
    from src.engines.mpc_engine_factory import get_mpc_engine, resolve_pipeline

    run_fn = run_with_weights
    model_config_name = infer_model_config_name(model)
    if model_config_name is None:
        raise ValueError("Could not infer model config from the loaded checkpoint for CKKS runtime setup.")
    model_config = get_config(model_config_name)
    pipeline_ckks, _ = resolve_pipeline()
    effective_ckks_backend = pipeline_ckks or ckks_backend
    if effective_ckks_backend == "phantom_native" and two_party:
        raise ValueError("two_party mode is not supported with phantom_native in eval_glue yet.")
    shared_ctx = None
    if effective_ckks_backend != "phantom_native":
        shared_ctx = _make_ckks_context(effective_ckks_backend, model_config.nslots, gpu=gpu)
    shared_mpc_backend = get_mpc_engine(engine_name)

    _bridge = None
    _client_thread = None
    _bridge_ch = None
    if two_party:
        import socket as _socket
        import threading

        from src.bridges.channel import SocketChannel
        from src.bridges.two_party_bridge import (
            TwoPartyServerBridge,
            client_bridge_loop,
        )

        srv_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        srv_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        srv_sock.bind(("127.0.0.1", 0))
        port = srv_sock.getsockname()[1]
        srv_sock.listen(1)

        _client_result = {}

        def _client_fn():
            cli_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            cli_sock.connect(("127.0.0.1", port))
            ch = SocketChannel(cli_sock)
            _client_result["stats"] = client_bridge_loop(ch)
            _client_result["comm"] = ch.stats
            ch.close()

        _client_thread = threading.Thread(target=_client_fn, daemon=True)
        _client_thread.start()
        conn, _ = srv_sock.accept()
        srv_sock.close()
        _bridge_ch = SocketChannel(conn)
        _bridge = TwoPartyServerBridge(shared_ctx, _bridge_ch)

    is_gpt2 = isinstance(model, EncFormerGPT2LM)
    num_layers = len(model.layers)
    all_weights = [extract_layer_weights(model, layer_idx=i) for i in range(num_layers)]
    all_prepared_denoms = [prepare_layer_running_denoms(denoms, i) for i in range(num_layers)]

    if not is_gpt2:
        cls_w = model.classifier.weight.detach().cpu().numpy().astype(np.float64)
        cls_b = model.classifier.bias.detach().cpu().numpy().astype(np.float64)
        pooler_w = model.pooler.weight.detach().cpu().numpy().astype(np.float64)
        pooler_b = model.pooler.bias.detach().cpu().numpy().astype(np.float64)
    else:
        lm_w = model.lm_head.weight.detach().cpu().numpy().astype(np.float64)

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    n = len(val_ds) if max_samples is None else min(max_samples, len(val_ds))
    print(f"  Running {num_layers} layers per sample ({'GPT-2 LM' if is_gpt2 else 'BERT cls'})")
    print("  Reusing CKKS/MPC runtime across layers; per-layer timings shown for the first sample only.")
    if secure_head:
        print("  [Secure head] Embeddings: client-side (local computation before encryption)")
        if is_gpt2:
            from src.encformer import decb, secure_final_ln, secure_linear_head

            _mc = get_config(infer_model_config_name(model))
            _vocab = model.lm_head.weight.shape[0]
            _C = shared_ctx.nslots // _mc.m
            _d_out_padded = ((_vocab + _C - 1) // _C) * _C
            print(f"  [Secure head] GPT-2 LM head: encrypted LN (MPC) + linear (CKKS)")
        else:
            from src.utils import to_mpc_mat as _to_mpc_mat

            print("  [Secure head] BERT: bridge-based masked decryption + task head")

    try:
        for i in range(n):
            sample = val_ds[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attn_mask = sample["attention_mask"].unsqueeze(0).to(device)

            if is_gpt2:
                seq_len = input_ids.shape[1]
                causal_mask = np.where(np.tril(np.ones((seq_len, seq_len))) == 0, -1e9, 0.0)
                mask_np = causal_mask

                with torch.no_grad():
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    embeddings = model.word_embeddings(input_ids) + model.position_embeddings(position_ids)
                    embeddings = model.embedding_dropout(embeddings)
            else:
                mask_np = attn_mask.squeeze(0).cpu().numpy().astype(np.float64)
                token_type_ids = sample.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.unsqueeze(0).to(device)

                with torch.no_grad():
                    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                    embeddings = (
                        model.word_embeddings(input_ids)
                        + model.position_embeddings(position_ids)
                        + model.token_type_embeddings(
                            token_type_ids if token_type_ids is not None else torch.zeros_like(input_ids)
                        )
                    )
                    embeddings = model.embedding_dropout(model.embedding_ln(embeddings))

            hidden = embeddings.squeeze(0).detach().cpu().numpy().astype(np.float64)

            t0 = time.perf_counter()
            for layer_idx in range(num_layers):
                verbose_layer = i == 0
                is_last = (layer_idx == num_layers - 1) and secure_head
                hidden = run_fn(
                    weights=all_weights[layer_idx],
                    running_denoms=denoms,
                    input_embeds=hidden,
                    attention_mask=mask_np,
                    layer_idx=layer_idx,
                    use_cc=True,
                    mpc_engine=engine_name,
                    ckks_backend=ckks_backend,
                    gpu=gpu,
                    ctx=shared_ctx,
                    mpc_backend=shared_mpc_backend,
                    prepared_denoms=all_prepared_denoms[layer_idx],
                    verbose=verbose_layer,
                    pre_ln=is_gpt2,
                    bridge=_bridge,
                    return_encrypted=is_last,
                )
            dt = time.perf_counter() - t0

            if secure_head and is_gpt2 and True:
                t_head = time.perf_counter()
                ln_f_w = model.ln_f.weight.detach().cpu().numpy().astype(np.float64)
                ln_f_b = model.ln_f.bias.detach().cpu().numpy().astype(np.float64)
                hidden_blocks = hidden
                ln_blocks = secure_final_ln(
                    shared_ctx,
                    hidden_blocks,
                    gamma=ln_f_w,
                    beta=ln_f_b,
                    m=_mc.m,
                    d=_mc.d_model,
                    mpc_engine=shared_mpc_backend,
                    bridge=_bridge,
                )
                logits_blocks = secure_linear_head(
                    shared_ctx,
                    ln_blocks,
                    lm_w.T,
                    m=_mc.m,
                )
                logits_padded = decb(shared_ctx, logits_blocks, m=_mc.m, d_out=_d_out_padded)
                seq_lengths = attn_mask.squeeze(0).sum().int().item()
                last_token_logits = logits_padded[seq_lengths - 1, :_vocab]
                pred = int(np.argmax(last_token_logits))
                if i == 0:
                    print(f"  [Secure head] Task head time: {time.perf_counter() - t_head:.2f}s")
            elif secure_head and not is_gpt2 and True:
                t_head = time.perf_counter()
                hidden_blocks = hidden
                _mc_b = get_config(infer_model_config_name(model))
                hidden, _ = _to_mpc_mat(
                    shared_ctx,
                    hidden_blocks,
                    m=_mc_b.m,
                    d_out=_mc_b.d_model,
                    use_cc=True,
                    bridge=_bridge,
                )
                cls_token = hidden[0]
                pooled = np.tanh(cls_token @ pooler_w.T + pooler_b)
                logits = pooled @ cls_w.T + cls_b
                pred = int(np.argmax(logits))
                if i == 0:
                    print(f"  [Secure head] Task head time: {time.perf_counter() - t_head:.2f}s")
            elif is_gpt2:
                from src.engines.mpc_engine_plain import PlainMpcEngine

                _ln = PlainMpcEngine()
                hidden = _ln.layer_norm(
                    hidden,
                    eps=1e-5,
                    gamma=model.ln_f.weight.detach().cpu().numpy().astype(np.float64),
                    beta=model.ln_f.bias.detach().cpu().numpy().astype(np.float64),
                )
                seq_lengths = attn_mask.squeeze(0).sum().int().item()
                last_token = hidden[seq_lengths - 1]
                logits = last_token @ lm_w.T
                pred = int(np.argmax(logits))
            else:
                cls_token = hidden[0]
                pooled = np.tanh(cls_token @ pooler_w.T + pooler_b)
                logits = pooled @ cls_w.T + cls_b
                pred = int(np.argmax(logits))

            label = sample["label"].item()

            all_preds.append(pred)
            all_labels.append(label)
            if pred == label:
                correct += 1
            total += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{n}] acc={correct / total:.4f} (last sample: {dt:.2f}s, {num_layers} layers)")
    finally:
        if _bridge is not None:
            _bridge.finish()
            _client_thread.join(timeout=10)
            sc = _bridge_ch.stats
            _bridge_ch.close()

            def _fmt(n):
                return f"{n / (1 << 20):.1f} MB" if n >= 1 << 20 else f"{n / (1 << 10):.1f} KB"

            print(
                f"\n  [Two-party comm] sent={_fmt(sc['bytes_sent'])} recv={_fmt(sc['bytes_recv'])} "
                f"total={_fmt(sc['bytes_total'])} msgs={sc['msgs_sent'] + sc['msgs_recv']}"
            )

    accuracy = correct / total
    return accuracy, all_preds, all_labels


def compute_metrics(preds, labels, metric_type):
    accuracy = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
    result = {"accuracy": accuracy}

    if metric_type == "accuracy_f1":
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        result["f1"] = f1

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate on GLUE tasks via EncFormer pipeline")
    parser.add_argument("--task", type=str, required=True, choices=["sst2", "mrpc", "rte"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--engine", type=str, default="plain", choices=["plain", "crypten"])
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (for quick testing)")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--max_len", type=int, default=None, help="Sequence length for eval; must match the selected model config"
    )
    parser.add_argument("--skip_pytorch", action="store_true", help="Skip PyTorch reference evaluation")
    parser.add_argument("--skip_encformer", action="store_true", help="Skip EncFormer pipeline evaluation")
    parser.add_argument(
        "--ckks_backend",
        type=str,
        default="plain",
        choices=["plain", "phantom", "desilo", "phantom_native"],
        help="CKKS engine backend (plain=simulator, phantom/desilo=GPU, phantom_native=native Phantom CUDA)",
    )
    parser.add_argument(
        "--gpu", type=str, default="3", help="CUDA_VISIBLE_DEVICES for phantom/desilo backends (default: 3)"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        choices=["bert-base", "bert-large", "gpt2-base"],
        help="Model configuration; inferred from checkpoint if omitted",
    )
    parser.add_argument("--two_party", action="store_true", help="Use real two-party bridge over TCP sockets")
    parser.add_argument(
        "--secure_head", action="store_true", help="Apply task head in encrypted domain (end-to-end secure)"
    )
    args = parser.parse_args()

    if args.skip_pytorch and args.skip_encformer:
        raise ValueError("At least one of --skip_pytorch or --skip_encformer must be false.")

    model, denoms = load_checkpoint(args.checkpoint)
    args.model_config = resolve_model_config_name(model, args.model_config)
    cfg = get_config(args.model_config)
    max_len = args.max_len or cfg.m
    if max_len != cfg.m:
        raise ValueError(
            f"--max_len must equal the selected model config sequence length ({cfg.m}) "
            f"for the secure pipeline, got {max_len}."
        )

    validate_model_matches_config(model, args.model_config)
    validate_eval_support(model, args.task, args.ckks_backend, args.model_config)

    os.environ["ENCFORMER_MODEL"] = args.model_config

    is_gpt2 = isinstance(model, EncFormerGPT2LM)
    print(
        f"[Eval] task={args.task} checkpoint={args.checkpoint} engine={args.engine} "
        f"ckks={args.ckks_backend} model={args.model_config} seq_len={max_len}"
    )
    print(
        f"[Model] type={'gpt2' if is_gpt2 else 'bert'} layers={len(model.layers)} denoms={list(denoms.keys())[:3]}..."
    )

    if is_gpt2:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    val_ds, task_cfg = load_eval_data(args.task, tokenizer, max_len)
    print(f"[Data] val={len(val_ds)}")

    if not args.skip_pytorch:
        print("\n--- PyTorch Reference Evaluation ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dev = model.to(device)
        ref_acc, ref_preds, ref_labels = eval_pytorch_reference(model_dev, val_ds, device, args.max_samples)
        ref_metrics = compute_metrics(ref_preds, ref_labels, task_cfg["metric"])
        print(f"  PyTorch reference: {ref_metrics}")
        model = model.cpu()

    if not args.skip_encformer:
        gpu_arg = args.gpu if args.ckks_backend != "plain" else None
        print(f"\n--- EncFormer Pipeline Evaluation (engine={args.engine}, ckks={args.ckks_backend}) ---")
        enc_acc, enc_preds, enc_labels = eval_encformer_pipeline(
            model,
            val_ds,
            denoms,
            args.engine,
            args.max_samples,
            ckks_backend=args.ckks_backend,
            gpu=gpu_arg,
            two_party=args.two_party,
            secure_head=args.secure_head,
        )
        enc_metrics = compute_metrics(enc_preds, enc_labels, task_cfg["metric"])
        print(f"  EncFormer pipeline: {enc_metrics}")

        if not args.skip_pytorch:
            agree = sum(1 for p1, p2 in zip(ref_preds, enc_preds) if p1 == p2)
            print(f"\n  Agreement: {agree}/{len(ref_preds)} ({agree / len(ref_preds):.4f})")

    print("\n[Done]")


if __name__ == "__main__":
    main()
