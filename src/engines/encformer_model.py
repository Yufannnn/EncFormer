from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_GELU_A = 0.020848611754127593
_GELU_B = -0.18352506127082727
_GELU_C = 0.5410550166368381
_GELU_D = -0.03798164612714154
_GELU_E = 0.001620808531841547
_GELU_THRESH = 2.7
_GELU_SCALE_BITS = 13
_GELU_SCALE = float(1 << _GELU_SCALE_BITS)


def _quantize_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * _GELU_SCALE) / _GELU_SCALE


def _approx_gelu_torch(x: torch.Tensor) -> torch.Tensor:

    xq = _quantize_torch(x)
    x2 = xq * xq
    x4 = x2 * x2

    a = _quantize_torch(torch.tensor(_GELU_A, dtype=x.dtype, device=x.device))
    b = _quantize_torch(torch.tensor(_GELU_B, dtype=x.dtype, device=x.device))
    c = _quantize_torch(torch.tensor(_GELU_C, dtype=x.dtype, device=x.device))
    p = _quantize_torch(torch.tensor(0.5 + _GELU_D, dtype=x.dtype, device=x.device))
    m = _quantize_torch(torch.tensor(0.5 - _GELU_D, dtype=x.dtype, device=x.device))
    e = _quantize_torch(torch.tensor(_GELU_E, dtype=x.dtype, device=x.device))
    neg_t = _quantize_torch(torch.tensor(-_GELU_THRESH, dtype=x.dtype, device=x.device))
    pos_t = _quantize_torch(torch.tensor(_GELU_THRESH, dtype=x.dtype, device=x.device))

    f0 = a * x4 - b * (x2 * xq) + c * x2 + m * xq + e
    f1 = a * x4 + b * (x2 * xq) + c * x2 + p * xq + e

    b0 = (xq < neg_t).to(x.dtype)
    b1 = (xq < 0).to(x.dtype)
    b2 = (xq > pos_t).to(x.dtype)

    z0 = b0 + b1 - 2.0 * b0 * b1
    z1 = (1.0 - b0) * (1.0 - b2)
    z1 = z1 - z0
    z2 = b2

    y = f0 * z0 + f1 * z1 + xq * z2
    return _quantize_torch(y)


class BPMaxAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        seq_len: int,
        p: int = 5,
        c: float = 5.0,
        eps: float = 1e-10,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.p = p
        self.c = c
        self.eps = eps
        self.seq_len = seq_len
        self.momentum = momentum

        self.exact_nonlinear = False

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.register_buffer(
            "running_denominator",
            torch.ones(1, num_heads, seq_len, 1),
        )
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_running_stats: bool = False,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.query(hidden_states).view(B, S, H, D).transpose(1, 2)
        K = self.key(hidden_states).view(B, S, H, D).transpose(1, 2)
        V = self.value(hidden_states).view(B, S, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        self._last_scores = scores

        if attention_mask is not None:
            scores = scores + attention_mask

        if self.exact_nonlinear:
            attn_probs = F.softmax(scores, dim=-1)
        else:
            z = (scores + self.c).clamp(min=0.0)
            pw = torch.pow(z, self.p)
            den = pw.sum(dim=-1, keepdim=True)

            if self.training:
                den_batch_max = den.max(dim=0, keepdim=True).values
                self.num_batches_tracked += 1

                with torch.no_grad():
                    self.running_denominator[:, :, :S, :] = den_batch_max

                attn_probs = pw / (den_batch_max + self.eps)
            elif use_running_stats:
                rd_slice = self.running_denominator[:, :, :S, :]
                attn_probs = pw / (rd_slice + self.eps)
            else:
                attn_probs = pw / (den + self.eps)

        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(B, S, -1)
        return context


class BatchLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        seq_len: int,
        l: float = 1.0,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.l = l
        self.eps = eps
        self.momentum = momentum

        self.exact_nonlinear = False
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.register_buffer(
            "running_denominator",
            torch.ones(1, seq_len, 1),
        )
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor, use_running_stats: bool = False) -> torch.Tensor:

        mean = x.mean(dim=-1, keepdim=True)
        xc = x - mean
        rms = torch.sqrt((xc * xc).mean(dim=-1, keepdim=True) + self.eps)

        S = x.shape[1]
        if self.exact_nonlinear:
            y = xc / rms
        elif self.training:
            rms_batch_max = rms.max(dim=0, keepdim=True).values * self.l
            self.num_batches_tracked += 1
            with torch.no_grad():
                self.running_denominator[:, :S, :] = rms_batch_max

            y = xc / (rms_batch_max + self.eps)
        elif use_running_stats:
            rd_slice = self.running_denominator[:, :S, :]
            y = xc / (rd_slice + self.eps)
        else:
            y = xc / (rms + self.eps)

        return y * self.weight + self.bias


class EncFormerBertLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        seq_len: int = 128,
        p: int = 5,
        c: float = 5.0,
        ln_l: float = 1.0,
    ):
        super().__init__()
        self.attention = BPMaxAttention(hidden_size, num_heads, seq_len, p=p, c=c)
        self.attn_output = nn.Linear(hidden_size, hidden_size)
        self.ln1 = BatchLayerNorm(hidden_size, seq_len, l=ln_l)
        self.ff1 = nn.Linear(hidden_size, intermediate_size)
        self.ff2 = nn.Linear(intermediate_size, hidden_size)
        self.ln2 = BatchLayerNorm(hidden_size, seq_len, l=ln_l)

        self.exact_nonlinear = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_running_stats: bool = False,
    ) -> torch.Tensor:
        attn_out = self.attention(hidden_states, attention_mask, use_running_stats=use_running_stats)
        attn_out = self.attn_output(attn_out)
        x = self.ln1(hidden_states + attn_out, use_running_stats=use_running_stats)
        ff_hidden = self.ff1(x)
        if self.exact_nonlinear:
            ff_act = F.gelu(ff_hidden)
        elif use_running_stats and not self.training:
            ff_act = _approx_gelu_torch(ff_hidden)
        else:
            ff_act = F.gelu(ff_hidden)
        ff_out = self.ff2(ff_act)
        out = self.ln2(x + ff_out, use_running_stats=use_running_stats)

        self._last_attn_scores = self.attention._last_scores
        self._last_hidden = out
        return out


class EncFormerBertForSequenceClassification(nn.Module):
    def __init__(
        self,
        num_labels: int = 2,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        intermediate_size: int = 3072,
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        seq_len: int = 128,
        p: int = 5,
        c: float = 5.0,
        ln_l: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.seq_len = seq_len

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.embedding_ln = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList(
            [
                EncFormerBertLayer(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    seq_len,
                    p=p,
                    c=c,
                    ln_l=ln_l,
                )
                for _ in range(num_layers)
            ]
        )

        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        use_running_stats: bool = False,
    ) -> torch.Tensor:
        B, S = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        hidden = self.embedding_dropout(self.embedding_ln(embeddings))
        self._last_embeddings = hidden

        if attention_mask is not None:
            ext_mask = attention_mask[:, None, None, :].to(hidden.dtype)
            ext_mask = (1.0 - ext_mask) * torch.finfo(hidden.dtype).min
        else:
            ext_mask = None

        for layer in self.layers:
            hidden = layer(hidden, ext_mask, use_running_stats=use_running_stats)

        pooled = torch.tanh(self.pooler(hidden[:, 0]))
        logits = self.classifier(self.dropout(pooled))
        return logits


class EncFormerGPT2Layer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        seq_len: int = 64,
        p: int = 5,
        c: float = 5.0,
        ln_l: float = 1.0,
    ):
        super().__init__()
        self.ln1 = BatchLayerNorm(hidden_size, seq_len, l=ln_l)
        self.attention = BPMaxAttention(hidden_size, num_heads, seq_len, p=p, c=c)
        self.attn_output = nn.Linear(hidden_size, hidden_size)
        self.ln2 = BatchLayerNorm(hidden_size, seq_len, l=ln_l)
        self.ff1 = nn.Linear(hidden_size, intermediate_size)
        self.ff2 = nn.Linear(intermediate_size, hidden_size)

        self.exact_nonlinear = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_running_stats: bool = False,
    ) -> torch.Tensor:

        normed = self.ln1(hidden_states, use_running_stats=use_running_stats)
        attn_out = self.attention(normed, attention_mask, use_running_stats=use_running_stats)
        attn_out = self.attn_output(attn_out)
        x = hidden_states + attn_out

        normed2 = self.ln2(x, use_running_stats=use_running_stats)
        ff_hidden = self.ff1(normed2)
        if self.exact_nonlinear:
            ff_act = F.gelu(ff_hidden)
        elif use_running_stats and not self.training:
            import numpy as np

            from src.engines.mpc_gelu_secure import secure_gelu_piecewise_reference

            ff_np = ff_hidden.detach().cpu().numpy()
            gelu_np = secure_gelu_piecewise_reference(ff_np)
            ff_act = torch.tensor(gelu_np, dtype=ff_hidden.dtype, device=ff_hidden.device)
        else:
            ff_act = F.gelu(ff_hidden)
        ff_out = self.ff2(ff_act)
        return x + ff_out


class EncFormerGPT2LM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 1024,
        seq_len: int = 64,
        p: int = 5,
        c: float = 5.0,
        ln_l: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList(
            [
                EncFormerGPT2Layer(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    seq_len,
                    p=p,
                    c=c,
                    ln_l=ln_l,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = BatchLayerNorm(hidden_size, seq_len, l=ln_l)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_running_stats: bool = False,
    ) -> torch.Tensor:
        B, S = input_ids.shape

        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        hidden = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden = self.embedding_dropout(hidden)

        causal_mask = torch.triu(
            torch.full((S, S), torch.finfo(hidden.dtype).min, device=hidden.device),
            diagonal=1,
        )

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :].to(hidden.dtype)
            pad_mask = (1.0 - pad_mask) * torch.finfo(hidden.dtype).min
            combined_mask = causal_mask + pad_mask
        else:
            combined_mask = causal_mask

        for layer in self.layers:
            hidden = layer(hidden, combined_mask, use_running_stats=use_running_stats)

        hidden = self.ln_f(hidden, use_running_stats=use_running_stats)
        logits = self.lm_head(hidden)
        return logits


class EncFormerGPT2ForSequenceClassification(nn.Module):
    def __init__(
        self,
        num_labels: int = 2,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        intermediate_size: int = 3072,
        vocab_size: int = 50257,
        max_position_embeddings: int = 1024,
        seq_len: int = 64,
        p: int = 5,
        c: float = 5.0,
        ln_l: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList(
            [
                EncFormerGPT2Layer(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    seq_len,
                    p=p,
                    c=c,
                    ln_l=ln_l,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = BatchLayerNorm(hidden_size, seq_len, l=ln_l)

        self.classifier = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_running_stats: bool = False,
    ) -> torch.Tensor:
        B, S = input_ids.shape

        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        hidden = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden = self.embedding_dropout(hidden)

        causal_mask = (
            torch.triu(
                torch.full((S, S), torch.finfo(hidden.dtype).min, device=hidden.device),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :].to(hidden.dtype)
            pad_mask = (1.0 - pad_mask) * torch.finfo(hidden.dtype).min
            combined_mask = causal_mask + pad_mask
        else:
            combined_mask = causal_mask

        for layer in self.layers:
            hidden = layer(hidden, combined_mask, use_running_stats=use_running_stats)

        hidden = self.ln_f(hidden, use_running_stats=use_running_stats)

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=-1).long() - 1
        else:
            seq_lengths = torch.full((B,), S - 1, device=hidden.device, dtype=torch.long)
        pooled = hidden[torch.arange(B, device=hidden.device), seq_lengths]

        logits = self.classifier(pooled)
        return logits


def load_from_gpt2_pretrained(
    model: EncFormerGPT2LM | EncFormerGPT2ForSequenceClassification,
    gpt2_name: str = "gpt2",
) -> None:

    from transformers import GPT2Model

    gpt2 = GPT2Model.from_pretrained(gpt2_name)
    sd_gpt2 = gpt2.state_dict()
    sd_model = model.state_dict()

    mapping = {}

    mapping["word_embeddings.weight"] = "wte.weight"
    mapping["position_embeddings.weight"] = "wpe.weight"

    mapping["ln_f.weight"] = "ln_f.weight"
    mapping["ln_f.bias"] = "ln_f.bias"

    for i in range(len(model.layers)):
        pfx_s = f"layers.{i}"
        pfx_g = f"h.{i}"

        mapping[f"{pfx_s}.ln1.weight"] = f"{pfx_g}.ln_1.weight"
        mapping[f"{pfx_s}.ln1.bias"] = f"{pfx_g}.ln_1.bias"

        mapping[f"{pfx_s}.ln2.weight"] = f"{pfx_g}.ln_2.weight"
        mapping[f"{pfx_s}.ln2.bias"] = f"{pfx_g}.ln_2.bias"

        mapping[f"{pfx_s}.attn_output.weight"] = f"{pfx_g}.attn.c_proj.weight"
        mapping[f"{pfx_s}.attn_output.bias"] = f"{pfx_g}.attn.c_proj.bias"

        mapping[f"{pfx_s}.ff1.weight"] = f"{pfx_g}.mlp.c_fc.weight"
        mapping[f"{pfx_s}.ff1.bias"] = f"{pfx_g}.mlp.c_fc.bias"
        mapping[f"{pfx_s}.ff2.weight"] = f"{pfx_g}.mlp.c_proj.weight"
        mapping[f"{pfx_s}.ff2.bias"] = f"{pfx_g}.mlp.c_proj.bias"

    loaded = 0
    for our_key, gpt2_key in mapping.items():
        if our_key in sd_model and gpt2_key in sd_gpt2:
            src = sd_gpt2[gpt2_key]
            dst_shape = sd_model[our_key].shape

            if src.shape != dst_shape and src.T.shape == dst_shape:
                src = src.T
            if sd_model[our_key].shape == src.shape:
                sd_model[our_key] = src
                loaded += 1
            else:
                print(f"  Shape mismatch: {our_key} {dst_shape} vs {gpt2_key} {src.shape}")
        else:
            if our_key not in sd_model:
                print(f"  Missing in model: {our_key}")
            if gpt2_key not in sd_gpt2:
                print(f"  Missing in GPT-2: {gpt2_key}")

    for i in range(len(model.layers)):
        fused_key = f"h.{i}.attn.c_attn.weight"
        fused_bias_key = f"h.{i}.attn.c_attn.bias"
        if fused_key in sd_gpt2:
            w = sd_gpt2[fused_key]
            b = sd_gpt2[fused_bias_key]
            hs = w.shape[0]

            wq, wk, wv = w[:, :hs], w[:, hs : 2 * hs], w[:, 2 * hs :]
            bq, bk, bv = b[:hs], b[hs : 2 * hs], b[2 * hs :]

            pfx = f"layers.{i}.attention"
            for name, tensor in [
                (f"{pfx}.query.weight", wq.T),
                (f"{pfx}.query.bias", bq),
                (f"{pfx}.key.weight", wk.T),
                (f"{pfx}.key.bias", bk),
                (f"{pfx}.value.weight", wv.T),
                (f"{pfx}.value.bias", bv),
            ]:
                if name in sd_model and sd_model[name].shape == tensor.shape:
                    sd_model[name] = tensor
                    loaded += 1

    model.load_state_dict(sd_model, strict=False)
    print(f"[EncFormerModel] Loaded {loaded}/{len(mapping) + len(model.layers) * 6} weight tensors from {gpt2_name}")

    if hasattr(model, "lm_head"):
        model.lm_head.weight = model.word_embeddings.weight


def load_from_bert_pretrained(
    model: EncFormerBertForSequenceClassification,
    bert_name: str = "bert-base-uncased",
) -> None:

    from transformers import BertModel

    bert = BertModel.from_pretrained(bert_name)
    sd_bert = bert.state_dict()
    sd_model = model.state_dict()

    mapping = {}

    mapping["word_embeddings.weight"] = "embeddings.word_embeddings.weight"
    mapping["position_embeddings.weight"] = "embeddings.position_embeddings.weight"
    mapping["token_type_embeddings.weight"] = "embeddings.token_type_embeddings.weight"
    mapping["embedding_ln.weight"] = "embeddings.LayerNorm.weight"
    mapping["embedding_ln.bias"] = "embeddings.LayerNorm.bias"

    mapping["pooler.weight"] = "pooler.dense.weight"
    mapping["pooler.bias"] = "pooler.dense.bias"

    for i in range(len(model.layers)):
        pfx_s = f"layers.{i}"
        pfx_b = f"encoder.layer.{i}"

        mapping[f"{pfx_s}.attention.query.weight"] = f"{pfx_b}.attention.self.query.weight"
        mapping[f"{pfx_s}.attention.query.bias"] = f"{pfx_b}.attention.self.query.bias"
        mapping[f"{pfx_s}.attention.key.weight"] = f"{pfx_b}.attention.self.key.weight"
        mapping[f"{pfx_s}.attention.key.bias"] = f"{pfx_b}.attention.self.key.bias"
        mapping[f"{pfx_s}.attention.value.weight"] = f"{pfx_b}.attention.self.value.weight"
        mapping[f"{pfx_s}.attention.value.bias"] = f"{pfx_b}.attention.self.value.bias"

        mapping[f"{pfx_s}.attn_output.weight"] = f"{pfx_b}.attention.output.dense.weight"
        mapping[f"{pfx_s}.attn_output.bias"] = f"{pfx_b}.attention.output.dense.bias"

        mapping[f"{pfx_s}.ln1.weight"] = f"{pfx_b}.attention.output.LayerNorm.weight"
        mapping[f"{pfx_s}.ln1.bias"] = f"{pfx_b}.attention.output.LayerNorm.bias"

        mapping[f"{pfx_s}.ff1.weight"] = f"{pfx_b}.intermediate.dense.weight"
        mapping[f"{pfx_s}.ff1.bias"] = f"{pfx_b}.intermediate.dense.bias"
        mapping[f"{pfx_s}.ff2.weight"] = f"{pfx_b}.output.dense.weight"
        mapping[f"{pfx_s}.ff2.bias"] = f"{pfx_b}.output.dense.bias"

        mapping[f"{pfx_s}.ln2.weight"] = f"{pfx_b}.output.LayerNorm.weight"
        mapping[f"{pfx_s}.ln2.bias"] = f"{pfx_b}.output.LayerNorm.bias"

    loaded = 0
    for our_key, bert_key in mapping.items():
        if our_key in sd_model and bert_key in sd_bert:
            if sd_model[our_key].shape == sd_bert[bert_key].shape:
                sd_model[our_key] = sd_bert[bert_key]
                loaded += 1
            else:
                print(f"  Shape mismatch: {our_key} {sd_model[our_key].shape} vs {bert_key} {sd_bert[bert_key].shape}")
        else:
            if our_key not in sd_model:
                print(f"  Missing in model: {our_key}")
            if bert_key not in sd_bert:
                print(f"  Missing in BERT: {bert_key}")

    model.load_state_dict(sd_model, strict=False)
    print(f"[EncFormerModel] Loaded {loaded}/{len(mapping)} weight tensors from {bert_name}")


def set_exact_nonlinear(model: nn.Module, enabled: bool = True) -> int:

    n = 0
    for mod in model.modules():
        if isinstance(mod, (BPMaxAttention, BatchLayerNorm, EncFormerBertLayer, EncFormerGPT2Layer)):
            mod.exact_nonlinear = enabled
            n += 1
    return n


def export_running_denominators(model: nn.Module) -> Dict[str, np.ndarray]:

    denoms = {}
    for i, layer in enumerate(model.layers):
        rd_attn = layer.attention.running_denominator.detach().cpu().numpy()
        denoms[f"layer_{i}_bpmax_running_denominator"] = rd_attn

        rd_ln1 = layer.ln1.running_denominator.detach().cpu().numpy()
        rd_ln2 = layer.ln2.running_denominator.detach().cpu().numpy()
        denoms[f"layer_{i}_ln1_running_denominator"] = rd_ln1
        denoms[f"layer_{i}_ln2_running_denominator"] = rd_ln2

    return denoms


def infer_model_config_name_from_metadata(
    *,
    model_type: str,
    hidden_size: int,
    num_layers: int,
    seq_len: int,
) -> str | None:

    from src.models.model_config import get_config

    candidates = ("gpt2-base",) if model_type == "gpt2" else ("bert-base", "bert-large")
    for name in candidates:
        cfg = get_config(name)
        if cfg.d_model == hidden_size and cfg.num_layers == num_layers and cfg.m == seq_len:
            return name
    return None


def infer_model_config_name(model: nn.Module) -> str | None:

    model_type = "gpt2" if isinstance(model, (EncFormerGPT2LM, EncFormerGPT2ForSequenceClassification)) else "bert"
    return infer_model_config_name_from_metadata(
        model_type=model_type,
        hidden_size=getattr(model, "hidden_size"),
        num_layers=len(model.layers),
        seq_len=getattr(model, "seq_len"),
    )


def save_checkpoint(
    model: nn.Module,
    output_dir: str,
    task: str = "",
) -> None:

    import os

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    denoms = export_running_denominators(model)
    np.savez(os.path.join(output_dir, "running_denominators.npz"), **denoms)

    is_gpt2_lm = isinstance(model, EncFormerGPT2LM)
    is_gpt2_cls = isinstance(model, EncFormerGPT2ForSequenceClassification)
    if is_gpt2_cls:
        model_type = "gpt2_cls"
    elif is_gpt2_lm:
        model_type = "gpt2"
    else:
        model_type = "bert"
    config = {
        "model_type": model_type,
        "hidden_size": model.hidden_size,
        "num_layers": len(model.layers),
        "seq_len": model.seq_len,
        "task": task,
    }
    if is_gpt2_lm:
        config["vocab_size"] = model.vocab_size
    if hasattr(model, "num_labels"):
        config["num_labels"] = model.num_labels
    import json

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Checkpoint] Saved to {output_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, np.ndarray]]:

    import json
    import os

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        config = json.load(f)

    model_type = config.get("model_type", "bert")
    if model_type == "gpt2_cls":
        model = EncFormerGPT2ForSequenceClassification(
            num_labels=config.get("num_labels", 2),
            hidden_size=config.get("hidden_size", 768),
            num_layers=config["num_layers"],
            seq_len=config["seq_len"],
        )
    elif model_type == "gpt2":
        model = EncFormerGPT2LM(
            vocab_size=config.get("vocab_size", 50257),
            hidden_size=config.get("hidden_size", 768),
            num_layers=config["num_layers"],
            seq_len=config["seq_len"],
        )
    else:
        model = EncFormerBertForSequenceClassification(
            num_labels=config.get("num_labels", 2),
            num_layers=config["num_layers"],
            seq_len=config["seq_len"],
        )
    model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_dir, "model.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    denoms = {}
    rd_path = os.path.join(checkpoint_dir, "running_denominators.npz")
    if os.path.exists(rd_path):
        data = np.load(rd_path, allow_pickle=False)
        for k in data.files:
            denoms[k] = data[k]

    return model, denoms


def extract_layer_weights(
    model: nn.Module,
    layer_idx: int = 0,
) -> Dict[str, np.ndarray]:

    layer = model.layers[layer_idx]

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64)

    weights = {}

    weights["WQ"] = to_np(layer.attention.query.weight.T)
    weights["WK"] = to_np(layer.attention.key.weight.T)
    weights["WV"] = to_np(layer.attention.value.weight.T)
    weights["bQ"] = to_np(layer.attention.query.bias)
    weights["bK"] = to_np(layer.attention.key.bias)
    weights["bV"] = to_np(layer.attention.value.bias)

    weights["WO"] = to_np(layer.attn_output.weight.T)
    weights["bO"] = to_np(layer.attn_output.bias)

    weights["ln1_w"] = to_np(layer.ln1.weight)
    weights["ln1_b"] = to_np(layer.ln1.bias)

    weights["W1"] = to_np(layer.ff1.weight.T)
    weights["b1"] = to_np(layer.ff1.bias)
    weights["W2"] = to_np(layer.ff2.weight.T)
    weights["b2"] = to_np(layer.ff2.bias)

    weights["ln2_w"] = to_np(layer.ln2.weight)
    weights["ln2_b"] = to_np(layer.ln2.bias)

    return weights
