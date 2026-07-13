from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str
    m: int
    d_model: int
    H: int
    d_ff: int
    num_layers: int
    nslots: int = 16384
    n1: int = 16
    causal: bool = False

    @property
    def d_h(self) -> int:

        return self.d_model // self.H

    @property
    def C(self) -> int:

        return self.nslots // self.m

    @property
    def d1(self) -> int:

        return 6 * self.C


BERT_BASE = ModelConfig(
    name="bert-base",
    m=128,
    d_model=768,
    H=12,
    d_ff=3072,
    num_layers=12,
    nslots=16384,
)

BERT_LARGE = ModelConfig(
    name="bert-large",
    m=128,
    d_model=1024,
    H=16,
    d_ff=4096,
    num_layers=24,
    nslots=32768,
)

GPT2_BASE = ModelConfig(
    name="gpt2-base",
    m=64,
    d_model=768,
    H=12,
    d_ff=3072,
    num_layers=12,
    nslots=16384,
    causal=True,
)

_CONFIGS = {
    "bert-base": BERT_BASE,
    "bert-large": BERT_LARGE,
    "gpt2-base": GPT2_BASE,
}


def get_config(name: str) -> ModelConfig:

    cfg = _CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown model config {name!r}. Available: {list(_CONFIGS.keys())}")
    return cfg
