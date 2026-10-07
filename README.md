# EncFormer

![IEEE TDSC](https://img.shields.io/badge/IEEE-TDSC-00629B?logo=ieee&logoColor=white)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![EncFormer](assets/EncFormer-Profile.png)

Official implementation of **EncFormer: Secure and Efficient Transformer Inference over Encrypted Data**, IEEE Transactions on Dependable and Secure Computing.

## Requirements

- Ubuntu 22.04
- Python 3.10
- CUDA 12.1+
- GCC 11+
- NVIDIA GPU with compute capability 7.0+ and 40 GB+ memory

## Setup

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libgmp-dev libssl-dev libomp-dev python3.10 python3.10-venv python3-pip zip
python3.10 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install -r requirements.txt
```

## Build

```bash
bash scripts/build.sh 80
```

The argument is the CUDA architecture: `80` for A100, `86` for RTX A6000, `89` for L40S, `90` for H100.

## Checkpoint

The KD-distilled BERT-base SST-2 checkpoint (`checkpoints/encformer-sst2/`) is tracked with [Git LFS](https://git-lfs.com), so a normal `git clone` fetches it when `git lfs` is installed. If it is missing or appears as a small pointer file (no Git LFS installed, or the repository is over its LFS bandwidth quota), download it directly from the release instead:

```bash
wget https://github.com/Yufannnn/EncFormer/releases/download/v1.0.0/encformer-sst2.tar.gz
tar xzf encformer-sst2.tar.gz -C checkpoints/
```

`sha256sum -c CHECKSUMS.sha256` (below) verifies either way.

## Run

```bash
sha256sum -c CHECKSUMS.sha256
python scripts/check.py
python scripts/eval.py --quick --gpu 0
python scripts/eval.py --full --gpu 0
bash scripts/2pc.sh 40 12 0
```

`2pc.sh <idx> <layers> <gpu>` classifies one SST-2 validation sentence with the two-party protocol: the server
(Phantom GPU CKKS, evaluation keys only) and the client (secret key) run as separate processes and evaluate the
non-linear layers with EzPC/SCI. `python scripts/demo.py --text "..."` classifies a custom sentence.

## Docker

```bash
docker build -t encformer .
docker run --rm --gpus all --shm-size=8g encformer --idx 40 --gpu 0
```

## Package

```bash
bash scripts/package.sh release
```

This creates `release/EncFormer-v2.0.0.zip` and its SHA-256 file. Use `assets/EncFormer-Profile.png` as the repository social preview and record image.

## Structure

```text
assets/                    EncFormer visual identity
checkpoints/               BERT-base SST-2 checkpoint
data/                      local tokenizer and SST-2 validation set
scripts/                   build, validation, evaluation, packaging
src/                       EncFormer runtime
tests/                     validation suite
third_party/phantom-fhe/   GPU CKKS source
third_party/ezpc-sci/      native two-party bindings
third_party/ezpc/SCI/      SCI, SEAL, and Eigen source
```

## Citation

```bibtex
@article{zhu2026encformer,
  title   = {EncFormer: Secure and Efficient Transformer Inference over Encrypted Data},
  author  = {Zhu, Yufan and Jin, Chao and Aung, Khin Mi Mi and Xiao, Xiaokui},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  year    = {2026}
}
```

## License

EncFormer is released under the MIT License. Vendored dependencies retain their original licenses.
