#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$root/release}"
name="EncFormer-v1.0.0"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out" "$tmp/$name"
out="$(cd "$out" && pwd)"
manifest=(
  .dockerignore .gitattributes .github .gitignore .zenodo.json
  CHECKSUMS.sha256 CITATION.cff Dockerfile LICENSE README.md
  assets checkpoints data pyproject.toml requirements.txt scripts src tests third_party
)
(cd "$root" && tar --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.log' --exclude='*.o' --exclude='*.so' --exclude='*.a' --exclude='build' --exclude='CMakeFiles' --exclude='CMakeCache.txt' --exclude='cmake_install.cmake' --exclude='install_manifest.txt' -cf - "${manifest[@]}") | tar -xf - -C "$tmp/$name"
(cd "$tmp" && zip -qr "$out/$name.zip" "$name")
(cd "$out" && sha256sum "$name.zip" > "$name.zip.sha256")
du -h "$out/$name.zip"
cat "$out/$name.zip.sha256"
