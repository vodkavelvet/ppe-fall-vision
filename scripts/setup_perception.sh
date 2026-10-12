#!/bin/sh
# Pasang grup [perception] (torch, ultralytics, supervision) dan verifikasi MPS.
# Ada sebagai file karena shell di mesin ini menduplikasi karakter pada perintah
# panjang.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=reports/setup_perception.txt
mkdir -p reports models

{
  echo "=== pasang ==="
  PATH="$HOME/.local/bin:$PATH" uv pip install -e ".[dev,perception]" 2>&1 | tail -20
  echo "exit=$?"

  echo
  echo "=== verifikasi ==="
  .venv/bin/python scripts/verify_perception.py
  echo "exit=$?"
} >"$OUT" 2>&1

cat "$OUT"
