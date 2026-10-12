#!/bin/sh
# Impor CAUCAFall. Ada sebagai file karena shell di mesin ini menduplikasi
# karakter pada perintah panjang, dan path dataset mengandung spasi.
set -u
cd "$(dirname "$0")/.." || exit 1

ROOT="${PFV_DATASET_ROOT:-Dataset CAUCAFall/CAUCAFall}"
OUT=reports/import.txt
mkdir -p reports

.venv/bin/python scripts/import_caucafall.py "$ROOT" >"$OUT" 2>&1
echo "exit=$?" >>"$OUT"

cat "$OUT"
