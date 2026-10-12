#!/bin/sh
# Pemeriksaan standar proyek. Ada sebagai file supaya command line tetap pendek;
# shell di mesin ini menduplikasi karakter pada perintah panjang.
#
# console_output_style=count membuat pytest mencetak "N/TOTAL" sebagai progres,
# sehingga jumlah test tetap terlihat meskipun baris ringkasan tidak muncul saat
# keluaran diarahkan ke file.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=reports/check.txt
mkdir -p reports

{
  .venv/bin/python -m pytest -q --tb=short -o console_output_style=count
  echo "pytest_exit=$?"
} >"$OUT" 2>&1

tail -n 40 "$OUT"
