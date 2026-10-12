#!/bin/sh
# Demo Task 2 dalam mode headless, untuk verifikasi otomatis.
# Ada sebagai file karena path dataset mengandung spasi dan shell di mesin ini
# menduplikasi karakter pada perintah panjang.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=reports/demo.txt
mkdir -p reports

{
  echo "=== pfv clip: klip jatuh ==="
  .venv/bin/pfv clip "Subject.1/Fall backwards" --no-window
  echo "exit=$?"

  echo
  echo "=== pfv clip: ADL berlutut (pengganggu utama) ==="
  .venv/bin/pfv clip "Subject.5/Kneel" --no-window --max-frames 20
  echo "exit=$?"

  echo
  echo "=== pfv clip: folder beranomali penamaan ==="
  .venv/bin/pfv clip "Subject.2/Fall backwards" --no-window --max-frames 10
  echo "exit=$?"

  echo
  echo "=== pfv clip: klip tidak ada ==="
  .venv/bin/pfv clip "Subject.99/Tidak ada" --no-window
  echo "exit=$?"
} >"$OUT" 2>&1

cat "$OUT"
