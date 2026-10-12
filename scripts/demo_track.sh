#!/bin/sh
# Demo Task 4 headless: berapa banyak ID berbeda yang muncul per klip nyata?
# Satu orang per frame, jadi jumlah ID yang sehat adalah 1. Lebih dari itu berarti
# track terputus lalu diberi identitas baru -- itu yang merusak lock pelanggaran
# di Task 9.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=reports/demo_track.txt
mkdir -p reports

{
  for clip in "Subject.1/Fall backwards" "Subject.3/Fall forward" "Subject.5/Kneel" "Subject.7/Walk" "Subject.9/Hop"; do
    echo "=== $clip ==="
    .venv/bin/pfv track --clip "$clip" --no-window 2>&1 | tail -n 2
  done
} >"$OUT" 2>&1

cat "$OUT"
