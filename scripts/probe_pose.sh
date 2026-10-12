#!/bin/sh
# Probe pose atas klip CAUCAFall nyata. Ada sebagai file karena shell di mesin
# ini menduplikasi karakter pada perintah panjang.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=reports/probe_pose.txt
mkdir -p reports

.venv/bin/python scripts/probe_pose.py >"$OUT" 2>&1
echo "exit=$?" >>"$OUT"

cat "$OUT"
