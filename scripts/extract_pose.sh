#!/bin/sh
# Jalankan pose atas seluruh CAUCAFall, simpan keypoint, laporkan statistik.
# Ada sebagai file karena path dataset mengandung spasi dan shell di mesin ini
# menduplikasi karakter pada perintah panjang.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=reports/extract_pose.txt
mkdir -p reports

.venv/bin/python scripts/extract_pose_caucafall.py \
  "${PFV_DATASET_ROOT:-Dataset CAUCAFall/CAUCAFall}" --skip-existing >"$OUT" 2>&1
echo "exit=$?" >>"$OUT"

tail -n 30 "$OUT"
