#!/bin/bash
set -euo pipefail
# Script untuk menjalankan deteksi jatuh dengan video demo

echo "═══════════════════════════════════════════════════"
echo "🎥 SISTEM DETEKSI JATUH - DEMO VIDEO"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Video: Subject 1 - Fall Backwards"
echo "Expected: ✅ Alarm akan muncul sekitar detik ke-6"
echo ""
echo "Tekan 'q' untuk keluar"
echo ""

cd -- "$(dirname -- "$0")"
.venv/bin/pfv fall --config config/fall_robust.yaml --source "Dataset CAUCAFall/CAUCAFall/Subject.1/Fall backwards/FallBackwardsS1.avi"

echo ""
echo "Sistem dimatikan. Terima kasih!"
